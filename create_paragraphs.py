import csv
import jsonlines
import nltk
import itertools
import linecache
import rouge as rouge_score
import numpy as np
import time
import sys

csv.field_size_limit(sys.maxsize)

BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_all.eval"
RANKNG_BERT_WITH_ANSWER = ["./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer0.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer1.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer2.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer3.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer4.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer5.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer6.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer7.tsv",
                "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer8.tsv",
                ]

RANKNG_BERT_WITHOUT_ANSWER = \
        ["./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer0.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer1.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer2.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer3.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer4.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer5.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer6.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer7.tsv",
         "./data/output/nqa_with_answer_24avril/nqa_predictions_with_answer8.tsv",
         ]
RANKING_BM25 = "./data/output/bm25/bm25_predictions.tsv"
RANKING_TFIDF = "./data/output/tfidf/tfidf_predictions.tsv" 

BAUER_FILE = "./data/narrativeqa/bauer_without_answer_format.jsonl"
MIN_FILE = "./data/narrativeqa/min_without_answer_format.json"
ANNOTATION_FILE = "./data/narrativeqa/amt.csv"


def convert_docs_in_dic(file_name):
    with open(file_name, "r", encoding="ascii", errors="ignore") as f:
        dataset = {}
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            query_id = row[0]
            story_id = query_id.split("_")[0]
            if story_id not in dataset.keys():
                dataset[story_id] = {'paragraphs': {}, 'queries':{}}
            paragraph_id = row[1]
            if paragraph_id not in dataset[story_id]['paragraphs'].keys():
                pargraph = row[3]
                dataset[story_id]['paragraphs'][paragraph_id] = pargraph
            if query_id not in dataset[story_id]['queries'].keys():
                query = row[2]
                answer1, answer2 = row[4], row[5]
                dataset[story_id]['queries'][query_id] = {
                    'query': query,
                    'answer1' : answer1,
                    'answer2' : answer2}
    return dataset

def find_and_convert(dataset, n, bauer, hardem, annotation):
    """
    Retrieve the n best paragraphs in a story based on a question
    And convert the data them to be processed by bauer and min models
    ranking_file -> str: the file output by run_nqa.py
    n -> int: the number of paragraph we want to retrieve
    """

    entries = {}

    if bauer:
        bauer_style_file = jsonlines.open(BAUER_FILE, mode="w")
    if hardem:
        min_style_file = jsonlines.open(MIN_FILE, mode="w")
    if annotation:
        annotation_file = open(ANNOTATION_FILE, "w")
        fieldnames = ['question_id', 'paragraph_id', 'question','answers','parahraph']
        annotation_file_csv = csv.DictWriter(annotation_file, delimiter="\t", fieldnames=fieldnames)
        already_writen = {}

    for ranking_file in RANKNG_BERT_WITHOUT_ANSWER:
        with open(ranking_file, 'r') as r_f:
            ranking_reader = csv.reader(r_f, delimiter="\t")
            for row in ranking_reader:
                if len(row) in [3,4] and eval(row[2]) in range(1, n+1):
                    if eval(row[2]) == 1:
                        query_id =  row[0]
                        story_id, _ = query_id.split("_")
                        paragraphs_ids = list()
                    if row[1] in paragraphs_ids:
                        print("ignore", query_id)
                    paragraphs_ids.append(row[1])
                    if eval(row[2]) == n and len(set(paragraphs_ids))==n:
                        context, query, answer1, answer2 = \
                                extract_extra(dataset, story_id, query_id, paragraphs_ids)
                        if context:
                            entry= {'query_id':query_id,
                                    'story_id':story_id,
                                    'paragraphs_id':paragraphs_ids,
                                    'query':query,
                                    'context':context,
                                    'answer1':answer1,
                                    'answer2':answer2}
                            if bauer:
                                write_to_bauer(bauer_style_file, entry)
                            if hardem:
                                write_to_min(min_style_file, entry)
                            if annotation:
                                write_to_annotation(annotation_file_csv,
                                                    entry, already_writen)
    if bauer:
        bauer_style_file.close()
    if hardem:
        min_style_file.close()
    if annotation:
        annotation_file.close()


def extract_extra(dataset, story_id, query_id, paragraphs_id):
    context = [paragraph_str
               for paragraph_id, paragraph_str in \
                    dataset[story_id]['paragraphs'].items() \
               if paragraph_id in paragraphs_id]
    if len(context) != 3:
        print(len(paragraphs_id), len(context), paragraphs_id, 
              len(dataset[story_id]['paragraphs'].values()))
    context = "\n".join(context)
    #print(dataset[story_id]['queries'][query_id])
    query, answer1, answer2 = dataset[story_id]['queries'][query_id].values()
    return context, query, answer1, answer2

def write_to_annotation(annotation_file_csv, entry, already_writen):
    if entry['query_id'] not in already_writen.keys():
        already_writen[entry['query_id']] = []
    for i, paragraph_id in enumerate(entry['paragraphs_id']):
        if paragraph_id not in already_writen[entry['query_id']]:
            already_writen[entry['query_id']].append(paragraph_id)
            annotation_file_csv.writerow({'question_id':entry['query_id'],
                           'paragraph_id':paragraph_id,
                           'question' : entry['query'],
                           'answers': "Answer 1 :"+entry['answer1']+
                                          " Answer2 : "+entry['answer2'],
                           'parahraph':entry['context'].split("\n")[i]})


def write_to_bauer(bauer_file, entry):
    bauer_file.write({'doc_num':entry['story_id'],
                      'summary':nltk.word_tokenize(entry['context'].lower()),
                      'ques':nltk.word_tokenize(entry['query'].lower()),
                      'answer1':nltk.word_tokenize(entry['answer1'].lower()),
                      'answer2':nltk.word_tokenize(entry['answer2'].lower()),
                      'commonsense':[]})

def extract_first_span(paragraph, subtext):
    n = len(subtext)
    s = []
    for sub in subtext:
        s.append(sub.replace('`','\''))
    subtext = s
    p = []
    for par in paragraph:
        p.append(par.replace('`','\''))
    paragraph = p
    start_index = [i for i,x in enumerate(paragraph) if x in subtext[0]]
    #print(paragraph, subtext, start_index)
    for i in start_index:
        if paragraph[i:i+n]==subtext:
            return i, i+n-1

def extract_answer(paragraph, answer1, answer2, max_n, threshold=0.5):
    previous_max_score = 0
    subtext = paragraph
    rouge = rouge_score.Rouge()
    max_n = min(max_n, len(paragraph))
    for i in reversed(range(1, max_n+1)):
        n_grams = [" ".join(n_gram).replace('`', '\'')
                   for n_gram in nltk.ngrams(subtext, i)]
        scores = [score['rouge-l']['f']
                  for score in rouge.get_scores(n_grams, [answer1]*len(n_grams))]
        scores += [score['rouge-l']['f']
                   for score in rouge.get_scores(n_grams, [answer2]*len(n_grams))]
        max_index_score = np.argmax(np.array(scores))
        max_score = scores[max_index_score]
        if previous_max_score > max_score or max_score == 0:
            if previous_max_score < threshold or max_score == 0:
                return []
            index_start, index_end = extract_first_span(paragraph, subtext)
            return {'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end}
        subtext = nltk.word_tokenize(n_grams[max_index_score % len(n_grams)])
        previous_max_score = max_score

    if max_score < threshold:
        return []
    index_start, index_end = extract_first_span(paragraph, subtext)
    return {'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end}


def write_to_min(min_file, entry):
    context = list()
    answers = list()
    final_answers = [entry['answer1'], entry['answer2']]
    paragraphs = entry['context'].split("\n")
    for paragraph in paragraphs : 
        p_tokenized = nltk.word_tokenize(paragraph.replace('.',''))
        #remove point because generate errors because oof rouge code
        context.append(p_tokenized)
        answer_dic = extract_answer(p_tokenized, entry['answer1'], entry['answer2'], 20)
        if answer_dic != []:
            answer_text = answer_dic['text']
            final_answers.append(answer_text)
        answers.append(answer_dic)
    min_file.write({'id':entry['query_id'],
                    'question' :entry['query'],
                    'context':context,
                    'answers':answers,
                    'final_answers':final_answers})


def main():
    dataset = convert_docs_in_dic(BOOK_EVAL_FILE)
    find_and_convert(dataset, n=3, bauer=True, hardem=True, annotation=False)

if __name__=="__main__":
    main()
