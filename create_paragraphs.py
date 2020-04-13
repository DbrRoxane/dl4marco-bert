import csv
import jsonlines
import nltk
import itertools
import linecache
import rouge as rouge_score
import numpy as np

RANKING_FILE = "./data/output/narrative_book_paragraphs_9avril/nqa_predictions.tsv"
BAUER_FILE = "./data/narrativeqa/bauer_format.jsonl"
MIN_FILE = "./data/narrativeqa/min_format.json"
BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_book.eval"

def find_and_convert(n):
    """
    Retrieve the n best paragraphs in a story based on a question
    And convert the data them to be processed by bauer and min models
    ranking_file -> str: the file output by run_nqa.py
    n -> int: the number of paragraph we want to retrieve
    """

    entries = {}
    bauer_style_file = jsonlines.open(BAUER_FILE, mode="w")
    min_style_file = jsonlines.open(MIN_FILE, mode="w")
    with open(RANKING_FILE, 'r') as r_f:
        ranking_reader = csv.reader(r_f, delimiter="\t")
        for row in ranking_reader:
            # len(row)==4 to ensure the line is complete
            if len(row) == 4 and eval(row[2]) in range(1, n+1):
                if eval(row[2]) == 1:
                    complete_id =  row[0].split("_")
                    story_id, query_id = complete_id[0], complete_id[1]
                    query_id = query_id.replace('q', '').strip()
                    paragraphs_ids = list()
                paragraphs_ids.append(row[1])
                if eval(row[2]) == n:
                    context, query, answer1, answer2 = \
                            extract_extra(story_id, query_id, paragraphs_ids)
                    if context:
                        entry= {'complete_id':complete_id,
                               'story_id':story_id,
                               'query_id':query_id,
                               'paragaphs_id':paragraphs_ids,
                               'query':query,
                               'context':context,
                               'answer1':answer1,
                               'answer2':answer2}
                        write_to_bauer(bauer_style_file, entry)
                        write_to_min(min_style_file, entry)
    bauer_style_file.close()
    min_style_file.close()

def extract_extra(story_id, query_id, paragraphs_id):
    book_eval_file = BOOK_EVAL_FILE
    answer1, answer2, query = None, None, None
    paragraphs = list()
    with open(book_eval_file, "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            if row[0] == story_id+"_q"+query_id and row[1] in paragraphs_id:
                if not answer1:
                    query = row[2]
                    answer1, answer2 = row[4], row[5]
                paragraphs.append(row[3])
                if len(paragraphs)==len(paragraphs_id):
                    context = "\n".join(paragraphs)
                    return context, query, answer1, answer2
    return None, None, None, None

def write_to_bauer(bauer_file, entry):
    bauer_file.write({'doc_num':entry['story_id'],
                      'summary':nltk.word_tokenize(entry['context'].lower()),
                      'ques':nltk.word_tokenize(entry['query'].lower()),
                      'answer1':nltk.word_tokenize(entry['answer1'].lower()),
                      'answer2':nltk.word_tokenize(entry['answer2'].lower()),
                      'commonsense':[]})

def extract_first_span(paragraph, subtext):
    n = len(subtext)
    start_index = [i for i,x in enumerate(paragraph) if x in subtext[0]]
    for i in start_index:
        if paragraph[i:i+n]==subtext:
            return i, i+n-1

def extract_answer(paragraph, answer1, answer2, max_n, threshold=0.3):
    previous_max_score = 0
    subtext = paragraph
    rouge = rouge_score.Rouge()
    max_n = min(max_n, len(paragraph))
    for i in reversed(range(1, max_n+1)):
        n_grams = [" ".join(n_gram) for n_gram in nltk.ngrams(subtext, i)]
        scores = [score['rouge-l']['f']
                  for score in rouge.get_scores(n_grams, [answer1]*len(n_grams))]
        scores += [score['rouge-l']['f']
                   for score in rouge.get_scores(n_grams, [answer2]*len(n_grams))]
        max_index_score = np.argmax(np.array(scores))
        max_score = scores[max_index_score]
        if previous_max_score > max_score or max_score == 0:
            if previous_max_score < 0.3 or max_score == 0:
                return []
            index_start, index_end = extract_first_span(paragraph, subtext)
            return {'text':subtext, 'word_start':index_start, 'word_end':index_end}
        subtext = nltk.word_tokenize(n_grams[max_index_score % len(n_grams)])
        previous_max_score = max_score

    if max_score < threshold:
        return []
    index_start, index_end = extract_first_span(paragraph, subtext)
    return {'text':subtext, 'word_start':index_start, 'word_end':index_end}


def write_to_min(min_file, entry):
    context = list()
    answers = list()
    final_answers = [entry['answer1'], entry['answer2']]
    paragraphs = entry['context'].split("\n")
    for paragraph in paragraphs : 
        p_tokenized = nltk.word_tokenize(paragraph.replace('.',''))
        #remove point because generate errors because oof rouge code
        context.append(p_tokenized)
        answer_dic = (extract_answer(p_tokenized, entry['answer1'], entry['answer2'], 20))
        if answer_dic != []:
            answer_text = answer_dic['text']
            final_answers.append(answer_text)
        answers.append(answer_dic)
    min_file.write({'id':"_".join(entry['complete_id']),
                    'question' :entry['query'],
                    'context':context,
                    'answers':answers,
                    'final_answers':final_answers})


def main():
    find_and_convert(3)

if __name__=="__main__":
    main()
