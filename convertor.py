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
RANKING_BERT_WITH_ANSWER = \
        ["./data/output/nqa_with_answer/nqa_predictions_with_answer0.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer1.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer2.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer3.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer4.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer5.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer6.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer7.tsv",
         "./data/output/nqa_with_answer/nqa_predictions_with_answer8.tsv",
        ]

RANKING_BERT_WITHOUT_ANSWER = \
        ["./data/output/nqa_without_answer/nqa_predictions_without_answer0.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer1.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer2.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer3.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer4.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer5.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer6.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer7.tsv",
         "./data/output/nqa_without_answer/nqa_predictions_without_answer8.tsv",
        ]

RANKING_BM25 = "./data/output/bm25/bm25_predictions.tsv"
RANKING_TFIDF = "./data/output/tfidf/tfidf_predictions.tsv"

BAUER_FILE_WITH_ANSWER = "./data/narrativeqa/bauer_with_answer_format.jsonl"
BAUER_FILE_WITHOUT_ANSWER = "./data/narrativeqa/bauer_without_answer_format.jsonl"
MIN_FILE_WITH_ANSWER = "./data/narrativeqa/min_with_answer_format.json"
MIN_FILE_WITHOUT_ANSWER = "./data/narrativeqa/min_without_answer_format.json"
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


class Convertor(object):
    """
    Convert best ranked paragraph to narrativeQA over summaries format
    Convertor is an abstract class of specific convertor for given format
    """
    def __init__(self, ranking_filenames, converted_filename, n, dataset):
        self.ranking_filenames = ranking_filenames
        self.converted_filename = converted_filename
        self.n = n
        self.dataset = dataset

    def find_and_convert(self):
        """
        Retrieve the n best paragraphs in a story based on a question
        """

        entries = {}
        converted_file = self.open_file()
        for ranking_filename in self.ranking_filenames:
            with open(ranking_filename, 'r') as ranking_file:
                ranking_reader = csv.reader(ranking_file, delimiter="\t")
                for row in ranking_reader:
                    if len(row) in [3, 4] and eval(row[2]) in range(1, self.n+1):
                        if eval(row[2]) == 1:
                            query_id = row[0]
                            story_id, _ = query_id.split("_")
                            paragraphs_ids = list()
                        #detect and ignore repeated paragraphs in ranking
                        if row[1] in paragraphs_ids:
                            print("ignore", query_id)
                        paragraphs_ids.append(row[1])
                        if eval(row[2]) == self.n and \
                           len(set(paragraphs_ids)) == self.n:
                            context, query, answer1, answer2 = \
                                    self.extract_query_details(
                                        story_id, query_id, paragraphs_ids)
                            if context:
                                entry = {'query_id':query_id,
                                         'story_id':story_id,
                                         'paragraphs_id':paragraphs_ids,
                                         'query':query,
                                         'context':context,
                                         'answer1':answer1,
                                         'answer2':answer2}
                                self.write_to_converted_file(converted_file, entry)
        converted_file.close()

    def open_file(self):
        pass

    def write_to_converted_file(self, converted_file, entry):
        pass


    def extract_query_details(self, story_id, query_id, paragraphs_id):
        context = [paragraph_str
                   for paragraph_id, paragraph_str in \
                        self.dataset[story_id]['paragraphs'].items() \
                   if paragraph_id in paragraphs_id]
        if len(context) != self.n:
            print(len(paragraphs_id), len(context), paragraphs_id)
        context = "\n".join(context)
        query, answer1, answer2 = self.dataset[story_id]['queries'][query_id].values()
        return context, query, answer1, answer2


class BauerConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset):
        super().__init__(ranking_filename, converted_filename, n, dataset)

    def open_file(self):
        return jsonlines.open(self.converted_filename, mode="w")

    def write_to_converted_file(self, converted_file, entry):
        converted_file.write({'doc_num':entry['story_id'],
                              'summary':nltk.word_tokenize(entry['context'].lower()),
                              'ques':nltk.word_tokenize(entry['query'].lower()),
                              'answer1':nltk.word_tokenize(entry['answer1'].lower()),
                              'answer2':nltk.word_tokenize(entry['answer2'].lower()),
                              'commonsense':[]})


class MinConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset):
        super().__init__(ranking_filename, converted_filename, n, dataset)

    def open_file(self):
        return jsonlines.open(self.converted_filename, mode="w")

    def write_to_converted_file(self, converted_file, entry):
        context = list()
        answers = list()
        final_answers = [entry['answer1'], entry['answer2']]
        paragraphs = entry['context'].split("\n")
        for paragraph in paragraphs : 
            p_tokenized = nltk.word_tokenize(paragraph.replace('.',''))
            #remove point because generate errors because of rouge code
            context.append(p_tokenized)
            answer_dic = self.find_likely_answer(p_tokenized,
                                                 entry['answer1'],
                                                 entry['answer2'])
            if answer_dic != []:
                answer_text = answer_dic['text']
                final_answers.append(answer_text)
            answers.append(answer_dic)
        converted_file.write({'id':entry['query_id'],
                              'question' :entry['query'],
                              'context':context,
                              'answers':answers,
                              'final_answers':final_answers})

    def match_first_span(self, paragraph, subtext):
        size_ngram = len(subtext)
        subtext = [sub.replace('`', '\'') for sub in subtext]
        paragraph = [par.replace('`', '\'') for par in paragraph]
        start_index = [i for i, x in enumerate(paragraph) if x in subtext[0]]
        for i in start_index:
            if paragraph[i:i+size_ngram] == subtext:
                return i, i+size_ngram-1

    def find_likely_answer(self, paragraph, answer1, answer2, max_n=20, threshold=0.5):
        """
        Knowing an answer, find spans in the paragraphs with high rouge score
        max_n is the biggest n-gram analyzed
        """

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
                if previous_max_score < threshold or max_score == 0:
                    return []
                index_start, index_end = self.match_first_span(paragraph, subtext)
                return {'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end}
            subtext = nltk.word_tokenize(n_grams[max_index_score % len(n_grams)])
            previous_max_score = max_score

        if max_score < threshold:
            return []
        index_start, index_end = self.match_first_span(paragraph, subtext)
        return {'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end}



class AnnotationConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset):
        super().__init__(ranking_filename, converted_filename, n, dataset)
        self.already_writen = {}

    def open_file(self):
        annotation_file = open(self.converted_filename, "w")
        fieldnames = ['question_id', 'paragraph_id', 'question','answers','parahraph']
        self.csv_writer = csv.DictWriter(annotation_file, delimiter="\t", fieldnames=fieldnames)
        self.csv_writer.writeheader()
        return annotation_file

    def write_to_converted_file(self, converted_file, entry):
        if entry['query_id'] not in self.already_writen.keys():
            self.already_writen[entry['query_id']] = []
        for i, paragraph_id in enumerate(entry['paragraphs_id']):
            if paragraph_id not in self.already_writen[entry['query_id']]:
                self.already_writen[entry['query_id']].append(paragraph_id)
                answers = "Answer 1 :" + entry['answer1'] + " Answer2 : " + entry['answer2']
                paragraph = entry['context'].split("\n")[i]
                self.csv_writer.writerow({'question_id':entry['query_id'],
                                          'paragraph_id':paragraph_id,
                                          'question' : entry['query'],
                                          'answers': answers,
                                          'paragraph': paragraph})

def main():
    dataset = convert_docs_in_dic(BOOK_EVAL_FILE)
    print("Created dataset")
    min_convertor_with_answer = MinConvertor(RANKING_BERT_WITH_ANSWER,
                                             MIN_FILE_WITH_ANSWER, 3, dataset)
    min_convertor_with_answer.find_and_convert()
    print("Created", MIN_FILE_WITH_ANSWER)
    min_convertor_no_answer = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
                                           MIN_FILE_WITHOUT_ANSWER, 3, dataset)
    min_convertor_no_answer.find_and_convert()
    print("Created", MIN_FILE_WITHOUT_ANSWER)
    all_ranking_files = RANKING_BERT_WITH_ANSWER + RANKING_BERT_WITHOUT_ANSWER + \
            [RANKING_BM25, RANKING_TFIDF]
    annotator_convertor = MinConvertor(all_ranking_files, ANNOTATION_FILE, 3, dataset)
    annotator_convertor.find_and_convert()
    print("Created", ANNOTATION_FILE)



if __name__=="__main__":
    main()
