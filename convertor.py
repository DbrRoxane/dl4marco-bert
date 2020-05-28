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
DOCUMENTS_FILE = "./data/narrativeqa/documents.csv"
SUMMARIES_FILE = "./data/narrativeqa/third_party/wikipedia/summaries.csv"

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
MIN_ALL_WITH_ANSWER_TRAIN = "./data/narrativeqa/min_all_with_answer_train.json"
MIN_ALL_WITH_ANSWER_DEV = "./data/narrativeqa/min_all_with_answer_dev.json"
MIN_ALL_WITH_ANSWER_TEST = "./data/narrativeqa/min_all_with_answer_test.json"
MIN_ALL_WITHOUT_ANSWER_TRAIN = "./data/narrativeqa/min_all_without_answer_train.json"
MIN_ALL_WITHOUT_ANSWER_DEV = "./data/narrativeqa/min_all_without_answer_dev.json"
MIN_ALL_WITHOUT_ANSWER_TEST = "./data/narrativeqa/min_all_without_answer_test.json"
MIN_BOOKS_WITH_ANSWER_TRAIN = "./data/narrativeqa/min_books_with_answer_train.json"
MIN_BOOKS_WITH_ANSWER_DEV = "./data/narrativeqa/min_books_with_answer_dev.json"
MIN_BOOKS_WITH_ANSWER_TEST = "./data/narrativeqa/min_books_with_answer_test.json"
MIN_BOOKS_WITHOUT_ANSWER_TRAIN = "./data/narrativeqa/min_books_without_answer_train.json"
MIN_BOOKS_WITHOUT_ANSWER_DEV = "./data/narrativeqa/min_books_without_answer_dev.json"
MIN_BOOKS_WITHOUT_ANSWER_TEST = "./data/narrativeqa/min_books_without_answer_test.json"
MIN_SUM_WITH_ANSWER_TRAIN = "./data/narrativeqa/min_sums_train.json"
MIN_SUM_WITH_ANSWER_DEV = "./data/narrativeqa/min_sums_dev.json"
MIN_SUM_WITH_ANSWER_TEST = "./data/narrativeqa/min_sums_test.json"

ANNOTATION_TRAIN_FILE = "./data/narrativeqa/amt_22mai_2booktrain.csv"
ANNOTATION_DEV_FILE ="./data/narrativeqa/amt_19mai_2bookdev.csv" 
def retrieve_doc_info(story_id):
    with open(DOCUMENTS_FILE, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            if row['document_id'] == story_id:
                return row['set'], row['kind']
        print("did not find story", story_id)

def retrieve_summary(story_id):
    with open(SUMMARIES_FILE, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            if row['document_id'] == story_id:
                return row['summary'].replace("\n", "")

def convert_docs_in_dic(file_name):
    with open(file_name, "r", encoding="ascii", errors="ignore") as f:
        dataset = {}
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            query_id = row[0]
            story_id = query_id.split("_")[0]
            if story_id not in dataset.keys():
                summary = retrieve_summary(story_id)
                train_dev_test, book_movie = retrieve_doc_info(story_id)
                dataset[story_id] = {'paragraphs': {}, 'queries':{},
                                     'summary' : summary,
                                     'set':train_dev_test, 'kind':book_movie}
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

    def find_and_convert(self, just_book, train_dev_test):
        """
        Retrieve the n best paragraphs in a story based on a question
        """

        entries = {}
        converted_file = self.open_file()
        for ranking_filename in self.ranking_filenames:
            with open(ranking_filename, 'r') as ranking_file:
                ranking_reader = csv.reader(ranking_file, delimiter="\t")
                for row in ranking_reader:
                    if len(row)==3 and eval(row[2]) in range(1, self.n+1):
                        if eval(row[2]) == 1:
                            query_id = row[0]
                            story_id, _ = query_id.split("_")
                            paragraphs_ids = list()
                        #detect and ignore repeated paragraphs in ranking
                        if row[1] in paragraphs_ids:
                            print("twice the same parahraph", query_id, row[1])
                        else:
                            paragraphs_ids.append(row[1])
                        select_book = self.dataset[story_id]['kind'] == 'gutenberg' if \
                                just_book else True
                        select_set = self.dataset[story_id]['set'] == train_dev_test

                        if eval(row[2]) == self.n and select_book and select_set : # and \
#                           len(set(paragraphs_ids)) == self.n:
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
        self.close_file(converted_file)

    def find_and_convert_from_summaries(self, train_dev_test):
        converted_file = self.open_file()
        for story_id, story_details in self.dataset.items():
            if story_details['set'] == train_dev_test:
                for query_id, query_details in story_details['queries'].items():
                    entry = {'story_id':story_id,
                             'query_id':query_id,
                             'query':query_details['query'],
                             'context':story_details['summary'],
                             'answer1':query_details['answer1'],
                             'answer2':query_details['answer2']
                            }
                    self.write_to_converted_file(converted_file, entry)
        self.close_file(converted_file)

    def open_file(self):
        pass

    def close_file(self, converted_file):
        pass

    def write_to_converted_file(self, converted_file, entry):
        pass

    def extract_query_details(self, story_id, query_id, paragraphs_id):
        context = [paragraph_str
                   for paragraph_id, paragraph_str in \
                        self.dataset[story_id]['paragraphs'].items() \
                   if paragraph_id in paragraphs_id]
        if len(context) != len(paragraphs_id):
            print("cannot retrieve passages", len(paragraphs_id), len(context), paragraphs_id)
            context = ""
        else:
            context = "\n".join(context)
        query, answer1, answer2 = self.dataset[story_id]['queries'][query_id].values()
        return context, query, answer1, answer2


class BauerConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset):
        super().__init__(ranking_filename, converted_filename, n, dataset)

    def open_file(self):
        return jsonlines.open(self.converted_filename, mode="w")

    def close_file(self, converted_file):
        converted_file.close()

    def write_to_converted_file(self, converted_file, entry):
        converted_file.write({'doc_num':entry['story_id'],
                              'summary':nltk.word_tokenize(entry['context'].lower()),
                              'ques':nltk.word_tokenize(entry['query'].lower()),
                              'answer1':nltk.word_tokenize(entry['answer1'].lower()),
                              'answer2':nltk.word_tokenize(entry['answer2'].lower()),
                              'commonsense':[]})


class MinConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset, rouge_threshold):

        super().__init__(ranking_filename, converted_filename, n, dataset)
        self.rouge_threshold = rouge_threshold
        self.already_exists = {}
    def open_file(self):
        return jsonlines.open(self.converted_filename, mode="w")

    def write_to_converted_file(self, converted_file, entry):
        if entry['query_id'] not in self.already_exists.keys():
            self.already_exists[entry['query_id']] = {
                'query':entry['query'],
                'final_answers':[entry['answer1'], entry['answer2']],
                'context':list(),
                'answers':list()}
        paragraphs = entry['context'].split("\n")
        new_paragraphs = [paragraph for paragraph in paragraphs \
                         if paragraph not in self.already_exists[
                             entry['query_id']]['context']]
        for paragraph in new_paragraphs:
            p_tokenized = nltk.word_tokenize(paragraph.replace('.',''))
            #remove point because generate errors because of rouge code
            answer_dic = self.find_likely_answer(p_tokenized,
                                                 entry['answer1'],
                                                 entry['answer2'])
            if answer_dic != []:
                answer_text = answer_dic[0]['text']
                self.already_exists[entry['query_id']]['final_answers'].append(answer_text)
            self.already_exists[entry['query_id']]['context'].append(p_tokenized)
            self.already_exists[entry['query_id']]['answers'].append(answer_dic)

    def close_file(self, converted_file):
        for k,v in self.already_exists.items():
            converted_file.write({'id':k,
                                  'question' :v['query'],
                                  'context':v['context'],
                                  'answers':v['answers'],
                                  'final_answers':v['final_answers']})

        converted_file.close()

    def match_first_span(self, paragraph, subtext):
        size_ngram = len(subtext)
        subtext = [sub.replace('`', '\'') for sub in subtext]
        paragraph = [par.replace('`', '\'') for par in paragraph]
        start_index = [i for i, x in enumerate(paragraph) if x in subtext[0]]
        for i in start_index:
            if paragraph[i:i+size_ngram] == subtext:
                return i, i+size_ngram-1

    def find_likely_answer(self, paragraph, answer1, answer2, max_n=20):
        """
        Knowing an answer, find spans in the paragraphs with high rouge score
        max_n is the biggest n-gram analyzed
        """

        previous_max_score = 0
        masked_paragraph = paragraph.copy()
        subtext = paragraph.copy()
        rouge = rouge_score.Rouge()
        max_n = min(max_n, len(paragraph))
        answers = []
        i = max_n
        test="no"
        while i > 0 :
        #for i in reversed(range(1, max_n+1)):
            n_grams = [" ".join(n_gram) for n_gram in nltk.ngrams(subtext, i)]
            scores = [score['rouge-l']['f']
                      for score in rouge.get_scores(n_grams, [answer1]*len(n_grams))]
            scores += [score['rouge-l']['f']
                       for score in rouge.get_scores(n_grams, [answer2]*len(n_grams))]
            max_index_score = np.argmax(np.array(scores))
            max_score = scores[max_index_score]

            #if the previous score was better than the actual, 
            #it means that we have a better span and no need to fo further
            #or if max_score=0, we know there is nothing interesting here
            if previous_max_score > max_score or max_score == 0:
                if previous_max_score < self.rouge_threshold or max_score == 0:
                    break
                index_start, index_end = self.match_first_span(masked_paragraph, subtext)
                answers.append({'text':" ".join(subtext), 'word_start':index_start, 'word_end':index_end})

                #once we find a good answer, we remove it from the initial paragraph
                #and rerun the exploration
                previous_max_score = 0
                i = max_n
                for j in range(index_start, index_end+1):
                    masked_paragraph[j] = "MASK"
                subtext = masked_paragraph.copy()
            else:
                subtext = nltk.word_tokenize(n_grams[max_index_score % len(n_grams)])
                previous_max_score = max_score
                i -= 1
        return answers



class AnnotationConvertor(Convertor):
    def __init__(self, ranking_filename, converted_filename, n, dataset):
        super().__init__(ranking_filename, converted_filename, n, dataset)
        self.already_writen = {}

    def open_file(self):
        annotation_file = open(self.converted_filename, "w")
        fieldnames = ['question_id', 'paragraph_id', 'question','answers','paragraph']
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

    def close_file(self, converted_file):
        converted_file.close()
def main():

    dataset = convert_docs_in_dic(BOOK_EVAL_FILE)
    print("Created dataset")

    ALL_RANKING_FILE = RANKING_BERT_WITH_ANSWER +\
            [RANKING_BM25, RANKING_TFIDF] +\
            RANKING_BERT_WITHOUT_ANSWER


    #====== MIN SUM ALL ANSWERS ======

    min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
                                         MIN_SUM_WITH_ANSWER_DEV+"_r6",
                                         3, dataset, rouge_threshold=0.6)
   min_with_answer_dev.find_and_convert_from_summaries(train_dev_test="valid")
   print("Created", MIN_SUM_WITH_ANSWER_DEV+"_r6")

    min_with_answer_test = MinConvertor(RANKING_BERT_WITH_ANSWER,
                                         MIN_SUM_WITH_ANSWER_TEST+"_r6",
                                         3, dataset, rouge_threshold=0.6)
    min_with_answer_test.find_and_convert_from_summaries(train_dev_test="test")
    print("Created", MIN_SUM_WITH_ANSWER_TEST+"_r6")

    min_with_answer_train = MinConvertor(RANKING_BERT_WITH_ANSWER,
                                         MIN_SUM_WITH_ANSWER_TRAIN+"_r6",
                                         3, dataset, rouge_threshold=0.6)
    min_with_answer_train.find_and_convert_from_summaries(train_dev_test="train")
    print("Created", MIN_SUM_WITH_ANSWER_TRAIN+"_r6")



    #====== MIN WITH ANSWER ALL ======


    #min_with_answer_train = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_TRAIN+"_r7",
    #                                     3, dataset, rouge_threshold=0.7)
    #min_with_answer_train.find_and_convert(just_book=False, train_dev_test="train")
    #print("Created", MIN_ALL_WITH_ANSWER_TRAIN+"_r7")
    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_DEV+"_r7",
    #                                     3, dataset, rouge_threshold=0.7)
    #min_with_answer_dev.find_and_convert(just_book=False, train_dev_test="valid")
    #print("Created", MIN_ALL_WITH_ANSWER_DEV+"_r7")

    #min_with_answer_train = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_TRAIN+"_r6",
    #                                     3, dataset, rouge_threshold=0.6)
    #min_with_answer_train.find_and_convert(just_book=False, train_dev_test="train")
    #print("Created", MIN_ALL_WITH_ANSWER_TRAIN+"_r6")

    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_DEV+"_r6",
    #                                     3, dataset, rouge_threshold=0.6)
    #min_with_answer_dev.find_and_convert(just_book=False, train_dev_test="valid")
    #print("Created", MIN_ALL_WITH_ANSWER_TRAIN+"_r6")

    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_DEV+"_several_answers_r5",
    #                                     3, dataset, rouge_threshold=0.5)
    #min_with_answer_dev.find_and_convert(just_book=False,train_dev_test="valid")
    #print("Created", MIN_ALL_WITH_ANSWER_DEV+"_several_answers_r5")

 
    #min_with_answer_test = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                     MIN_ALL_WITHOUT_ANSWER_TEST+"_bert_severalanswers_r5",
    #                                     10, dataset, rouge_threshold=0.5)
    #min_with_answer_test.find_and_convert(just_book=False, train_dev_test="test")
    #print("Created", MIN_ALL_WITHOUR_ANSWER_TEST+"_bert__severalanswers_r5")

    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                     MIN_ALL_WITHOUT_ANSWER_DEV+"_bert_severalanswers_r5",
    #                                     10, dataset, rouge_threshold=0.5)
    #min_with_answer_dev.find_and_convert(just_book=False, train_dev_test="valid")
    #print("Created", MIN_ALL_WITH_ANSWER_DEV+"_bert_severalanswers_r5")

    #min_with_answer_train = MinConvertor(ALL_RANKING_FILE,
    #                                    MIN_ALL_WITH_ANSWER_TRAIN+"_allrankingtech_severalanswers_r5",
    #                                    3, dataset, rouge_threshold=0.5)
    #min_with_answer_train.find_and_convert(just_book=False, train_dev_test="train")
    #print("Created", MIN_ALL_WITH_ANSWER_TEST+"_allrankingtech_severalanswers_r5")


    #min_with_answer_train = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                     MIN_ALL_WITH_ANSWER_TRAIN+"_several_answers_r5",
    #                                     3, dataset, rouge_threshold=0.5, sw=True)
    #min_with_answer_train.find_and_convert(just_book=False, train_dev_test="train")
    #print("Created", MIN_ALL_WITH_ANSWER_TRAIN+"_several_answers_r5")


    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                    MIN_ALL_WITH_ANSWER_DEV, 3, dataset)
    #min_with_answer_dev.find_and_convert_from_summaries(train_dev_test="valid")
    #print("Created", MIN_ALL_WITH_ANSWER_DEV)
    
    #min_with_answer_test = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                    MIN_ALL_WITH_ANSWER_TEST, 3, dataset)
    #min_with_answer_test.find_and_convert(just_book=False, train_dev_test="test")
    #print("Created", MIN_ALL_WITH_ANSWER_TEST)
    
    #====== MIN WITHOUT ANSWER ALL ======

    #min_without_answer_train = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                    MIN_ALL_WITHOUT_ANSWER_TRAIN, 3, dataset)
    #min_without_answer_train.find_and_convert(just_book=False, train_dev_test="train")
    #print("Created", MIN_ALL_WITHOUT_ANSWER_TRAIN)

    #min_without_answer_dev = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                    MIN_ALL_WITHOUT_ANSWER_DEV, 3, dataset)
    #min_without_answer_dev.find_and_convert(just_book=False, train_dev_test="valid")
    #print("Created", MIN_ALL_WITHOUT_ANSWER_DEV)
    
    #min_without_answer_test = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                    MIN_ALL_WITHOUT_ANSWER_TEST, 3, dataset)
    #min_without_answer_test.find_and_convert(just_book=False, train_dev_test="test")
    #print("Created", MIN_ALL_WITHOUT_ANSWER_TEST)
    

    #====== MIN WITH ANSWER BOOK ======
    #min_with_answer_train = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                    MIN_BOOKS_WITH_ANSWER_TRAIN, 3, dataset)
    #min_with_answer_train.find_and_convert(just_book=True, train_dev_test="train")
    #print("Created", MIN_BOOKS_WITHOUT_ANSWER_TRAIN)

    #min_with_answer_dev = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                    MIN_BOOKS_WITH_ANSWER_DEV, 3, dataset)
    #min_with_answer_dev.find_and_convert(just_book=True, train_dev_test="dev")
    #print("Created", MIN_BOOKS_WITH_ANSWER_DEV)
    
    #min_with_answer_test = MinConvertor(RANKING_BERT_WITH_ANSWER,
    #                                    MIN_BOOKS_WITH_ANSWER_TEST, 3, dataset)
    #min_with_answer_test.find_and_convert(just_book=True, train_dev_test="test")
    #print("Created", MIN_BOOKS_WITH_ANSWER_TEST)

    #====== MIN WITHOUT ANSWER BOOK ======

    #min_without_answer_dev = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
    #                                    MIN_BOOKS_WITHOUT_ANSWER_DEV, 3, dataset)
    #min_without_answer_dev.find_and_convert(just_book=True, train_dev_test="dev")
    #print("Created", MIN_BOOKS_WITHOUT_ANSWER_DEV)
    
   # min_without_answer_test = MinConvertor(RANKING_BERT_WITHOUT_ANSWER,
   #                                     MIN_BOOKS_WITHOUT_ANSWER_TEST, 3, dataset)
   # min_without_answer_test.find_and_convert(just_book=True, train_dev_test="test")
   # print("Created", MIN_BOOKS_WITHOUT_ANSWER_TEST)
    
    #all_ranking_files = RANKING_BERT_WITH_ANSWER + [RANKING_BM25]
    #+ RANKING_BERT_WITHOUT_ANSWER + \
    #        [RANKING_BM25, RANKING_TFIDF]
    #print(all_ranking_files)
    #annotator_train_convertor = AnnotationConvertor(all_ranking_files, ANNOTATION_TRAIN_FILE, 2, dataset)
    #annotator_train_convertor.find_and_convert(just_book=True, train_dev_test="train")
    #print("Created", ANNOTATION_TRAIN_FILE)

    #annotator_dev_convertor = AnnotationConvertor(all_ranking_files, ANNOTATION_DEV_FILE, 2, dataset)
    #annotator_dev_convertor.find_and_convert(just_book=True, train_dev_test="valid")
    #print("Created", ANNOTATION_DEV_FILE)




if __name__=="__main__":
    main()
