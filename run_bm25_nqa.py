import csv
import time
import re
import nltk
import numpy as np
from rank_bm25 import BM25Okapi

import sys

from convertor import convert_docs_in_dic

csv.field_size_limit(sys.maxsize)

BM25_FILE = "./data/output/bm25/bm25_predictions_without_answer.tsv"
BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_all.eval"

def compute_bm25(tokenized_query, story_id, paragraphs, n):
    tokenized_paragraphs = [paragraph.split(" ") \
                            for paragraph in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    best_p = bm25.get_top_n(tokenized_query, paragraphs, n=n)
    best_i = [p.split(" ")[0] for p in best_p]
    return best_i

def gather_bm25(dataset, n, attach_answer):
    predictions = dict()
    for story_id, story_details in dataset.items():
        paragraphs = ["{} {}".format(k,v) for k,v in dataset[story_id]['paragraphs'].items()]
        for query_id, query_details in story_details['queries'].items():
            query = "{} {} {}".format(query_details['query'],
                                        query_details['answer1'],
                                        query_details['answer2']) if \
                    attach_answer else query_details['query']
            tokenized_query = query.split(" ")
            predictions[query_id] = compute_bm25(
                tokenized_query, story_id,
                paragraphs, n)
    return predictions

def write_bm25_pred(dataset, n, attach_answer):
    with open(BM25_FILE, "w") as f:
        predictions = gather_bm25(dataset, n, attach_answer)
        for query_id, paragraphs in predictions.items():
            for i, p in enumerate(paragraphs):
                f.write("{}\t{}\t{}\n".format(query_id, p, i+1))

def main():
    dataset = convert_docs_in_dic(BOOK_EVAL_FILE)
    write_bm25_pred(dataset, n=20, attach_answer=False)

if __name__=="__main__":
    main()
