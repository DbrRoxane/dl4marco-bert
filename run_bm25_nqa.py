import csv
import time
import re
import nltk
import numpy as np
from rank_bm25 import BM25Okapi

import sys

csv.field_size_limit(sys.maxsize)

BM25_FILE = "./data/narrativeqa/bm25_predictions.tsv"
BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_all.eval"

def get_complete_story(query_id):
    start = time.time()
    book = []
    with open(BOOK_EVAL_FILE, "r") as f:
        csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            if len(row)==6 and query_id == row[0]:
                book.append(nltk.word_tokenize(row[3].lower()))
    end = time.time()
    print("{} required to extract story".format(end-start))
    return book

def compute_bm25(query, story_id, tokenized_corpus, n):
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_corpus)
    best_p = bm25.get_top_n(tokenized_query, tokenized_corpus, n=n)
    print(query)
    for p in best_p:
        print(" ".join(p))
    best_i = [story_id+"_p" +
              re.findall("\((.*?)\)", " ".join(p))[0].strip() for p in best_p]
    print(best_i)
    return best_i

def gather_bm25(n, attach_answer):
    with open(BOOK_EVAL_FILE, "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        predictions = dict()
        tokenized_books = dict()
        for row in csv_reader:
            query_id = row[0]
            story_id = query_id.split("_")[0]
            query = "{} {} {}".format(row[2],row[4],row[5]) if \
                    attach_answer else row[2]
            query.lower()
            if query_id not in predictions.keys():
                if story_id not in tokenized_books.keys():
                    tokenized_books[story_id] = get_complete_story(query_id)
                predictions[query_id] = compute_bm25(
                    query, story_id, tokenized_books[story_id], n)
    return predictions

def write_bm25_pred(n, attach_answer):
    with open(BM25_FILE, "w") as f:
        predictions = gather_bm25(n, attach_answer)
        for query_id,paragraphs in predictions.items():
            for i,p in enumerate(paragraphs):
                f.write("{}\t{}\t{}\n".format(query_id, p, i+1))

def main():
    write_bm25_pred(n=3, attach_answer=True)

if __name__=="__main__":
    main()
