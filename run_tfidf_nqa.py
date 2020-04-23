import csv
import time
import re
import nltk
import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv.field_size_limit(sys.maxsize)

TFIDF_FILE = "./data/narrativeqa/tfidf_predictions.tsv"
BOOK_EVAL_FILE = "./data/narrativeqa/narrativeqa_all.eval"

def get_complete_story(query_id):
    start = time.time()
    book = []
    with open(BOOK_EVAL_FILE, "r") as f:
        csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            if len(row)==6 and query_id == row[0]:
                book.append(row[3])
    end = time.time()
    print("{} required to extract story".format(end-start))
    return book

def retrieve_paragraph_id(story_id, paragraphs):
    return [story_id+"_p" +
            re.findall("\((.*?)\)", " ".join(p))[0].strip() for p in paragraphs]


def compute_tfidf(query, story_id, book, n):
    print(query)
    query_tfidf = book['vectorizer'].transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, book['tfidf']).flatten()
    best_index = np.flip(np.argsort(cosine_similarities), axis=0)[:n]
    best_p = [book['story'][i] for i in best_index]
    print(best_p)
    best_id = retrieve_paragraph_id(story_id, best_p)
    return best_id


def gather_tfidf(n, attach_answer):
    with open(BOOK_EVAL_FILE, "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        predictions = dict()
        books = dict()
        for row in csv_reader:
            query_id = row[0]
            story_id = query_id.split("_")[0]
            query = "{} {} {}".format(row[2],row[4],row[5]) if \
                    attach_answer else row[2]
            query.lower()
            if query_id not in predictions.keys():
                if story_id not in books.keys():
                    story = get_complete_story(query_id)
                    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize,
                                                ngram_range=(1,2))
                    docs_tfidf = vectorizer.fit_transform(story)
                    books[story_id] = {'story':story,
                                       'vectorizer':vectorizer,
                                       'tfidf':docs_tfidf}
                predictions[query_id] = compute_tfidf(
                    query, story_id, books[story_id], n)
    return predictions

def write_tfidf_pred(n, attach_answer):
    with open(TFIDF_FILE, "w") as f:
        predictions = gather_tfidf(n, attach_answer)
        for query_id,paragraphs in predictions.items():
            for i,p in enumerate(paragraphs):
                f.write("{}\t{}\t{}\n".format(query_id, p, i))

def main():
    write_tfidf_pred(n=3, attach_answer=True)

if __name__=="__main__":
    main()
