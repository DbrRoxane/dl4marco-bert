import textwrap
import csv
import re

def chunk_story(story_str, chunk_size):
    return textwrap.wrap(re.sub('\s+', ' ',story_str), chunk_size)

def chunk_story_paragraphs(story_id, data_dir):
    story_file = NQA_DIR + "tmp/" + story_id + ".content"
    with open(story_file, 'r', encoding="utf-8", errors="ignore") as f:
        story_str = f.read()
        chunks =  [re.sub('\s+', ' ',paragraph)
                   for paragraph in story_str.split("\n\n")]
    return chunks

def is_book(data_dir, story_id):
    with open(data_dir+"documents.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        book = [row[2]=="gutenberg" for row in csv_reader if row[0]==story_id][0]
    return book

def extract_data_entries(data_dir):
    entries = list()
    with open(data_dir+"qaps.csv", "r") as qa_file:
        csv_reader = csv.reader(qa_file, delimiter=',')
        next(csv_reader)
        stories_chunked = dict()
        for question_idx, question_set in enumerate(csv_reader):
            story_id = question_set[0]
            question_str = question_set[2]
            answer_1, answer_2 = question_set[3], question_set[4]
            book = is_book(data_dir, story_id)
            chunks = stories_chunked.get(story_id, list())
            if not chunks and book:
                chunks = chunk_story_paragraphs(story_id, data_dir)
                stories_chunked[story_id] = chunks
            for cnt, chunk in enumerate(chunks):
                entries.append({
                    'question_id':"{}_q{}".format(story_id,question_idx),
                    'passage_id':"{}_p{}".format(story_id,cnt),
                    'question':question_str,
                    'passage':"({}) - {}".format(cnt, chunk),
                    'answer1':answer_1,
                    'answer2':answer_2})
    return entries

if __name__=="__main__":
    NQA_DIR = "./data/narrativeqa/"
    #CHUNK_SIZE = 2000
    print("Start processing data")
    entries = extract_data_entries(NQA_DIR)
    fieldnames = entries[0].keys()
    print("Finished processing. Now write data in {}narrative_qa_book.eval".format(NQA_DIR))
    with open(NQA_DIR+"narrativeqa_book.eval", "w", newline='') as writer:
        dict_writer = csv.DictWriter(writer, fieldnames=fieldnames, delimiter='\t')
        dict_writer.writerows(entries)

