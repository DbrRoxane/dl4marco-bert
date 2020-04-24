import textwrap
import csv
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


CHUNK_SIZE = 1500
NQA_DIR = "./data/narrativeqa/"

def chunk_story(story_str, chunk_size):
    return textwrap.wrap(re.sub('\s+', ' ',story_str), chunk_size)

def show_chunks_size(stories):
    nb_chunks = []
    size_chunks = []
    for story_chunks in stories.values():
        nb_chunks.append(len(story_chunks))
        for chunk in story_chunks:
            size_chunks.append(len(chunk))
    plt.figure()

    plt.subplot(121)
    plt.hist(nb_chunks, bins=10)
    plt.xlabel("number of chunks per book")
    plt.savefig("./chunks_per_book.png")

    plt.subplot(122)
    plt.hist(size_chunks, bins=list(range(1,3502, 500)))
    plt.xlabel("size of each chunk")
    plt.savefig("./chunk_size.png")

def chunk_story_paragraphs(story_id, data_dir):
    story_file = NQA_DIR + "tmp/" + story_id + ".content"
    with open(story_file, 'r', encoding="utf-8", errors="ignore") as f:
        story_str = f.read()
        if is_html(data_dir, story_id):
            story_str = preprocess_html(story_str)
        chunks = []
        for paragraph in story_str.split("\n\n"):
            if len(paragraph) > CHUNK_SIZE:
                for smaller_paragraph in paragraph.split("\n"):
                    chunk = re.sub('\s+', ' ', smaller_paragraph).replace('\0', '')
                    chunks.append(chunk)
            else:
                chunk = re.sub('\s+', ' ',paragraph).replace('\0', '')
                chunks.append(chunk)
    #chunks =  [re.sub('\s+', ' ',paragraph).replace('\0', '')
    #           for paragraph in story_str.split("\n\n")]
    return chunks

def gather_paragraphs(paragraphs):
    chunks = []
    chunk = ""
    for paragraph in paragraphs:
        chunk += " " + paragraph + " "
        if len(chunk) >= CHUNK_SIZE:
            chunks.append(chunk)
            #print(len(chunk))
            chunk = ""
    chunks.append(chunk)
    return chunks


def preprocess_html(story_str):
    soup = BeautifulSoup(story_str, "html.parser")
    clean_story = soup.get_text().replace("\t","")
    return clean_story


def is_html(data_dir, story_id):
    with open(data_dir+"documents.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        html = ["html" in row[3] for row in csv_reader if row[0]==story_id][0]
    return html


def is_book(data_dir, story_id):
    with open(data_dir+"documents.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        book = [row[2]=="gutenberg" for row in csv_reader if row[0]==story_id][0]
    return book

def extract_data_entries(data_dir, gather_char_min):
    entries = list()
    with open(data_dir+"qaps.csv", "r") as qa_file:
        csv_reader = csv.reader(qa_file, delimiter=',')
        next(csv_reader)
        stories_chunked = dict()
        for question_idx, question_set in enumerate(csv_reader):
            story_id = question_set[0]
            question_str = question_set[2]
            answer_1, answer_2 = question_set[3], question_set[4]
            chunks = stories_chunked.get(story_id, list())
            if not chunks:
                chunks = chunk_story_paragraphs(story_id, data_dir)
                if gather_char_min:
                    chunks = gather_paragraphs(chunks)
                stories_chunked[story_id] = chunks
            for cnt, chunk in enumerate(chunks):
                entries.append({
                    'question_id':"{}_q{}".format(story_id,question_idx),
                    'passage_id':"{}_p{}".format(story_id,cnt),
                    'question':question_str,
                    'passage':"({}) - {}".format(cnt, chunk),
                    'answer1':answer_1,
                    'answer2':answer_2})
    show_chunks_size(stories_chunked)
    return entries

if __name__=="__main__":
    print("Start processing data")
    entries = extract_data_entries(NQA_DIR, gather_char_min=True)
    fieldnames = entries[0].keys()
    print("Finished processing. Now write data in {}narrativeqa_all.eval".format(NQA_DIR))
    with open(NQA_DIR+"narrativeqa_all.eval", "w", newline='') as writer:
        dict_writer = csv.DictWriter(writer, fieldnames=fieldnames, delimiter='\t')
        dict_writer.writerows(entries)

