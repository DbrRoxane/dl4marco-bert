import csv
import glob
import jsonlines
import nltk

def gather_paragraphs(story_id, query_id, paragraphs_ids):
    context = list()
    paragraphs_tokenized = {}
    found = False
    with open("./data/narrativeqa/tmp/"+story_id+"_"+query_id+"_book.eval", 'r') as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            paragraph_id = row[1].strip()
            if paragraph_id in paragraphs_ids:
                paragraphs_tokenized[paragraph_id] = nltk.word_tokenize(row[3].lower())
                if not found:
                    found=True
                    question = nltk.word_tokenize(row[2].lower())
                    answer1 = nltk.word_tokenize(row[4].lower())
                    answer2 = nltk.word_tokenize(row[5].lower())
    for paragraph_id in paragraphs_ids:
        context += paragraphs_tokenized[paragraph_id]
    return context, question, answer1, answer2

def select_n_best_paragraphs(ranking_file, n):
    with open(ranking_file, 'r') as r_f:
        ranking_reader = csv.reader(r_f, delimiter="\t")
        best_story = next(ranking_reader)
        story_id, query_id = best_story[0].split("_")
        query_id = query_id.replace('q','').strip()
        paragraphs_ids = [best_story[1].strip()]
        for i in range(1, n):
            try:
                next_story = next(ranking_reader)
                paragraphs_ids.append(next_story[1].strip())
            except:
                print("StopIteration Error : {} ".format(r_f.name))
    context, question, a1, a2 = gather_paragraphs(story_id, query_id, paragraphs_ids)
    return context, question, a1, a2, story_id

def write_jsonl(n_best=3):
    jsonl_list = list()
    for ranking_file in glob.iglob("./data/output/narrative_book_paragraphs/nqa_predictions_*"):
        context, q, a1, a2, doc_num = select_n_best_paragraphs(ranking_file, n_best)
        jsonl_list.append({'doc_num':doc_num, 'summary':context, 'ques':q, 'answer1':a1, 'answer2':a2, 'commonsense':[]})
    with jsonlines.open('./data/narrativeqa/mhpg_mit_answer.jsonl', mode='w') as f:
        f.write_all(jsonl_list)

def main():
    write_jsonl()

if __name__=="__main__":
    main()
