import csv
import glob

def gather_paragraphs(story_id, query_id, paragraphs_ids):
    context = ""
    paragraphs_str = {}
    found = False
    with open("./data/narrativeqa/tmp/"+story_id+"_"+query_id+"_book.eval", 'r') as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for line in csv_reader:
            if line[1] in paragraphs_ids:
                paragraphs_str[line[1]] = line[3]
                if not found:
                    pass
    for paragraph_id in paragraphs_ids:
        context += paragraphs_str[paragraph_id]
    return context

def select_n_best_parapgraphs(ranking_file, n):
    with open(ranking_file, 'r') as r_f:
        ranking_reader = csv.reader(r_f, delimiter="\t")
        best_story = next(ranking_reader)
        story_id, query_id = best_story[0].split("_")
        query_id = query_id.replace('q','')
        paragraphs_ids = [best_story[1]]
        for i in range(1, n):
            next_story = next(ranking_reader)
            paragraphs_ids.append(next_story[1])
    context = gather_paragraphs(story_id, query_id, paragraphs_ids)
    return context

def write_jsonl(output_file, n_best):
    for ranking_file in glob.iglob("./data/output/narrative_book_paragraphs/nqa_predictions_*"):
        context = select_n_best_paragraphs(ranking_file, n_best)
        doc_num = 0
        ques = 0
        answer1 = 0
