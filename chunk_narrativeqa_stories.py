import textwrap
import csv
import re

def chunk_story(story_str, chunk_size):
    return textwrap.wrap(re.sub('\s+', ' ',story_str), chunk_size)

if __name__=="__main__":
    NQA_DIR = "./data/narrativeqa/"
    CHUNK_SIZE = 2000
    with open(NQA_DIR+"qaps.csv", "r") as qa_file:
        csv_reader = csv.reader(qa_file, delimiter=',')
        next(csv_reader)
        for question_idx, question_set in enumerate(csv_reader):
            story_id = question_set[0]
            question_str = question_set[2]
            story_file = NQA_DIR + "tmp/" + story_id
            with open(story_file + ".content", "r" , encoding="utf-8", errors="ignore") as story_str:
                chunks = chunk_story(story_str.read(), CHUNK_SIZE)
            with open(story_file + "_" + str(question_idx) + ".eval", "w") as writer:
                for cnt, chunk in enumerate(chunks):
                    line = "{0} \t {1}_{4} \t {2} \t ({4}) - {3} \n".\
                            format(question_idx, story_id, question_str, chunk, cnt)
                    writer.write(line)


