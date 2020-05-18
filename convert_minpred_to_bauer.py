import jsonlines
import json
from convertor import convert_docs_in_dic
import nltk

BAUER_TEST_FILE = "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl"
NARRATIVEQA_FILE = "./data/narrativeqa/narrativeqa_all.eval"
MIN_PREDICTIONS = "./data/output/min_predictions/min_with_answer2.jsonpredictions.json"
OUTPUT_PRED = "../CommonSenseMultiHopQA/min_with_answer_pred.txt"

def main():
    dataset = convert_docs_in_dic(NARRATIVEQA_FILE)
    with open(MIN_PREDICTIONS, "r") as pred_file:
        pred = json.load(pred_file)

    with open(OUTPUT_PRED, "w") as writer:
        with jsonlines.open(BAUER_TEST_FILE, "r") as bauer_file:
            for example in bauer_file:
                if example['doc_num'] in dataset.keys():
                    writen = False
                    for query_key, query_value in dataset[example['doc_num']]['queries'].items():
                        if example['ques'] == nltk.word_tokenize(query_value['query'].lower()):
                            query_id = query_key
                            generated_answer = pred.get(query_id, ["NO PREDICTION"])[0]
                            writer.write(generated_answer+"\n")
                            writen = True
                            break
                    if not writen:
                        writer.write("NO PREDICTION\n")
                #else:
                #    writer.write("NO PREDICTION\n")
if __name__=="__main__":
    main()
