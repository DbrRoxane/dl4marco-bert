import jsonlines
import json
from convertor import convert_docs_in_dic
import nltk

BAUER_TEST_FILE = "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl"
NARRATIVEQA_FILE = "./data/narrativeqa/narrativeqa_all.eval"
MIN_PREDICTIONS = ["./data/output/min_trainsum_hardem_r5/min_predsum_r5_trainedsum_testpredictions.json",
                   #"./data/output/min_trainsum_hardem_r5/min_predsum_r5_trainedsum_devpredictions.json",
                   "./data/output/min_trainsum_hardem_r5/min_predall_oracle_r5_allranking_trainedsum_testpredictions.json"]
                   #"./data/output/min_trainsum_hardem_r5/min_predall_oracle_r5_allranking_trainedsum_devpredictions.json"]
OUTPUT_PRED = ["../CommonSenseMultiHopQA/min_trainedsum_hardem_r5_predsum_r5_test_converted.txt",
               #"../CommonSenseMultiHopQA/min_trainedsum_hardem_r5_predsum_r5_dev_converted.txt",
               "../CommonSenseMultiHopQA/min_trainedsum_hardem_r5_predall_r5_allranking_test_converted.txt"]
               #"../CommonSenseMultiHopQA/min_trainedsum_hardem_r5_predall_r5_allranking_dev_converted.txt"]


def convert(dataset, input_file, output_file, n=0):
    with open(input_file, "r") as pred_file:
        pred = json.load(pred_file)
    with open(output_file, "w") as writer:
        with jsonlines.open(BAUER_TEST_FILE, "r") as bauer_file:
            for example in bauer_file:
                if example['doc_num'] in dataset.keys():
                    writen = False
                    for query_key, query_value in dataset[example['doc_num']]['queries'].items():
                        if example['ques'] == nltk.word_tokenize(query_value['query'].lower()):
                            query_id = query_key
                            generated_answer = pred.get(query_id, ["NO PREDICTION"]*5)[n]
                            writer.write(generated_answer+"\n")
                            writen = True
                            break
                    if not writen:
                        print("fuck")
                        writer.write("NO PREDICTION\n")
                #else:
                #    writer.write("NO PREDICTION\n")
if __name__=="__main__":
    assert len(MIN_PREDICTIONS)==len(OUTPUT_PRED)
    dataset = convert_docs_in_dic(NARRATIVEQA_FILE)
    for i in range(len(MIN_PREDICTIONS)):
        n = 0 if i<1 else 4
        convert(dataset, MIN_PREDICTIONS[i], OUTPUT_PRED[i], n)
