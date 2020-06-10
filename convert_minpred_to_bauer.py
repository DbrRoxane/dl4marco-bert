import jsonlines
import json
from convertor import convert_docs_in_dic
import nltk

BAUER_FILE = [
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
#    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
    "../CommonSenseMultiHopQA/data/narrative_qa_valid.jsonl",
    "../CommonSenseMultiHopQA/data/narrative_qa_test.jsonl",
]
NARRATIVEQA_FILE = "./data/narrativeqa/narrativeqa_all.eval"
MIN_PREDICTIONS = [
#    "../data/output/P2_CRCAE/CRCAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P2_CRCAE/CRCAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P2_CRCAE/CRNAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P2_CRCAE/CRNAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P3_CRNAE/CRNAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P3_CRNAE/CRNAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P1_SUM/CRNAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P1_SUM/CRNAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P1_SUM/NRNAE_BERT_20para_r5_dev_predictions.json",
#    "../data/output/P1_SUM/NRNAE_BERT_20para_r5_test_predictions.json",
#    "../data/output/P1_SUM/SUM_dev_predictions.json",
#    "../data/output/P1_SUM/SUM_test_predictions.json",
#    "../data/output/P1_SUM/CRCAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P1_SUM/CRCAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P2_CRCAE/NRNAE_BERT_20para_r5_dev_predictions.json",
#    "../data/output/P2_CRCAE/NRNAE_BERT_20para_r5_test_predictions.json",
#    "../data/output/P3_CRNAE/NRNAE_BERT_20para_r5_dev_predictions.json",
#    "../data/output/P3_CRNAE/NRNAE_BERT_20para_r5_test_predictions.json",
#    "../data/output/P3_CRNAE/CRCAE_allranking_3para_r5_dev_predictions.json",
#    "../data/output/P3_CRNAE/CRCAE_allranking_3para_r5_test_predictions.json",
#    "../data/output/P3-P2_CRNAE/trainP3-P2_testCRCAE_3para_r5_oralce_dev18mai_predictions.json",
#    "../data/output/P3-P2_CRNAE/trainP3-P2_testCRCAE_3para_r5_oralce_test18mai_predictions.json",
#    "../data/output/P3-P2_CRNAE/trainP3-P2_testCRNAE_3para_r5_final_dev18mai_predictions.json",
#    "../data/output/P3-P2_CRNAE/trainP3-P2_testCRNAE_3para_r5_final_test18mai_predictions.json",
    "../data/output/P3noP2_CRNAE/NRNAE_20para_r6_dev_predictions.json",
    "../data/output/P3noP2_CRNAE/NRNAE_20para_r6_test_predictions.json",

]

OUTPUT_PRED = [
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testCRCAE_allranking_3para_r5_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testCRCAE_allranking_3para_r5_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testCRNAE_allranking_3para_r5_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testCRNAE_allranking_3para_r5_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testCRNAE_allranking_3para_r5_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testCRNAE_allranking_3para_r5_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testCRNAE_allranking_3para_r5_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testCRNAE_allranking_3para_r5_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testNRNAE_BERT_20para_r5_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testNRNAE_BERT_20para_r5_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testSUM_dev_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testSUM_test_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testCRCAE_allranking_6para_r5_dev_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P1_SUMtrain_testCRCAE_allranking_6para_r5_test_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testNRNAE_BERT_20para_r5_dev_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P2_CRCAEtrain_testNRNAE_BERT_20para_r5_test_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testNRNAE_BERT_20para_r5_dev_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testNRNAE_BERT_20para_r5_test_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testCRCAE_allranking_9para_r5_dev_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/P3_CRNAEtrain_testCRCAE_allranking_9para_r5_test_predictions.json",
#    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testCRCAE_5para_r5_oralce_dev18mai_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testCRCAE_5para_r5_oralce_test18mai_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testCRNAE_5para_r5_final_dev18mai_predictions.txt",
#    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testCRNAE_5para_r5_final_test18mai_predictions.txt",
    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testNRNAE_15para_r6_dev_nbest_predictions.txt",
    "../CommonSenseMultiHopQA/paper_results/trainP3-P2_testNRNAE_15para_r6_test_nbest_predictions.txt",
]

def convert(dataset, input_file, output_file, bauer, n=0):
    with open(input_file, "r") as pred_file:
        pred = json.load(pred_file)
    with open(output_file, "w") as writer:
        with jsonlines.open(bauer, "r") as bauer_file:
            for example in bauer_file:
                if example['doc_num'] in dataset.keys():
                    writen = False
                    for query_key, query_value in dataset[example['doc_num']]['queries'].items():
                        if example['ques'] == nltk.word_tokenize(query_value['query'].lower()):
                            query_id = query_key
                            generated_answer = pred.get(query_id, ["NO PREDICTION"]*(n))[n-1]
                            writer.write(generated_answer+"\n")
                            writen = True
                            break
                    if not writen:
                        print("fuck")
                        writer.write("NO PREDICTION\n")
if __name__=="__main__":
    print(len(MIN_PREDICTIONS),len(OUTPUT_PRED),len(BAUER_FILE))
    assert len(MIN_PREDICTIONS)==len(OUTPUT_PRED)==len(BAUER_FILE)
    dataset = convert_docs_in_dic(NARRATIVEQA_FILE)
    n=[2,2]
    for i in range(len(MIN_PREDICTIONS)):
        convert(dataset, MIN_PREDICTIONS[i], OUTPUT_PRED[i], BAUER_FILE[i], n[i])
