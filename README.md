**DRAFT**

# Passage Re-ranking with BERT adapted to Narrative-QA over stories!! 

Adapted the code from Nogueria, described [here](https://arxiv.org/abs/1901.04085) and available [here](https://github.com/nyu-dl/dl4marco-bert) to 
run it (eval only for the moment) on NarrativeQA dataset, which is described
[here](https://arxiv.org/abs/1712.07040) and is available [here](https://github.com/deepmind/narrativeqa).

## 1. Download NarrativeQA dataset
`cd data` 
 
 `git clone https://github.com/deepmind/narrativeqa.git`
 
 `cd narrativeqa`
 
 `./download_stories.sh` (be carreful! I have some errors with `sh download_stories.sh`)
 
## Download BERT pretrained on MSMARCO

Base model

`sh dl_bert_base_msm.sh`

 `unzip BERT_BASE_MSMARCO.zip -d BERT_BASE_MSMARCO`
 
Large model
 `sh dl_bert_large_msm.sh`
 
 `unzip BERT_LARGE_MSMARCO.zip -d BERT_LARGE_MSMARCO`

## Prepare the data
Will be in the same foder as the stories, i.e `data/narrativeqa/tmp/`

`python chunk_narrativeqa_stories.py`

`mkdir data/narrativeqa/nqa_tf`

We currently test the model on a single question from a movie.

`python convert_nqa_to_tfrecord.py`

`mkdir data/output`

## Run evaluation

`python run_nqa.py`
=======
# Narrative-Reader

