"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import re
import tensorflow as tf
import time
# local module
import tokenization


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "output_folder", "./data/processed",
    "Folder where the tfrecord files will be written.")

flags.DEFINE_string(
    "vocab_file",
    "./BERT_LARGE_MSMARCO/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "eval_dataset_path",
    "./data/processed/narrativeqa_all.eval",
    "Path to the MSMARCO eval dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")
flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_eval_docs", 500,
    "The maximum number of docs per query for dev and eval sets.")

flags.DEFINE_integer(
    "num_examples_per_tf", 1000000,
    "The number of examples, which can be divided by 500, to split")

flags.DEFINE_boolean(
    "use_answer", False,
    "Either to use the answer and so have an oracle model for ranking or not")

def write_to_tf_record(writer, tokenizer, query, docs, labels,
                       ids_file=None, query_id=None, doc_ids=None):
  query = tokenization.convert_to_unicode(query)
  query_token_ids = tokenization.convert_to_bert_input(
      text=query, max_seq_length=FLAGS.max_query_length, tokenizer=tokenizer, 
      add_cls=True)

  query_token_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_token_ids))

  for i, (doc_text, label) in enumerate(zip(docs, labels)):

    doc_token_id = tokenization.convert_to_bert_input(
          text=tokenization.convert_to_unicode(doc_text),
          max_seq_length=FLAGS.max_seq_length - len(query_token_ids),
          tokenizer=tokenizer,
          add_cls=False)

    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_id))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]))

    features = tf.train.Features(feature={
        'query_ids': query_token_ids_tf,
        'doc_ids': doc_ids_tf,
        'label': labels_tf,
    })
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())

    if ids_file:
     ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')

def convert_eval_dataset(set_name, tokenizer, use_answer):
  print('Converting {} set to tfrecord...'.format(set_name))
  start_time = time.time()

  if set_name == 'dev':
    dataset_path = FLAGS.dev_dataset_path
    relevant_pairs = set()
    with open(FLAGS.dev_qrels_path) as f:
      for line in f:
        query_id, _, doc_id, _ = line.strip().split('\t')
        relevant_pairs.add('\t'.join([query_id, doc_id]))
  else:
    dataset_path = FLAGS.eval_dataset_path

  split_dataset = []
  queries_docs = collections.defaultdict(list)
  query_ids = {}
  with open(dataset_path, 'r') as f:
    for i, line in enumerate(f):
      if i % FLAGS.num_examples_per_tf == 0 and i > 0:
        processed_dataset = queries_docs.copy()
        queries_docs = collections.defaultdict(list)
        queries_docs[query] = processed_dataset[query].copy()
        del processed_dataset[query]
        split_dataset.append(processed_dataset)
        print("{} checkpoint, save {} queries ".format(i, len(processed_dataset.items())))
      query_id, doc_id, query, doc, a1, a2 = line.strip().split('\t')
      if use_answer:
        query += " " + a1 +" "+ a2
      label = 0
      if set_name == 'dev':
        if '\t'.join([query_id, doc_id]) in relevant_pairs:
          label = 1
      queries_docs[query].append((doc_id, doc, label))
      query_ids[query] = query_id
  processed_dataset = queries_docs.copy()
  queries_docs = collections.defaultdict(list)
  queries_docs[query] = processed_dataset[query].copy()
  del processed_dataset[query]
  split_dataset.append(processed_dataset)

  for j, queries_docs in enumerate(split_dataset):
  # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    for query in queries:
      docs = queries_docs[query]
      docs += max(
        0, FLAGS.num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
      queries_docs[query] = docs[:FLAGS.num_eval_docs]
    assert len(
        set(len(docs) == FLAGS.num_eval_docs for docs in queries_docs.values())) == 1, (
            'Not all queries have {} docs'.format(FLAGS.num_eval_docs))

    writer = tf.python_io.TFRecordWriter(
      FLAGS.output_folder + '/dataset_' + set_name + str(j) + '.tf')

    query_doc_ids_path = (
      FLAGS.output_folder + '/query_doc_ids_' + set_name + str(j) + '.txt')
    with open(query_doc_ids_path, 'w') as ids_file:
      for i, (query, doc_ids_docs) in enumerate(queries_docs.items()):
        doc_ids, docs, labels = zip(*doc_ids_docs)
        query_id = query_ids[query]

        write_to_tf_record(writer=writer,
                         tokenizer=tokenizer,
                         query=query,
                         docs=docs,
                         labels=labels,
                         ids_file=ids_file,
                         query_id=query_id,
                         doc_ids=doc_ids)

        if i % 1000 == 0:
          print('Writing {} set, query {} of {}'.format(
            set_name, i, len(queries_docs)))
          time_passed = time.time() - start_time
          hours_remaining = (
            len(queries_docs) - i) * time_passed / (max(1.0, i) * 3600)
          print('Estimated hours remaining to write the {} set: {}'.format(
            set_name, hours_remaining))
    writer.close()


def main():

  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)

  convert_eval_dataset(set_name='eval',
                       tokenizer=tokenizer,
                       use_answer=FLAGS.use_answer)
  print('Done!')

if __name__ == '__main__':
  main()
