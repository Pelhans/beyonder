#!/usr/bin/env python3
# coding=utf-8

"""Training function """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../../basic_net/"))
sys.path.append(os.path.join(BASE_DIR, "../tools/"))

import re
import random
import collections
import optimization
import tokenization
from tokenizer import Tokenizer
import tensorflow as tf
from network.bert_crf import bert_crf
from network.bilstm_crf import blstm_crf
from evaluation import chain, precision_recall_f1, decode_ner
import modeling
from util import read_dataset

task = "bert_crf"
#task = "blstm_crf"
lr = 3e-5

if task == "bert_crf":
    dir = os.path.join(BASE_DIR, "models/bert_crf/")
    MODEL = bert_crf
elif task == "blstm_crf":
    dir = os.path.join(BASE_DIR, "models/blstm_crf/")
    MODEL = blstm_crf
    lr = 3e-3
else:
    raise ValueError("Wrong task name")

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", os.path.join(BASE_DIR,"data/"), 
                    "train/test data dir")
flags.DEFINE_string("serving_model_save_path", os.path.join(BASE_DIR, "pb_model/"),
                    "dir for pb output")
flags.DEFINE_string("task_name", "ner",
                    "different task use different processor")
flags.DEFINE_string("vocab_file", os.path.join(BASE_DIR, "./pb_model/vocab.txt"),
                    "path to vocab file")
flags.DEFINE_string("output_dir", dir, "ckpt dir")
flags.DEFINE_string("init_checkpoint", os.path.join(BASE_DIR, "../bert_base/bert_model.ckpt"),
                    "path to bert model from google or bert")
flags.DEFINE_string("bert_config_file", os.path.join(BASE_DIR, "../bert_base/bert_config.json"), 
                   "The config json file corresponding to the pre-trained BERT model. ",)
flags.DEFINE_integer("max_seq_length", 128, 
                     "max sequence length for each sentence")
flags.DEFINE_integer("vocab_size", 21128, 
                     "chinese vocab size")
flags.DEFINE_integer("embedding_size", 128, 
                     "max sequence length for each sentence")
flags.DEFINE_integer("hidden_size_blstm", 256, 
                     "hidden unit size for bilstm")
flags.DEFINE_integer("layer_num", 2, 
                     "layer num for bilstm")

flags.DEFINE_bool("do_train", True, "excute train steps")
flags.DEFINE_bool("do_eval", True, "excute eval step")
flags.DEFINE_bool("do_predict", True, "excute predict step")

flags.DEFINE_integer("train_batch_size", 32, "train batch size")
flags.DEFINE_integer("eval_batch_size", 32, "eval batch size")
flags.DEFINE_integer("predict_batch_size", 32, "predict batch size")

flags.DEFINE_float("learning_rate", lr, "learning rate")
flags.DEFINE_integer("num_train_epochs", 5,
                     "train epoch, for BMES task, 3 is enough, content task is 5")

flags.DEFINE_integer("save_checkpoints_steps", 10000, 
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 2000, 
                     "How many steps to make in each estimator call.")

flags.DEFINE_float("warmup_proportion", 0.1, 
                   "Proportion of training to perform linear learning rate warmup for.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, textA, textB, label):
        """Constructs a InputExample
        
        :param guid: Unique id for example
        :type guid: string
        :param text: Char based text, split with space
        :type text: string
        :param label: BMES label
        :type label: list
        :param schema_label: set/tag/content label
        :type schema_label: list
        :return:None
        """

        self.guid = guid
        self.textA = textA
        self.textB = textB
        self.label = label

class InputFeatures(object):
    """A single set of features of data"""

    def __init__(self,
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                sequence_length,
                is_real_example=True):
        """
        :param input_ids: input text ids for each char
        :type input_ids: list
        :param input_mask: mask for sentence to suit bert model, 
                        for this task, all is 1, length is max_seq_length
        :type input_mask: list
        :param segment_ids: 0 for first sentence and 1 for second sentence,
                            for this task, all is 0, length is max_seq_length
        :type segment_ids: list
        :param sequence_length: sequence_length for each input sentence
        :type sequence_length: int
        """

        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.sequence_length = sequence_length
        self.is_real_example = is_real_example

class NERProcessor(object):
    """Processor for POI Parsing."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(read_dataset(os.path.join(BASE_DIR, "data/ner/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(read_dataset(os.path.join(BASE_DIR, "data/ner/dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(read_dataset(os.path.join(BASE_DIR, "./data/ner/test.csv")), "test")

    def get_labels(self):
        """Gets thie list of BIO labels for this dataset
        """
        return ['O', 'B', 'I']

    def _create_examples(self, lines, set_type="test"):
        """Creates examples for the training and dev sets.
        :param lines: all input lines from input file
        :type lines: list
        :return: a list of InputExample element
        :rtype: list
        """
        examples = []
        for (i, line) in enumerate(lines):
            line = line.split("\t")
            guid = "%s-%s" % (set_type, i)
            textA = tokenization.convert_to_unicode(line[0])
            label = eval(line[1])
            examples.append(InputExample(guid=guid, textA=textA, textB=None, label=label))
        # examples 包含了所有数据的列表, 其中每个数据类型为 InputExample
        # 对于训练数据进行随机打乱
        if set_type == "train":
            random.shuffle(examples)
        return examples

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size"""

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "sequence_length": tf.FixedLenFeature([1], tf.int64),
    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                )
            )
        return d

    return input_fn

def  model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps,
                     num_warmup_steps, use_one_hot_embeddings):
    """Return 'model_fn' closure for TPUEstimator."""

    def model_fn(features, labels,  mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        sequence_length = features["sequence_length"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids)[0], dtype=tf.float32)
        is_training = (mode == "train")

        (total_loss, per_example_loss, logits, pred_ids) = MODEL(bert_config, is_training, input_ids, segment_ids, input_mask, label_ids, sequence_length, num_labels,  use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss,
                            learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn, )
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                scaffold_fn=scaffold_fn, )

        else:
            in_labels_b = tf.identity(label_ids)
            sequence_length_out = tf.identity(sequence_length)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"label_ids": in_labels_b,
                             "sequence_length": sequence_length_out,
                             "pred_ids": pred_ids},
                scaffold_fn=scaffold_fn, )

        return output_spec
    return model_fn

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, output_file):
    
    writer = tf.python_io.TFRecordWriter(output_file)
    wordid_map = word2id(FLAGS.vocab_file)
    tokenizer = Tokenizer(wordid_map)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i 

#    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=False)
    num_words = 0
    num_unk = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature, num_words, num_unk = convert_single_example(ex_index, example, label_list, max_seq_length, wordid_map, label_map,  tokenizer, num_words, num_unk)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["sequence_length"] = create_int_feature([feature.sequence_length])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def word2id(vocab_file):
    with open(vocab_file, "r") as vf:
        return {k.strip(): idx for (idx, k) in enumerate(vf.readlines())}

def get_ids_seg(text,tokenizer):
    '''
    得到bert的输入
    :param text:
    :param tokenizer:
    :param max_len:
    :return:
    '''
    indices= []
    for ch in text:
        indice, _ = tokenizer.encode(first=ch)
        if len(indice) != 3:
            indices += [100]
        else:
            indices += indice[1:-1]
    return indices

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate thelonger sequence
    # one token at a time. This makes more sense than truncatingan equal percent
    # of tokens from each, since if one sequence is very shortthen each token
    # that's truncated likely contains more information than alonger sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()
    return "".join(tokens_a), "".join(tokens_b)

def convert_single_example(ex_index, example, label_list, max_seq_length, wordid_map, label_map, tokenizer, num_words=1, num_unk=1):
    if isinstance(example, PaddingInputExample): 
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            sequence_length=0,
            is_real_example=False, )

    input_ids = []
    # char based
    if example.textB is None:
        input_ids = [101] + get_ids_seg(example.textA, tokenizer) + [102]
        label_id = [0] + [label_map[l] for l in example.label] + [0]
        segment_ids = [0] * len(input_ids)
    else: 
        example.textA, example.textB = _truncate_seq_pair(list(example.textA), list(example.textB),max_seq_length - 3)
          
        input_ids = [101] + get_ids_seg(example.textA, tokenizer) + [102] + get_ids_seg(example.textB, tokenizer) + [102]
        segment_ids = [0] * (len(example.textA) + 2) + [1] * (len(example.textB) + 1)
        label_id = [int(example.label)]* len(input_ids)
          
    sequence_length = len(input_ids) if len(input_ids) <= max_seq_length else max_seq_length
    input_mask = [1] * len(input_ids)
    assert len(label_id) == len(input_ids)
    assert len(input_ids) == len(segment_ids)
          
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        label_id.append(0)
        segment_ids.append(0)
                       
    input_ids = input_ids[:max_seq_length]
    label_id = label_id[:max_seq_length] if FLAGS.task_name == "ner" else [label_id[0]]
    input_mask = input_mask[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
                       
                       
    assert len(input_ids) == max_seq_length
                       
    if ex_index < 5:   
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % "".join("[CLS]" + str(example.textA) + "[SEP]" +  str(example.textB) + "[SEP]"))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in [label_id]]))
        tf.logging.info("sequence_length: %s" % sequence_length)
                    
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_id,
        sequence_length=sequence_length,
        is_real_example=True,)
    return feature, num_words, num_unk


def serving_input_receiver_fn():
    input_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_ids')
    segment_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    input_mask = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='input_mask')
    label_ids = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length], name='label_ids')
    sequence_length = tf.placeholder(dtype=tf.int64, shape=[None,], name='sequence_length')

    receive_tensors = {'input_ids': input_ids, 'input_mask': input_mask, "sequence_length": sequence_length,
                       'segment_ids': segment_ids, 'label_ids': label_ids,}

    features = {"input_ids": input_ids, 'input_mask': input_mask, "sequence_length": sequence_length,
                'segment_ids': segment_ids, "label_ids": label_ids,}
    return tf.estimator.export.ServingInputReceiver(features, receive_tensors)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"ner": NERProcessor}

    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found {}".format(task_name))

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    
    processor = processors[task_name]()
    label_list = processor.get_labels()
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    total_num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    each_num_train_steps = int(len(train_examples) / FLAGS.train_batch_size)
    num_warmup_steps = int(each_num_train_steps * FLAGS.warmup_proportion)
    
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=4,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2,
            )
        )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=total_num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        )

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    if not os.path.exists(train_file):
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", total_num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True )


    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples) 
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        file_based_convert_examples_to_features(eval_examples, label_list,  FLAGS.max_seq_length, eval_file)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                   len(eval_examples), num_actual_eval_examples,
                   len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_steps = None
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                               seq_length=FLAGS.max_seq_length,
                                               is_training=False,
                                               drop_remainder=eval_drop_remainder,)

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if not os.path.exists(predict_file):
        file_based_convert_examples_to_features(predict_examples, label_list,
                                               FLAGS.max_seq_length, predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                   len(predict_examples), num_actual_predict_examples,
                   len(predict_examples) - num_actual_predict_examples)
    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder, )

    best_f1 = 0
    best_epoch = 0
    for epoch in range(int(total_num_train_steps/each_num_train_steps)):
#    for epoch in range(1):
        if FLAGS.do_train:
            estimator.train(input_fn=train_input_fn, max_steps=each_num_train_steps*(epoch+1))
#            estimator.train(input_fn=train_input_fn, max_steps=total_num_train_steps)
        if FLAGS.do_eval:
            estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        if not FLAGS.do_predict:
            continue
        result = estimator.predict(input_fn=predict_input_fn)

        tf.logging.info("***** Test results for epoch {} *****".format(epoch))
        correct = 0
        total_count = 0
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results_epoch_{}.tsv".format(epoch))
        infile = open( os.path.join(BASE_DIR, "data/ner/test.csv")).readlines()
        outf = open(FLAGS.output_dir + "test.txt", "w")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            for (i, prediction) in enumerate(result):
                line = infile[i].split("\t")[0].strip()
                pred_ids = prediction["pred_ids"]
                sequence_length = prediction["sequence_length"]
                len_seq = sequence_length[0]
                correct += (prediction["label_ids"] == pred_ids)[1:len_seq].sum()
                total_count += len_seq - 1
                result = [str(r) for r in  pred_ids][1:len_seq-1]
                label_ids = [str(l) for l in prediction["label_ids"]][1:len_seq-1]
                if i >= num_actual_predict_examples:
                    break
                result_decode = decode_ner(result)
                result_decode_str = []
                if result_decode:
                    for res_decode in result_decode:
                        result_decode_str.append( line[res_decode[0]: res_decode[1]+1] )
                output_line = str(result)  + "\t" + str(label_ids) + "\n"
                real_out = {"origin": line, "entity": result_decode_str}
                outf.write(str(real_out) + "\n")
                writer.write(output_line)
                num_written_lines += 1

        assert num_written_lines == num_actual_predict_examples
        tf.logging.info("*** Accuracy for BIO: {}".format(correct/total_count))
        if FLAGS.task_name == "ner":
            res_pred, res_gold = chain("{}/test_results_epoch_{}.tsv".format(dir, epoch))
            results = precision_recall_f1(y_pred=res_pred, y_true=res_gold)
            f1 = results['__total__']['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
    tf.logging.info("Best F1 score is {} in epoch {}".format(best_f1, best_epoch))

    estimator._export_to_tpu = False
    estimator.export_savedmodel(FLAGS.serving_model_save_path, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.app.run()
