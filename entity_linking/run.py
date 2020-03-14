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

import tensorflow as tf
import os
import re
import random
import collections
import optimization
import tokenization
from sklearn.metrics import f1_score
from tokenizer import Tokenizer
from network.esim import esim
from network.esim_transformer import esim_transformer
from util import read_dataset

cur_task = "esim_transformer"
#cur_task = "esim"
FINETUNE = False

# Dataset ECD, epoch 40
all_tasks = {"esim": {"model": esim, "lr": 2e-4, "epoch": 10},
             "esim_transformer": {"model": esim_transformer, "lr": 2e-4, "epoch": 10},}

MODEL = all_tasks[cur_task]["model"]
lr = all_tasks[cur_task]["lr"]
epoch = all_tasks[cur_task]["epoch"]
dir = "./models/" + str(MODEL.__name__)
dir_public = "./models/"

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "./data/", 
                    "train/test data dir")
flags.DEFINE_string("serving_model_save_path", "./pb_model/",
                    "dir for pb output")
flags.DEFINE_string("task_name", "rerank",
                    "different task use different processor")
flags.DEFINE_string("vocab_file", "./pb_model/vocab.txt",
                    "path to vocab file")
#flags.DEFINE_string("word_vec", "./data/poi_word_vec.txt",
#                    "path to pretrain word vocab")
flags.DEFINE_string("output_dir", dir, "ckpt dir")
flags.DEFINE_string("init_checkpoint", dir+"/model.ckpt",
                    "path to bert model from google or bert")
flags.DEFINE_integer("max_seq_length_A", 64, 
                     "max sequence length for each sentence")
flags.DEFINE_integer("max_seq_length_B", 300, 
                     "max sequence length for each sentence")

flags.DEFINE_bool("do_train", True, "excute train steps")
flags.DEFINE_bool("do_eval", True, "excute eval step")
flags.DEFINE_bool("do_predict", True, "excute predict step")

flags.DEFINE_integer("train_batch_size", 64, "train batch size")
flags.DEFINE_integer("eval_batch_size", 64, "eval batch size")
flags.DEFINE_integer("predict_batch_size", 64, "predict batch size")

flags.DEFINE_float("learning_rate", lr, "learning rate")
flags.DEFINE_integer("num_train_epochs", epoch,
                     "train epoch, for BMES task, 3 is enough, content task is 5")

flags.DEFINE_integer("save_checkpoints_steps", 100000, 
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("hidden_size_cnn", 256, 
                     "Number of CNN filters")
flags.DEFINE_integer("embedding_size", 256, 
                     "Size of word embedding")
flags.DEFINE_integer("vocab_size", 21120, 
                     "vocab size")
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
                input_idsA,
                input_idsB,
                label_ids,
                input_maskA,
                input_maskB,
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

        self.input_idsA = input_idsA
        self.input_idsB = input_idsB
        self.label_ids = label_ids
        self.input_maskA = input_maskA
        self.input_maskB = input_maskB
        self.is_real_example = is_real_example

class DisambiProcessor(object):
    """Processor for Query Parsing."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(read_dataset("./data/disambi/train.txt"), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(read_dataset("./data/disambi/dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(read_dataset("./data/disambi/test.txt"), "test")

    def get_labels(self):
        """Gets thie list of BIO labels for this dataset"""
        return [0, 1]

    def _create_examples(self, lines, set_type="test"):
        """Creates examples for the training and dev sets.
        :param lines: all input lines from input file
        :type lines: list
        :return: a list of InputExample element
        :rtype: list
        """
        examples = []
        mode = re.compile("(\[Type\].*)?\[")
        wrong = 0
        for (i, line) in enumerate(lines):
            line = eval(line.strip())
            guid = "%s-%s" % (set_type, i)
            label = line["tag"]
            textA = line["query_text"]
            # 移除训练数据中的 ["Type"]Thing 这种
            textB = line["candi_abstract"].strip("[Type]")
            try:
                textB = textB[re.search(mode, textB).span()[1]-1:]
            except:
                wrong += 1
            textA = tokenization.convert_to_unicode(textA)
            textB = tokenization.convert_to_unicode(textB)
            examples.append(InputExample(guid=guid, textA=textA, textB=textB, label=label))
        tf.logging.info("There is no Type of sentence: {}".format(wrong))
        # examples 包含了所有数据的列表, 其中每个数据类型为 InputExample
        # 对于训练数据进行随机打乱
        if set_type == "train":
            random.shuffle(examples)
        return examples

class ECDProcessor(object):
    """Processor for POI Parsing."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(read_dataset("./data/E-commerce_dataset/train.txt"), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(read_dataset("./data/E-commerce_dataset/dev.txt"), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(read_dataset("./data/E-commerce_dataset/test.txt"), "test")

    def get_labels(self):
        """Gets thie list of BIO labels for this dataset
        """
        return ['0', '1']

    def _create_examples(self, lines, set_type="test"):
        """Creates examples for the training and dev sets.
        :param lines: all input lines from input file
        :type lines: list
        :return: a list of InputExample element
        :rtype: list
        """
        examples = []
        for (i, line) in enumerate(lines):
            line = line.strip().split("\t")
            guid = "%s-%s" % (set_type, i)
            textA = tokenization.convert_to_unicode(line[-1])
            textB = tokenization.convert_to_unicode("\t".join(line[1:-1]))
            label = line[0]
            examples.append(InputExample(guid=guid, textA=textA, textB=textB, label=label))
        # examples 包含了所有数据的列表, 其中每个数据类型为 InputExample
        # 对于训练数据进行随机打乱
        if set_type == "train":
            random.shuffle(examples)
        return examples

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size"""

def file_based_input_fn_builder(input_file, seq_length_A, seq_length_B, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_idsA": tf.FixedLenFeature([seq_length_A], tf.int64),
        "input_idsB": tf.FixedLenFeature([seq_length_B], tf.int64),
        "input_maskA": tf.FixedLenFeature([seq_length_A], tf.float32),
        "input_maskB": tf.FixedLenFeature([seq_length_B], tf.float32),
        "label_ids": tf.FixedLenFeature([], tf.int64),
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

#def  model_fn_builder(num_labels, init_checkpoint, learning_rate, num_train_steps,
def  model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps, use_one_hot_embeddings):
    """Return 'model_fn' closure for TPUEstimator."""

    def model_fn(features, labels,  mode, params):
        tf.logging.info("*** Features ***")
#        for name in sorted(features.keys()):
#            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_idsA = features["input_idsA"]
        input_idsB = features["input_idsB"]
        input_maskA = features["input_maskA"]
        input_maskB = features["input_maskB"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids)[0], dtype=tf.float32)
        is_training = (mode == "train")

        (total_loss, per_example_loss, logits, probabilities, pred_ids) = MODEL(is_training, 
                                                                                input_idsA,
                                                                                input_idsB,
                                                                                input_maskA,
                                                                                input_maskB,
                                                                                label_ids,
                                                                                num_labels,
                                                                               )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
#        if os.path.exists(init_checkpoint):
#            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
#            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

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
            in_labels = tf.identity(label_ids)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"label_ids": in_labels,
                             "probabilities": probabilities,
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
    examples, label_list, max_seq_length_A, max_seq_length_B, output_file):
    
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

        feature, num_words, num_unk = convert_single_example(ex_index, example,  label_list,
                                                             max_seq_length_A, max_seq_length_B, 
                                                             wordid_map, label_map,  tokenizer,
                                                             num_words, num_unk)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_idsA"] = create_int_feature(feature.input_idsA)
        features["input_idsB"] = create_int_feature(feature.input_idsB)
        features["input_maskA"] = create_float_feature(feature.input_maskA)
        features["input_maskB"] = create_float_feature(feature.input_maskB)
        features["label_ids"] = create_int_feature([feature.label_ids])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info("number of total words: {},\nNumber of UNK words: {}".format(num_words, num_unk))

def word2id(vocab_file):
    with open(vocab_file, "r") as vf:
        return {k.strip(): idx for (idx, k) in enumerate(vf.readlines())}

def get_ids_seg(text, wordid_map, num_words, num_unk):
    '''
    得到bert的输入
    :param text:
    :param tokenizer:
    :param max_len:
    :return:
    '''
    indices= []
    text = "".join(text.split(" "))
    num_words += len(text)
    for ch in text:
        if ch in wordid_map:
            indice = wordid_map[ch]
        else:
            indice = wordid_map["[UNK]"]
            num_unk += 1
        indices.append(indice)
    return indices, num_words, num_unk

def convert_single_example(ex_index, example, label_list, max_seq_length_A, max_seq_length_B,
                           wordid_map, label_map, tokenizer, num_words=1, num_unk=1):
    if isinstance(example, PaddingInputExample): 
        return InputFeatures(
            input_idsA=[0] * max_seq_length_A,
            input_idsB=[0] * max_seq_length_B,
            input_maskA=[0] * max_seq_length_A,
            input_maskB=[0] * max_seq_length_B,
            label_ids= 0,
            is_real_example=False, )

    input_idsA, num_words, num_unk = get_ids_seg(example.textA, wordid_map, num_words, num_unk)
    input_idsA = input_idsA + (max_seq_length_A - len(input_idsA)) * [0]
    input_idsA = input_idsA[:max_seq_length_A]
    input_maskA = [1] * len(example.textA) + [0] * (max_seq_length_A-len(example.textA))
    input_maskA = input_maskA[:max_seq_length_A]

    input_idsB, num_words, num_unk = get_ids_seg(example.textB, wordid_map, num_words, num_unk)
    input_idsB = input_idsB + (max_seq_length_B - len(input_idsB)) * [0]
    input_idsB = input_idsB[:max_seq_length_B]
    input_maskB = [1] * len(example.textB) + [0] * (max_seq_length_B-len(example.textB))
    input_maskB = input_maskB[:max_seq_length_B]

    label_id = int(example.label)
    
          
    assert len(input_idsB) == max_seq_length_B
    assert len(input_idsA) == max_seq_length_A
    assert len(input_maskA) == max_seq_length_A
                       
    if ex_index < 5:   
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % "".join("[CLS]" + str(example.textA) + "[SEP]" +  str(example.textB) + "[SEP]"))
        tf.logging.info("input_idsA: %s" % " ".join([str(x) for x in input_idsA]))
        tf.logging.info("input_maskA: %s" % " ".join([str(x) for x in input_maskA]))
        tf.logging.info("input_idsB: %s" % " ".join([str(x) for x in input_idsB]))
        tf.logging.info("input_maskB: %s" % " ".join([str(x) for x in input_maskB]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in [label_id]]))
                    
    feature = InputFeatures(
        input_idsA=input_idsA,
        input_idsB=input_idsB,
        input_maskA=input_maskA,
        input_maskB=input_maskB,
        label_ids=label_id,
        is_real_example=True,)
    return feature, num_words, num_unk

def serving_input_receiver_fn():
    input_idsA = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length_A], name='input_idsA')
    input_idsB = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.max_seq_length_B], name='input_idsB')
    input_maskA = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length_A], name='input_maskA')
    input_maskB = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.max_seq_length_B], name='input_maskB')
    label_ids = tf.placeholder(dtype=tf.int64, shape=[None, ], name='label_ids')

    receive_tensors = {'input_idsA': input_idsA,
                       'input_idsB': input_idsB,
                       "input_maskA": input_maskA,
                       "input_maskB": input_maskB,
                       'label_ids': label_ids,}

    features = {"input_idsA": input_idsA,
                "input_idsB": input_idsB,
                "input_maskA": input_maskA,
                "input_maskB": input_maskB,
                "label_ids": label_ids,}
    return tf.estimator.export.ServingInputReceiver(features, receive_tensors)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"rerank": DisambiProcessor,
                  "ecd": ECDProcessor}

    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found {}".format(task_name))


    
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
        num_labels=len(label_list),
#        init_checkpoint=FLAGS.init_checkpoint,
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

    train_file = os.path.join(dir_public, "train.tf_record")
    if not os.path.exists(train_file):
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length_A, FLAGS.max_seq_length_B,  train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", total_num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length_A=FLAGS.max_seq_length_A,
        seq_length_B=FLAGS.max_seq_length_B,
        is_training=True,
        drop_remainder=True )


    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples) 
    eval_file = os.path.join(dir_public, "eval.tf_record")
    if not os.path.exists(eval_file):
        file_based_convert_examples_to_features(eval_examples, label_list,  FLAGS.max_seq_length_A, FLAGS.max_seq_length_B,  eval_file)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                   len(eval_examples), num_actual_eval_examples,
                   len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_steps = None
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                seq_length_A=FLAGS.max_seq_length_A,
                                                seq_length_B=FLAGS.max_seq_length_B,
                                                is_training=False,
                                                drop_remainder=eval_drop_remainder,)

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(dir_public, "predict.tf_record")
    if not os.path.exists(predict_file):
        file_based_convert_examples_to_features(predict_examples, label_list,
                                               FLAGS.max_seq_length_A, FLAGS.max_seq_length_B,  predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                   len(predict_examples), num_actual_predict_examples,
                   len(predict_examples) - num_actual_predict_examples)
    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length_A=FLAGS.max_seq_length_A,
        seq_length_B=FLAGS.max_seq_length_B,
        is_training=False,
        drop_remainder=predict_drop_remainder, )

    max_f1 = 0
    max_epoch = 0
    for epoch in range(int(total_num_train_steps/each_num_train_steps)):
#    for epoch in range(1):
        if FLAGS.do_train:
            estimator.train(input_fn=train_input_fn, max_steps=each_num_train_steps*(epoch+1))
        if FLAGS.do_eval:
            estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        if not FLAGS.do_predict:
            continue
        result = estimator.predict(input_fn=predict_input_fn)

        tf.logging.info("***** Test results for epoch {} *****".format(epoch))
        correct = 0
        total_count = 0
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results_epoch_{}.tsv".format(epoch))
        if task_name == "rerank":
            infile = open("./data/disambi/test.txt").readlines()
        elif task_name == "ecd":
            infile = open("./data/E-commerce_dataset/test.txt").readlines()
        else:
            raise ValueError("Task doesn't in rerank or ecd, please check it")
        outf = open(FLAGS.output_dir + "test.txt", "w")
        _gold_ids = []
        _pred_ids = []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            for (i, prediction) in enumerate(result):
                line = eval(infile[i].strip())
                pred_ids = prediction["pred_ids"]
                _pred_ids.append(pred_ids)
                _gold_ids.append(prediction["label_ids"])

                correct += prediction["label_ids"] == pred_ids
                total_count += 1
                result = str(pred_ids)
                label_ids = prediction["label_ids"]
                if i >= num_actual_predict_examples:
                    break
                output_line = str(result)  + "\t" + str(label_ids) + "\n"
                line["pred_tag"] = pred_ids
                if int(pred_ids) != int(label_ids):
                    outf.write(str(line) + "\n")
                writer.write(output_line)
                num_written_lines += 1

        assert num_written_lines == num_actual_predict_examples
        macro_f1 = f1_score(_gold_ids, _pred_ids, average="macro")
        micro_f1 = f1_score(_gold_ids, _pred_ids, average="micro")
        tf.logging.info("*** Accuracy for BIO: {}".format(correct/total_count))
        tf.logging.info("*** Macro F1 is: {}".format(macro_f1))
        tf.logging.info("*** Micro F1 is: {}".format(micro_f1))

        if micro_f1 > max_f1:
            max_f1 = micro_f1
            max_epoch = epoch

    tf.logging.info("The best F1 is {}, appear in the epoch {}".format(max_f1, max_epoch))
    estimator._export_to_tpu = False
    estimator.export_savedmodel(FLAGS.serving_model_save_path, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.app.run()
