#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import string
import codecs
import subprocess
import subprocess as sp
import os

def read_dataset(infile):
    if not os.path.exists(infile):
        raise ValueError("输入文件不存在: {}".format(infile))
    line_num = sp.getoutput("wc -l {}".format(infile))
    line_num = int(line_num.split()[0])
    inf = codecs.open(infile, "r", "utf-8")
    for i in range(line_num):
        yield inf.readline()

def get_tokens(text):
    """ 移除标签符号并分词 """
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    """ 去除停用词 """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def dict_writer(indict, outfile):
    outwriter = codecs.open(outfile, "w", "utf-8")
    for key in indict:
        outwriter.write(str(key) + str(indict[key]) + "\n")
    outwriter.close()

def list_writer(indict, outfile):
    outwriter = codecs.open(outfile, "w", "utf-8")
    for key in indict:
        outwriter.write(str(key) + "\n")
    outwriter.close()

def divide(path, infile):
    total_line = int(subprocess.getoutput("wc -l {}".format(infile)).split()[0])
    datasets = read_dataset(infile)
    
    train_writer = codecs.open(os.path.join(path, "train.txt"), "w", "utf-8")
    dev_writer = codecs.open(os.path.join(path, "dev.txt"), "w", "utf-8")
    test_writer = codecs.open(os.path.join(path, "test.txt"), "w", "utf-8")
    for idx, data in enumerate(datasets):
        if idx < 0.8*total_line:
            train_writer.write(data)
        elif idx < 0.9*total_line:
            dev_writer.write(data)
        else:
            test_writer.write(data)
    train_writer.close()
    dev_writer.close()
    test_writer.close()

