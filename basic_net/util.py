#!/usr/bin/env python
# coding=utf-8

def build_pretrain_vec(infile, embedding_size):
    lines = open("./data/poi_word_vec.txt").readlines()
    pretrain_word_vec = []
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        line = line.strip().split()
        line = [float(l) for l in line[1:]]
        assert len(line) == embedding_size
        pretrain_word_vec.append(line)
    return pretrain_word_vec
