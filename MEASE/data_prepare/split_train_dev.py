import os
import torch
import random
import json
import codecs
import logging
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Prepare clean information json.', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_file', metavar='<file>', required=True,
                    help='input wav.scp.')
parser.add_argument('--output_file', metavar='<file>', required=True,
                    help='split testset.')
parser.add_argument('--number', metavar='<file>', required=True,
                    help='number')
args = parser.parse_args()

random.seed(20)

resultFile = args.output_file
fresultFile = open(resultFile, 'w')
def read_scp(txt_file):
    txt = open(txt_file, 'r')
    keys = []
    for line in txt.readlines():
        keys.append(line[:-1])
    return keys

def split(scp_path, testnum):
    keys = read_scp(scp_path)
    whole_num = len(keys)
    indices = np.arange(whole_num)
    random.shuffle(indices)
    testset_index = indices[0: testnum]
    trainset_index = indices[testnum: ]
    for i in testset_index:
        fresultFile.write(keys[i] + '\n')

split(args.input_file, int(args.number))