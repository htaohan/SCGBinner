#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from itertools import product
from Bio import SeqIO
# optimized sliding window function from
# http://stackoverflow.com/a/7636587
from itertools import tee
from collections import Counter, OrderedDict
import pandas as p
from concurrent.futures import ProcessPoolExecutor

def window(seq,k):
    return (tuple(seq[i:i+k]) for i in range(len(seq) - k + 1))

def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A":"T","T":"A","G":"C","C":"G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC",repeat=kmer_len):
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash,counter

# 子进程处理一个 contig
def process_one_contig(args):
    contig_id, seq_str, kmer_len, kmer_dict, nr_features, length_threshold = args
    if len(seq_str) <= length_threshold:
        return None
    kmers = [
        kmer_dict[kmer]
        for kmer in window(seq_str.upper(), kmer_len)
        if kmer in kmer_dict
    ]
    vec = np.bincount(kmers, minlength=nr_features)
    return contig_id, vec

def generate_features_from_fasta(re_seqs: dict, length_threshold: int, kmer_len: int, outfile: str, num_threads: int):
    """
    Generate composition features from a FASTA file.

    :param fasta_file: The path to the input FASTA file.
    :param length_threshold: The minimum length of sequences to include in the feature generation.
    :param kmer_len: The length of k-mers to consider.
    :param outfile: The path to the output CSV file where features will be saved.
    """
    #4mer的索引以及特征的维度
    kmer_dict,nr_features = generate_feature_mapping(kmer_len)

    # 准备参数
    contig_args = [
        (seq_id, sequence, kmer_len, kmer_dict, nr_features, length_threshold)
        for seq_id, sequence in re_seqs.items()
    ]

    # 并行处理
    features = {}
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(process_one_contig, contig_args, chunksize=16):
            if result is None:
                continue
            contig_id, vec = result
            features[contig_id] = vec

    # 转换为 DataFrame
    df = p.DataFrame.from_dict(features, orient='index', dtype=float)
    df.to_csv(outfile)


def run_gen_kmer(re_seqs, fasta_file, length_threshold, kmer_len, num_threads):
    outfile = os.path.join(os.path.dirname(fasta_file), 'kmer_' + str(kmer_len) + '_f' + str(length_threshold) + '.csv')
    generate_features_from_fasta(re_seqs,length_threshold,kmer_len,outfile,num_threads)

