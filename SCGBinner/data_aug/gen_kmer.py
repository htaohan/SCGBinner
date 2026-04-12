#!/usr/bin/env python
from __future__ import print_function
import gzip
import os
import csv
import numpy as np
from itertools import product
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor


def window(seq, k):
    return (tuple(seq[i:i+k]) for i in range(len(seq) - k + 1))


def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A":"T", "T":"A", "G":"C", "C":"G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash, counter


# 子进程处理一个 contig
def process_one_contig(args):
    contig_id, seq_str, kmer_len, kmer_dict, nr_features, length_threshold = args
    if len(seq_str) <= length_threshold:
        return None
    vec = np.zeros(nr_features, dtype=np.int64)
    for kmer in window(seq_str.upper(), kmer_len):
        if kmer in kmer_dict:
            vec[kmer_dict[kmer]] += 1
    return contig_id, vec


def _detect_sequence_format(handle):
    first_char = handle.read(1)
    handle.seek(0)
    if first_char == '@':
        return 'fastq'
    if first_char == '>':
        return 'fasta'
    raise RuntimeError('Invalid sequence file: %s' % handle.name)


def generate_features_from_fasta(re_seqs: dict, length_threshold: int, kmer_len: int, outfile: str, num_threads: int):
    """
    Generate composition features from a FASTA sequence dictionary.
    """
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    contig_args = [
        (seq_id, sequence, kmer_len, kmer_dict, nr_features, length_threshold)
        for seq_id, sequence in re_seqs.items()
    ]

    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([''] + [str(i) for i in range(nr_features)])

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(process_one_contig, contig_args, chunksize=16):
                if result is None:
                    continue
                contig_id, vec = result
                writer.writerow([contig_id] + [float(x) for x in vec])


def generate_features_from_fasta_file(fasta_file: str, length_threshold: int, kmer_len: int, outfile: str, num_threads: int):
    """
    Generate composition features from a FASTA file path.
    """
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    is_gzip = fasta_file.endswith('.gz')
    open_func = gzip.open if is_gzip else open

    with open_func(fasta_file, 'rt') as f:
        file_format = _detect_sequence_format(f)
        contig_args = (
            (record.id, str(record.seq), kmer_len, kmer_dict, nr_features, length_threshold)
            for record in SeqIO.parse(f, file_format)
            if len(record.seq) > length_threshold
        )

        with open(outfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + [str(i) for i in range(nr_features)])

            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for result in executor.map(process_one_contig, contig_args, chunksize=16):
                    if result is None:
                        continue
                    contig_id, vec = result
                    writer.writerow([contig_id] + [float(x) for x in vec])


def run_gen_kmer(re_seqs, fasta_file, length_threshold, kmer_len, num_threads):
    outfile = os.path.join(os.path.dirname(fasta_file), 'kmer_' + str(kmer_len) + '_f' + str(length_threshold) + '.csv')
    if isinstance(re_seqs, str):
        generate_features_from_fasta_file(re_seqs, length_threshold, kmer_len, outfile, num_threads)
    else:
        generate_features_from_fasta(re_seqs, length_threshold, kmer_len, outfile, num_threads)


def run_gen_kmer_from_fasta(fasta_file, length_threshold, kmer_len, num_threads):
    outfile = os.path.join(os.path.dirname(fasta_file), 'kmer_' + str(kmer_len) + '_f' + str(length_threshold) + '.csv')
    generate_features_from_fasta_file(fasta_file, length_threshold, kmer_len, outfile, num_threads)

