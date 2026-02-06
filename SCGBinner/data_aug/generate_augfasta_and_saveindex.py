from Bio import SeqIO
import mimetypes
import os
import gzip
import random
import shutil
from typing import Dict

def get_inputsequences(fastx_file: str):
    """
    Retrieve sequences from a FASTX file and return them as a dictionary.

    :param fastx_file: Path to the FASTX file (either FASTA or FASTQ).
    :return: A dictionary where sequence IDs are keys and sequences are values.
    """
    if os.path.getsize(fastx_file) == 0:
        return {}

    # Check if gzip-compressed
    is_gzip = fastx_file.endswith(".gz")
    open_func = gzip.open if is_gzip else open

    # Detect file format
    with open_func(fastx_file, "rt") as f:
        first_char = f.read(1)
        file_format = "fastq" if first_char == '@' else "fasta" if first_char == '>' else None

    if not file_format:
        raise RuntimeError(f"Invalid sequence file: '{fastx_file}'")

    # Read and convert sequences
    with open_func(fastx_file, "rt") as f:
        return {record.id: record.seq for record in SeqIO.parse(f, file_format)}


def gen_augfasta(seqs: Dict[str, str], augprefix: str, out_file: str,
                  contig_len: int = 1000):
    """
    Generate augmented sequences and save them to a FASTA file along with sequence information.

    :param seqs: A dictionary of input sequences where keys are sequence IDs, and values are sequences.
    :param augprefix: A prefix used in the augmented sequence IDs.
    :param out_file: Path to the output FASTA file.
    :param p: Proportion of the original sequence to include in the augmented sequences (default is None).
    :param contig_len: Minimum length of the original sequence required for augmentation (default is 1000).
    """
    seqkeys = [k for k, v in seqs.items() if len(v) >= contig_len + 1]
    re_seqs = {}
    aug_seq_info = []
    for seqid in seqkeys:
        cur_seq_len = len(seqs[seqid])
        mid = cur_seq_len // 2
        segment_lenth = cur_seq_len // 3
        if cur_seq_len >= 3000:
            if augprefix == 'aug1':
                start = 0
                sim_len = mid
                end = start + sim_len - 1
            elif augprefix == 'aug2':
                start = mid
                sim_len = cur_seq_len - mid
                end = start + sim_len - 1
            elif augprefix == 'aug3':
                start = 0
                sim_len = segment_lenth
                end = start + sim_len - 1
            elif augprefix == 'aug4':
                start = segment_lenth
                sim_len = segment_lenth
                end = start + sim_len - 1
            elif augprefix == 'aug5':
                start = segment_lenth * 2
                sim_len = cur_seq_len - (2 * segment_lenth)
                end = start + sim_len - 1
        elif 3000 > cur_seq_len >= 2000:
            if augprefix == 'aug1':
                start = 0
                sim_len = mid
                end = mid - 1
            elif augprefix == 'aug2':
                start = mid
                sim_len = cur_seq_len - mid
                end = cur_seq_len - 1
            else:
                start = random.randint(0, len(seqs[seqid]) - (contig_len + 1))
                sim_len = random.randint(contig_len, len(seqs[seqid]) - start)
                end = start + sim_len - 1
                # gen_seqs_dict[genome_name+"_sim_"+str(sim_count)] =seqs[seqid][start:end+1]

        elif 2000 > cur_seq_len:
            start = random.randint(0, len(seqs[seqid]) - (contig_len + 1))
            sim_len = random.randint(contig_len, len(seqs[seqid]) - start)
            end = start + sim_len - 1
            # gen_seqs_dict[genome_name+"_sim_"+str(sim_count)] =seqs[seqid][start:end+1]
        sequence = str(seqs[seqid][start:end + 1])
        re_seqs[seqid] = sequence
        aug_seq_info.append((seqid, start, end, sim_len))


    aug_seq_info_out_file = f"{out_file}.aug_seq_info.tsv"

    with open(aug_seq_info_out_file, 'w') as afile:
        afile.write('seqid\tstart\tend\tlength\n')
        for sid, start, end, length in aug_seq_info:
            afile.write(f'{sid}\t{start}\t{end}\t{length}\n')
    return re_seqs


def run_gen_augfasta(logger, args):
    """
    Generate augmentation fasta file and save index
    """
    num_aug = args.n_views - 1  # Generate several copies of augmented data
    fasta_file = args.contig_file
    out_path = args.out_augdata_path
    contig_len = args.contig_len
    num_threads = args.num_threads

    outdir = out_path + '/aug0'
    os.makedirs(outdir)
    out_file = outdir + '/sequences_aug0.fasta'
    shutil.copyfile(fasta_file, out_file)

    from .gen_kmer import run_gen_kmer

    seqs = get_inputsequences(fasta_file)
    run_gen_kmer(seqs, out_file, 0, 4, num_threads)
    for i in range(num_aug):
        outdir = out_path + '/aug' + str(i + 1)
        os.makedirs(outdir)
        logger.info("aug:\t" + str(i+1))

        out_file = outdir + '/sequences_aug' + str(i + 1) + '.fasta'
        #生成aug的fasta和 fasta_seq_info
        re_seqs = gen_augfasta(seqs, 'aug' + str(i + 1), out_file,  contig_len=contig_len)

        run_gen_kmer(re_seqs, out_file, 0, 4, num_threads)
