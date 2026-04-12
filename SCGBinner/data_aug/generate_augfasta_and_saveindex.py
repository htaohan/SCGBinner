from Bio import SeqIO
import mimetypes
import os
import gzip
import random
import shutil
from typing import Dict, Iterator, Tuple


def _detect_sequence_format(handle):
    first_char = handle.read(1)
    handle.seek(0)
    if first_char == '@':
        return 'fastq'
    if first_char == '>':
        return 'fasta'
    raise RuntimeError(f"Invalid sequence file: '{handle.name}'")


def iter_sequences(fastx_file: str) -> Iterator[Tuple[str, str]]:
    if os.path.getsize(fastx_file) == 0:
        return

    is_gzip = fastx_file.endswith('.gz')
    open_func = gzip.open if is_gzip else open

    with open_func(fastx_file, 'rt') as f:
        file_format = _detect_sequence_format(f)
        for record in SeqIO.parse(f, file_format):
            yield record.id, str(record.seq)


def _select_aug_segment(seq: str, augprefix: str, contig_len: int = 1000):
    cur_seq_len = len(seq)
    if cur_seq_len < contig_len + 1:
        return None

    mid = cur_seq_len // 2
    segment_length = cur_seq_len // 3

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
            sim_len = segment_length
            end = start + sim_len - 1
        elif augprefix == 'aug4':
            start = segment_length
            sim_len = segment_length
            end = start + sim_len - 1
        elif augprefix == 'aug5':
            start = segment_length * 2
            sim_len = cur_seq_len - (2 * segment_length)
            end = start + sim_len - 1
        else:
            return None
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
            start = random.randint(0, cur_seq_len - (contig_len + 1))
            sim_len = random.randint(contig_len, cur_seq_len - start)
            end = start + sim_len - 1
    else:
        start = random.randint(0, cur_seq_len - (contig_len + 1))
        sim_len = random.randint(contig_len, cur_seq_len - start)
        end = start + sim_len - 1

    return seq[start:end + 1], start, end, sim_len


def gen_augfasta(fastx_file: str, augprefix: str, out_file: str,
                  contig_len: int = 1000):
    """
    Generate augmented sequences and save them to a FASTA file along with sequence information.

    :param fastx_file: Path to the input FASTA/FASTQ file.
    :param augprefix: A prefix used in the augmented sequence IDs.
    :param out_file: Path to the output FASTA file.
    :param contig_len: Minimum length of the original sequence required for augmentation (default is 1000).
    """
    aug_seq_info = []
    with open(out_file, 'w') as outfile:
        for seqid, seq in iter_sequences(fastx_file):
            selected = _select_aug_segment(seq, augprefix, contig_len)
            if selected is None:
                continue
            sequence, start, end, sim_len = selected
            outfile.write(f'>{seqid}\n{sequence}\n')
            aug_seq_info.append((seqid, start, end, sim_len))

    aug_seq_info_out_file = f"{out_file}.aug_seq_info.tsv"
    with open(aug_seq_info_out_file, 'w') as afile:
        afile.write('seqid\tstart\tend\tlength\n')
        for sid, start, end, length in aug_seq_info:
            afile.write(f'{sid}\t{start}\t{end}\t{length}\n')


def run_gen_augfasta(logger, args):
    """
    Generate augmentation fasta file and save index
    """
    num_aug = args.n_views - 1  # Generate several copies of augmented data
    fasta_file = args.contig_file
    out_path = args.out_augdata_path
    contig_len = args.contig_len
    num_threads = args.num_threads

    from .gen_kmer import run_gen_kmer_from_fasta

    outdir = os.path.join(out_path, 'aug0')
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, 'sequences_aug0.fasta')
    if fasta_file.endswith('.gz'):
        with gzip.open(fasta_file, 'rt') as inf, open(out_file, 'wt') as outf:
            shutil.copyfileobj(inf, outf)
    else:
        try:
            os.link(fasta_file, out_file)
        except OSError:
            shutil.copyfile(fasta_file, out_file)

    run_gen_kmer_from_fasta(out_file, 0, 4, num_threads)
    for i in range(num_aug):
        outdir = os.path.join(out_path, f'aug{i + 1}')
        os.makedirs(outdir, exist_ok=True)
        logger.info('aug:\t' + str(i + 1))

        out_file = os.path.join(outdir, f'sequences_aug{i + 1}.fasta')
        gen_augfasta(fasta_file, f'aug{i + 1}', out_file, contig_len=contig_len)
        run_gen_kmer_from_fasta(out_file, 0, 4, num_threads)
