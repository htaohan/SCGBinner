# modified from https://github.com/BigDataBiology/SemiBin
import os
import random
import argparse
from typing import Iterator, Tuple
from Bio import SeqIO
import gzip
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

def main():
    parser = argparse.ArgumentParser(
    description="Split contigs into five different views"
)
    parser.add_argument(
        "-a", "--contig_file",
        required=True,
        help="Contig file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output fasta file with augmented contigs"
    )
    parser.add_argument(
        "-l", "--contig_min_len",
        default=1000,
        type=int,
        help="Minimum contig length"
    )
    args = parser.parse_args()

    contig_file = args.contig_file
    output_file = args.output
    contig_min_len = args.contig_min_len

    with open(output_file, "w") as ofile:
        for seq_id, seq in iter_sequences(contig_file):
            cur_seq_len = len(seq)
            if cur_seq_len < contig_min_len + 1:
                continue

            mid = cur_seq_len // 2
            segment_length = cur_seq_len // 3
            ##aug0###
            ofile.write(f">{seq_id}_aug0\n{seq}\n")
            if cur_seq_len >= 3000:
                ##aug1##
                start = 0
                sim_len = mid
                end = start + sim_len
                ofile.write(f">{seq_id}_aug1\n{seq[start:end]}\n")
                ##aug2##
                start = mid
                sim_len = cur_seq_len - mid
                end = start + sim_len
                ofile.write(f">{seq_id}_aug2\n{seq[start:end]}\n")
                ##aug3##
                start = 0
                sim_len = segment_length
                end = start + sim_len
                ofile.write(f">{seq_id}_aug3\n{seq[start:end]}\n")
                ##aug4##
                start = segment_length
                sim_len = segment_length
                end = start + sim_len
                ofile.write(f">{seq_id}_aug4\n{seq[start:end]}\n")
                ##aug5##
                start = segment_length * 2
                sim_len = cur_seq_len - (2 * segment_length)
                end = start + sim_len
                ofile.write(f">{seq_id}_aug5\n{seq[start:end]}\n")

            elif 3000 > cur_seq_len >= 2000:
                ##aug1##
                start = 0
                end = mid
                ofile.write(f">{seq_id}_aug1\n{seq[start:end]}\n")
                ##aug2##
                start = mid
                end = cur_seq_len
                ofile.write(f">{seq_id}_aug2\n{seq[start:end]}\n")
                ##aug3-5##
                for i in range(3, 6):
                    start = random.randint(0, cur_seq_len - (contig_min_len + 1))
                    sim_len = random.randint(contig_min_len, cur_seq_len - start)
                    end = start + sim_len
                    ofile.write(f">{seq_id}_aug{i}\n{seq[start:end]}\n")
            else:
                ##aug1-5##
                for i in range(1, 6):
                    start = random.randint(0, cur_seq_len - (contig_min_len + 1))
                    sim_len = random.randint(contig_min_len, cur_seq_len - start)
                    end = start + sim_len
                    ofile.write(f">{seq_id}_aug{i}\n{seq[start:end]}\n")

if __name__ == "__main__":
    main()
