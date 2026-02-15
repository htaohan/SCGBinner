import gzip
import os
import shutil
from Bio import SeqIO
from collections import defaultdict


def gen_bins(sequences: dict, dic: dict, outputdir: str) -> None:
    """
    Generate bins from contigs based on a result file and save them to the specified output directory.
    """
    print("Writing bins:\t{}".format(outputdir))
    os.makedirs(outputdir, exist_ok=True)

    bin_name = 0
    for _, cluster in dic.items():
        binfile = os.path.join(outputdir, f"{bin_name}.fa")
        with open(binfile, "w") as f:
            for contig_name in cluster:
                if contig_name in sequences:
                    sequence = sequences[contig_name]
                    f.write(f">{contig_name}\n{sequence}\n")
        bin_name += 1


def filter_small_bins(logger, fastafile: str, resultfile: str, args, minbinsize: int = 200000) -> None:
    """
    Filter small bins from the result file.

    :param fastafile: The path to the input FASTA file containing contigs.
    :param resultfile: The path to the binning result file.
    :param args: The additional arguments used in the process.
    :param minbinsize: The minimum bin size (default: 200,000).
    :return: None
    """
    outputdir = args.output_path + '/best_score.tsv'

    logger.info("Processing file:\t{}".format(fastafile))
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fastafile, "fasta")}

    logger.info("Reading Map:\t{}".format(resultfile))
    dic = defaultdict(list)
    bin_size = defaultdict(int)
    with open(resultfile, "r") as f:
        for line in f:
            contig_name, cluster_name = line.strip().split('\t')
            dic[cluster_name].append(contig_name)
            bin_size[cluster_name] += len(sequences[contig_name])

    to_remove = []
    with open(outputdir, "w") as f:
        for cluster_name, contigs in dic.items():
            if bin_size[cluster_name] >= minbinsize:
                for contig in contigs:
                    f.write(f"{contig}\t{cluster_name}\n")
            else:
                to_remove.append(cluster_name)
    f.close()
    for cluster_name in to_remove:
        del dic[cluster_name]

    gen_bins(sequences, dic, args.output_path + '/best_score_bins')
