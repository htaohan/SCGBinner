# modified from https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/generate_coverage.py
import multiprocessing
import traceback
from multiprocessing.pool import Pool

import os
import subprocess
from atomicwrites import atomic_write
import pandas as pd
import numpy as np
from itertools import groupby

### Return error message when using multiprocessing
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            error(traceback.format_exc())
            raise

        return result


class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)


def _checkback(msg):
    msg[1].info('Processed:{}'.format(msg[0]))


def gen_bedtools_out(bam_file: str, bam_index: int, out: str, logger):
    """
    Call bedtools and generate coverage file.

    :param bam_file: Path to the BAM file used (str).
    :param bam_index: Index for identifying the BAM file (int).
    :param out: Output directory (str).

    :return: A tuple containing the path to the processed BAM file and the logger.
    """
    logger.info('Processing `{}`'.format(bam_file))
    bam_name = os.path.split(bam_file)[-1] + '_{}'.format(bam_index)
    bam_depth = os.path.join(out, '{}_depth.txt'.format(bam_name))

    result = subprocess.run(
        ['bedtools', 'genomecov', '-bga', '-ibam', bam_file],
        capture_output=True,
        check=True,
        text=True
    )
    with open(bam_depth, 'w', buffering=1 << 20) as f:
        f.write(result.stdout)

    return (bam_file, logger)



def run_gen_bedtools_out(bam_file_path: str, out: str, logger, num_process: int = 10):
    """
    Run the `gen_bedtools_out` function for multiple BAM files in parallel using multiprocessing.

    :param bam_file_path: Directory containing BAM files (str).
    :param out: Output directory for storing coverage files (str).
    :param num_process: Number of processes to run in parallel (int, default: 10).

    :return: None
    """
    namelist = bam_file_path.split()
    namelist.sort()

    os.makedirs(out, exist_ok=True)

    pool = LoggingPool(num_process) if num_process != 0 else LoggingPool()

    for i, bam_file in enumerate(namelist):
        bam_index = i
        pool.apply_async(
            gen_bedtools_out,
            args=(bam_file, bam_index, out, logger),
            callback=_checkback)

    pool.close()
    pool.join()


# modifed from https://github.com/BigDataBiology/SemiBin/blob/3bad22c58e710d8a5455f7411bc8d4202d557c61/SemiBin/generate_coverage.py#L5
def calculate_coverage_samplebyindex(depth_file: str, augpredix: str, aug_seq_info_dict: dict, logger, edge: int = 0,
                                     contig_threshold: int = 1000):
    """
    Calculate coverage from a position depth file for a set of contigs by index.

    :param depth_file: Input position depth file generated from bedtools genomecov (str).
    :param augpredix: Prefix used for generating output files (str).
    :param aug_seq_info_dict: Dictionary containing information on contigs (dict).
    :param edge: Number of bases to exclude from the edges of contigs (int, default: 0).
    :param contig_threshold: Threshold for contig length, below which contigs are skipped (int, default: 1000).

    :return: A tuple containing the input depth_file and logger (Tuple[str, logging.Logger).
    """
    contigs = []
    mean_coverage = []
    var_coverage = []

    with open(depth_file) as f:
        for contig_name, group in groupby(f, key=lambda x: x.split('\t', 1)[0]):
            if contig_name not in aug_seq_info_dict:
                continue
            start, end = aug_seq_info_dict[contig_name]
            start_offset = start + edge
            end_offset = end - edge + 1

            total_len = 0.0
            sum_depth = 0.0
            sum_sq_depth = 0.0

            for line in group:
                _, s, e, d = line.strip().split('\t')
                s, e, d = float(s), float(e), float(d)
                overlap_start = max(s, start_offset)
                overlap_end = min(e, end_offset)
                overlap_len = max(0.0, overlap_end - overlap_start)

                if overlap_len > 0:
                    total_len += overlap_len
                    sum_depth += d * overlap_len
                    sum_sq_depth += d * d * overlap_len

            if total_len < contig_threshold:
                continue

            mean = sum_depth / total_len
            var = sum_sq_depth / total_len - mean ** 2

            contigs.append(contig_name)
            mean_coverage.append(mean)
            var_coverage.append(var)

    contig_cov = pd.DataFrame({f'{depth_file}_cov': mean_coverage}, index=contigs)
    contig_var = pd.DataFrame({f'{depth_file}_var': var_coverage}, index=contigs)

    with atomic_write(f'{depth_file}_{augpredix}_data_cov.csv', mode='w', buffering=1 << 20, overwrite=True) as f_cov:
        contig_cov.to_csv(f_cov, sep='\t')

    with atomic_write(f'{depth_file}_{augpredix}_data_var.csv', mode='w', buffering=1 << 20, overwrite=True) as f_var:
        contig_var.to_csv(f_var, sep='\t')

    return (depth_file, logger)


def calculate_coverage(depth_file: str, logger, edge: int = 0,
                       contig_threshold: int = 1000, sep: str = None,
                       contig_threshold_dict: dict = None):
    """
    Calculate coverage based on a position depth file generated from mosdepth or bedtools genomecov.

    :param depth_file: Path to the position depth file (str).
    :param edge: Unused parameter, kept for compatibility (int, default: 0).
    :param contig_threshold: Threshold of contigs for must-link constraints (int, default: 1000).
    :param sep: Separator for multi-sample binning (str, default: None).
    :param contig_threshold_dict: Dictionary of contig thresholds by sample (dict, default: None).

    :return: None
    """
    mean_coverage = []
    var_coverage = []
    contigs = []

    with open(depth_file) as f:
        for contig_name, group in groupby(f, key=lambda x: x.split('\t', 1)[0]):
            values = []
            for line in group:
                line_split = line.strip().split('\t')
                start, end = int(line_split[1]), int(line_split[2])
                value = int(float(line_split[3]))
                values.append((value, end - start))

            cov_threshold = contig_threshold if sep is None else contig_threshold_dict[contig_name.split(sep)[0]]
            total_len = sum(l for _, l in values)
            if total_len <= cov_threshold:
                continue

            #depth_value_np = np.concatenate([np.full(l, v, dtype=np.uint16) for v, l in values])
            mean = sum(v * l for v, l in values) / total_len
            mean_sq = sum((v ** 2) * l for v, l in values) / total_len
            var = mean_sq - mean ** 2

            contigs.append(contig_name)
            mean_coverage.append(mean)
            var_coverage.append(var)

    contig_cov = pd.DataFrame({f'{depth_file}_cov': mean_coverage}, index=contigs)
    contig_var = pd.DataFrame({f'{depth_file}_var': var_coverage}, index=contigs)

    with atomic_write(f"{depth_file}_aug0_data_cov.csv", mode='w', buffering=1 << 20, overwrite=True) as f_cov:
        contig_cov.to_csv(f_cov, sep='\t')

    with atomic_write(f"{depth_file}_aug0_data_var.csv", mode='w', buffering=1 << 20, overwrite=True) as f_var:
        contig_var.to_csv(f_var, sep='\t')

    return (depth_file, logger)

def merge_coverage_files(file_list: list, output_file: str):
    df_list = [pd.read_csv(f, sep='\t', header=0, index_col=0) for f in file_list]
    merged_df = pd.concat(df_list, axis=1, join='inner')
    merged_df.to_csv(output_file, sep='\t', header=True)

def gen_cov_from_bedout(logger, out_path: str, depth_file_path: str,
                        num_process: int = 10, num_aug: int = 5, edge: int = 0, contig_len: int = 1000):
    """
    Generate coverage data from bedtools output for original and augmented sequences.

    :param out_path: Output directory for storing coverage files (str).
    :param depth_file_path: Directory containing depth files (str).
    :param num_process: Number of processes to run in parallel (int, default: 10).
    :param num_aug: Number of augmented datasets (int, default: 5).
    :param edge: Number of bases at contig edges to exclude (int, default: 0).
    :param contig_len: Minimum contig length for inclusion (int, default: 1000).

    :return: None
    """
    ########
    filenames = os.listdir(depth_file_path)
    namelist = []
    for filename in filenames:
        if filename.endswith('_depth.txt'):
            namelist.append(filename)

    namelist.sort()

    pool = LoggingPool(num_process) if num_process != 0 else LoggingPool()
    ##generate coverage for original data
    for i in range(len(namelist)):
        depth_file = depth_file_path + namelist[i]
        pool.apply_async(
            calculate_coverage,
            args=(depth_file, logger, edge, contig_len),
            callback=_checkback)

    pool.close()
    pool.join()

    # merge coverage files
    cov_files = [depth_file_path + name + '_aug0_data_cov.csv' for name in namelist]
    outfile = out_path + 'aug0_datacoverage_mean.tsv'
    merge_coverage_files(cov_files, outfile)

    var_files = [depth_file_path + name + '_aug0_data_var.csv' for name in namelist]
    outfile = out_path + 'aug0_datacoverage_var.tsv'
    merge_coverage_files(var_files, outfile)

    for i in range(num_aug):
        outdir = out_path + 'aug' + str(i + 1)
        aug_seq_info_out_file = outdir + '/sequences_aug' + str(i + 1) + '.fasta' + '.aug_seq_info.tsv'
        aug_seq_info_dict = read_aug_seq_info(aug_seq_info_out_file)

        # num_process = 40
        pool = LoggingPool(num_process) if num_process != 0 else LoggingPool()

        ####generate coverage files
        for nameid in range(len(namelist)):
            depth_file = depth_file_path + namelist[nameid]
            pool.apply_async(
                calculate_coverage_samplebyindex,
                args=(depth_file, 'aug' + str(i + 1), aug_seq_info_dict, logger, edge, contig_len),
                callback=_checkback)

        pool.close()
        pool.join()

        # merge coverage files
        cov_files = [depth_file_path + name + f'_aug{i + 1}_data_cov.csv' for name in namelist]
        outfile = out_path + f'aug{i + 1}_datacoverage_mean.tsv'
        merge_coverage_files(cov_files, outfile)

        var_files = [depth_file_path + name + f'_aug{i + 1}_data_var.csv' for name in namelist]
        outfile = out_path + f'aug{i + 1}_datacoverage_var.tsv'
        merge_coverage_files(var_files, outfile)

        logger.info("Finish calculating coverage for aug_"+str(i+1))


def read_aug_seq_info(aug_seq_info_out_file):
    aug_seq_info = pd.read_csv(aug_seq_info_out_file, sep='\t', header=0).values[:]
    aug_seq_info_dict = {}
    for i in range(len(aug_seq_info)):
        aug_seq_info_dict[aug_seq_info[i][0]] = [aug_seq_info[i][1], aug_seq_info[i][2]]

    return aug_seq_info_dict


def run_gen_cov(logger, args):
    logger.info("Generate coverage files from bam files.")

    bam_file_path = args.bam_file_path
    # if not bam_file_path.endswith('/'):
    #     bam_file_path = bam_file_path + '/'

    if not args.out_augdata_path.endswith('/'):
        args.out_augdata_path = args.out_augdata_path + '/'

    out = args.out_augdata_path + 'depth/'
    #生成每个contig的 start 到end的覆盖度
    run_gen_bedtools_out(bam_file_path, out, logger,num_process=args.num_threads)
    gen_cov_from_bedout(logger, args.out_augdata_path, out, num_aug=args.n_views-1, contig_len=args.contig_len,num_process=args.num_threads)
