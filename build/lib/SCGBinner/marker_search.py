import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from tempfile import mkstemp
from subprocess import DEVNULL, PIPE, STDOUT, run
from multiprocessing import Pool
from math import ceil, inf
import numpy
def initialize_parser(parameters):
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Search markers.',
    )

    subparser = parser.add_subparsers(title = 'program', dest = 'program', required = True, help = 'Run coverage|seed|cluster --help for help.')

    seed_parser = subparser.add_parser('seed', formatter_class = RawTextHelpFormatter, help = 'Map the marker genes (HMMs) to sequences.\nUsage: metadecoder seed -f input.fasta -o output.seed.')
    seed_parser.add_argument(
        '-f', '--fasta', default = '', type = str, required = True, metavar = 'str',
        help = 'The fasta formatted assembly file.'
    )
    seed_parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = 'str',
        help = 'The output seed file.'
    )
    seed_parser.add_argument(
        '--threads', default = 20, type = int, required = False, metavar = 'int',
        help = 'The number of threads.\nThe value should be a positive integer.\nDefault: 20.'
    )
    seed_parser.add_argument(
        '--coverage', default = 0.5, type = float, required = False, metavar = 'float',
        help = 'The min coverage of domain.\nThe value should be from 0 to 1.\nDefault: 0.5.'
    )
    seed_parser.add_argument(
        '--accuracy', default = 0.6, type = float, required = False, metavar = 'float',
        help = 'The min accuracy.\nThe value should be from 0 to 1.\nDefault: 0.6.',
    )
    return parser.parse_args(parameters)

def parse_parameters(parameters):
    if parameters.program == 'seed':
        # all files are readable.
        parameters.fasta = os.path.abspath(parameters.fasta)
        #is_readable(parameters.fasta)
        # the output file is writeable.
        #is_writeable(parameters.output)
        parameters.output = os.path.abspath(parameters.output)

def make_file(prefix = None, suffix = None, folder=None):
    '''
    Parameters:
        prefix: the prefix of the temp file [None].
        suffix: the suffix of the temp file [None].
    Return:
        the path to the temp file.
    '''
    file_descriptor, file = mkstemp(prefix = prefix, suffix = suffix, dir=folder)
    os.close(file_descriptor)
    return file

def worker(command, message):
    assert not run(command, stdout = DEVNULL, stderr = DEVNULL).returncode, message
    return None

def run_fraggenescan(fraggenescan, input_fasta, output_fasta, threads):
    '''
    Run FragGeneScan to predict all protein sequences.
    '''
    pkg_dir = os.path.dirname(__file__)
    fraggenescan_path = os.path.join(pkg_dir, "fraggenescan")
    worker(
        [fraggenescan_path, '-s', input_fasta, '-o', output_fasta, '-w', '0', '-t', 'complete', '-p', str(threads)],
        'An error has occured while running fraggenescan.'
    )
    os.remove(output_fasta + '.ffn')
    os.remove(output_fasta + '.out')
    os.replace(output_fasta + '.faa', output_fasta)
    return None
def read_fasta_file(input_fasta):
    '''
    Parameters:
        input_fasta: the path to the fasta file.
    Return:
        a generator (sequence_id, sequence)
    '''
    container = list()
    open4r = open(input_fasta, 'r')
    for line in open4r:
        line = line.rstrip('\n')
        if line.startswith('>'):
            if container:
                yield (sequence_id, ''.join(container))
            sequence_id = line.split(' ', maxsplit = 1)[0][1 : ]
            container.clear()
        else:
            container.append(line)
    yield (sequence_id, ''.join(container))
    open4r.close()
    return None
def parse_sequence_id(file):
    '''
    fraggenescan: >id_start_end_strand
    '''
    gene = 1
    output = os.path.basename(file) + '.proteins'
    open_file = open(output, 'w')
    for sequence_id, sequence in read_fasta_file(file):
        open_file.write('>' + str(gene) + '_' + sequence_id.rsplit('_', maxsplit = 3)[0] + '\n')
        open_file.write(sequence + '\n')
        gene += 1
    open_file.close()
    return output
def split_fasta(input_fasta, output_files):
    '''
    Split a fasta into small files.
    Parameters:
        input_fasta: the path to the fasta file.
        output_files: the number of output files.
    Return:
        a generator of path of each output file.
    '''
    total_size = os.path.getsize(input_fasta)
    block_size = ceil(total_size / output_files) # block_size <= total_size #
    file_position = 0
    file_position_ = 0
    open4r = open(input_fasta, 'rb')
    while file_position < total_size:
        line = open4r.readline()
        file_position += len(line)
        if line.startswith(b'>'):
            file_position -= len(line)
            if file_position > 0:
                open4r.seek(file_position_, os.SEEK_SET)
                output_file = make_file()
                open4w = open(output_file, 'wb')
                while file_position_ < file_position:
                    file_position_ += open4w.write(open4r.read(min(10485760, file_position - file_position_)))
                open4w.close()
                yield output_file
                # file_position_ will be equal to file_position, open4r.tell() will be equal to file_position #
            file_position = open4r.seek(min(file_position + block_size, total_size), os.SEEK_SET)
    open4r.seek(file_position_, os.SEEK_SET)
    output_file = make_file()
    open4w = open(output_file, 'wb')
    while file_position_ < file_position:
        file_position_ += open4w.write(open4r.read(min(10485760, file_position - file_position_)))
    open4w.close()
    yield output_file
    open4r.close()
    return None

def run_hmmsearch(hmmsearch, input_hmm, input_fasta, output_file, threads):
    '''
    Run Hmmsearch to map hmms to sequences.
    '''
    input_fastas = list()
    output_files = list()
    process_pool = Pool(os.cpu_count())
    pkg_dir = os.path.dirname(__file__)
    hmmsearch_path = os.path.join(pkg_dir, "hmmsearch") 
    #
    for INPUT_FASTA in split_fasta(input_fasta, threads):
        input_fastas.append(INPUT_FASTA)
        output_files.append(make_file())
        process_pool.apply_async(
            worker,
            (
                [hmmsearch_path, '--cpu', '1', '--noali', '--domtblout', output_files[-1], input_hmm, INPUT_FASTA],
                'An error has occured while running hmmsearch.'
            )
        )
    process_pool.close()
    process_pool.join()
    for INPUT_FASTA in input_fastas:
        os.remove(INPUT_FASTA)
    open4w = open(output_file, 'wb')
    for OUTPUT_FILE in output_files:
        open4r = open(OUTPUT_FILE, 'rb')
        while open4w.write(open4r.read(1024)):
            pass
        open4r.close()
        os.remove(OUTPUT_FILE)
    open4w.close()
    return None

def read_hmm_file(input_hmm):
    model2tc = dict()
    open_file = open(input_hmm, 'r')
    for line in open_file:
        if line.startswith('NAME'):
            model = line.rstrip('\n').split()[1]
        elif line.startswith('TC'):
            score1, score2 = line.rstrip(';\n').split()[1 : ]
            model2tc[model] = (float(score1), float(score2))
    open_file.close()
    return model2tc

def get_seeds(file, model2tc, coverage, accuracy, output):
    model2sequences = dict()
    open_file = open(file, 'r', encoding = 'utf-8')
    for line in open_file:
        if not line.startswith('#'):
            lines = line.rstrip('\n').split()
            tc = model2tc.get(lines[3], (-inf, -inf))
            if float(lines[7]) >= tc[0] and float(lines[13]) >= tc[1] and (int(lines[16]) - int(lines[15]) + 1) / int(lines[5]) >= coverage and float(lines[21]) >= accuracy:
                model2sequences.setdefault(lines[3], list()).append(lines[0].split('_', maxsplit = 1)[1])
    open_file.close()
    open_file = open(output, 'w')
    for model, sequences in model2sequences.items():
        open_file.write(model + '\t' + '\t'.join(list(set(sequences))) + '\n')
    open_file.close()

def extract_3rd_quartile_gene(logger, input_file, output_file):
    gene_contig_dict = {}

    #
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            gene = parts[0]
            contigs = parts[1:]
            if len(contigs) > 0:
                gene_contig_dict[gene] = contigs

    #
    gene_counts = [(gene, len(contigs)) for gene, contigs in gene_contig_dict.items()]
    #
    gene_counts.sort(key=lambda x: x[1], reverse=True)

    #
    total = len(gene_counts)
    q3_index = int(0.3 * total)
    q3_index = min(q3_index, total - 1)  #

    #
    target_gene, count = gene_counts[q3_index]
    target_contigs = gene_contig_dict[target_gene]

    logger.info(f"{q3_index + 1} / {total}")
    logger.info(f"{target_gene}， {count}")
    logger.info(f"{output_file}")

    #
    with open(output_file, 'w') as fout:
        for contig in target_contigs:
            fout.write(contig + '\n')

def gen_scg_file(logger, contig_file: str, output_path: str, threads: int):
    PROTEIN = make_file(folder=output_path)
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '->' + 'Identifying protein sequences.')
    logger.info(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fraggenescan'))
    run_fraggenescan(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fraggenescan'),
        contig_file, PROTEIN, threads
    )
    protein = parse_sequence_id(PROTEIN)
    os.remove(PROTEIN)
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '->'+ 'Done.')
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '->'+ 'Mapping marker genes to protein sequences.')
    hmmsearch_output = make_file(folder=output_path)
    #
    run_hmmsearch(
         os.path.join(os.path.dirname(os.path.realpath(__file__)),'hmmsearch'),
         os.path.join(os.path.dirname(os.path.realpath(__file__)),'markers.hmm'),
        protein, hmmsearch_output, threads
    )
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '->'+ 'Done.')
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '->'+ 'Writing to file.')
    os.remove(protein)
    model2tc = read_hmm_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),'markers.hmm'))
    #
    coverage_thre, accurcy_thre = 0.5, 0.6
    get_seeds(hmmsearch_output, model2tc, coverage_thre, accurcy_thre, output_path + '/fasta.SEED')
    os.remove(hmmsearch_output)
    extract_3rd_quartile_gene(logger, output_path + '/fasta.SEED', output_path + '/fasta_quarter.SEED')
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ '->'+ 'Finished.')