import logging
import copy
import os
from Bio import SeqIO
from collections import defaultdict
import pandas as pd
from multiprocessing import Pool, cpu_count
from filter_small_bins import filter_small_bins

from typing import List, Optional, Union, Dict

# for each bin
# update for checkm marker
def get_binstats(bin_contig_names, markers):
    _, comp, cont = markers.bin_quality(bin_contig_names)

    return comp, cont

def read_bins_nosequences(bin_dirs, contig_dict):
    """Read sequences in bins.
    bins: key1 method_id key2 cluster_id value
    contigs_in_bins: key1:contig name. key2:method_id value:cluster_id
    contig_list: seqid list
    Leiden_result_dict: key:method_id value:
    """
    #bins. key1:cluster file name. key2:cluster id. value:contig set
    bins = defaultdict(lambda: defaultdict(set))
    #contigs_in_bins. key1:contig name. key2:cluster file name. value:cluster id
    contigs_in_bins = defaultdict(lambda: {})
    #key:cluster file. value:labels
    Leiden_result_dict = {}
    contig_list = list(contig_dict.keys())
    for method_id, bin_dir in bin_dirs.items():
        #
        cluster_ids_sorted = []
        cur_seqid_2_cluster_id = defaultdict(lambda: -1)
        df = pd.read_csv(bin_dir, sep='\t', header=None)
        namelist = df.iloc[:, 0].values
        cluster_ids = df.iloc[:, 1].values
        #
        cur_seqid_2_cluster_id.update(zip(list(namelist), list(cluster_ids)))
        for item in contig_list:
            cluster_ids_sorted.append(cur_seqid_2_cluster_id[item])
        Leiden_result_dict[method_id] = cluster_ids_sorted
        for i in range(len(namelist)):
            bins[method_id][cluster_ids[i]].add(namelist[i])
            contigs_in_bins[namelist[i]][method_id] = cluster_ids[i]

    return bins, contigs_in_bins, contig_list, Leiden_result_dict

def get_top_10_methods(bin_quality_dict):
    sorted_methods_5010 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_5010'], item[1]['sum'], item[1]['sum_cont5']),
        reverse=True
    )
    sorted_methods_7010 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_7010'], item[1]['sum'], item[1]['sum_cont5']),
        reverse=True
    )
    sorted_methods_9010 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_9010'], item[1]['sum'], item[1]['sum_cont5']),
        reverse=True
    )
    sorted_methods_505 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_505'], item[1]['sum'], item[1]['sum_cont5']),
        reverse=True
    )
    sorted_methods_705 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_705'], item[1]['sum'], item[1]['sum_cont5']),
        reverse=True
    )
    sorted_methods_905 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['num_905'],item[1]['num_705'],item[1]['num_505'], item[1]['sum_cont5'], item[1]['sum'],item[1]['num_9010'],item[1]['num_7010'],item[1]['num_5010'],),
        reverse=True
    )
    sorted_methods_sum = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['sum'], item[1]['sum_cont5'],item[1]['num_905']),
        reverse=True
    )
    sorted_methods_sum_cont5 = sorted(
        bin_quality_dict.items(),
        key=lambda item: (item[1]['sum_cont5'], item[1]['sum'], item[1]['num_905']),
        reverse=True
    )

    top_10_methods = []
    #print("NUM_sorted_methods_905", len(sorted_methods_905))
    for item in sorted_methods_905[:120]:
        top_10_methods.append(item[0])
    return top_10_methods


# update for checkm marker
def savecontigs_with_high_bin_quality(orig_bins: Dict[str, Dict[int, List[int]]],
                                    best_method: str, markers: List[int], outpath: str):
    """
    Save contigs with high bin quality to text files based on specified criteria.

    :param orig_bins: A dictionary of original bins with method IDs as keys and bin IDs as sub-keys.
    :param best_method: The best method to consider.
    :param markers: A list of markers used for bin quality calculations.
    :param outpath: The path to save the output text files.
    :return: None
    """
    bin_count_5010 = 0
    bin_count_5005 = 0
    with open(outpath+'/'+best_method+'5010_res.txt','w') as f1:
        with open(outpath+'/'+best_method+'5005_res.txt','w') as f2:
            for bin_id in orig_bins[best_method]:
                comp, cont = get_binstats(orig_bins[best_method][bin_id], markers)
                if comp > 50 and cont < 10:
                    for key in orig_bins[best_method][bin_id]:
                        f1.write(key+'\t'+str(bin_count_5010)+'\n')
                    bin_count_5010 += 1

                if comp > 50 and cont < 5:
                    for key in orig_bins[best_method][bin_id]:
                        f2.write(key+'\t'+str(bin_count_5005)+'\n')
                    bin_count_5005 += 1


def write_estimated_bin_quality(bin_quality_dict, output_file):
    fout = open(os.path.join(output_file), 'w')
    fout.write('Binning_method\tnum_5010\tnum_7010\tnum_9010\tnum_505\tnum_705\tnum_905\tsum\tsum_cont5\n')
    for method_id in bin_quality_dict:
        fout.write(method_id + '\t' + str(bin_quality_dict[method_id]['num_5010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_7010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_9010']) + '\t'
                   + str(bin_quality_dict[method_id]['num_505']) + '\t'
                   + str(bin_quality_dict[method_id]['num_705']) + '\t'
                   + str(bin_quality_dict[method_id]['num_905']) + '\t'
                   + str(bin_quality_dict[method_id]['sum']) + '\t'
                   + str(bin_quality_dict[method_id]['sum_cont5']) + '\n')

    fout.close()


def estimate_bins_quality_nobins(contig_dict, res_path):
    """
    Estimate the quality of bins based on SCG information.

    :param bac_mg_table: The path to the marker gene table for bacteria.
    :param ar_mg_table: The path to the marker gene table for archaea.
    :param res_path: The path to the result files.
    :param ignore_kmeans_res: Whether to ignore K-means results (default: False).

    :return: The best method based on estimated bin quality.
    """

    # bin_dirs = get_bin_dirs(bin_dirs_file)
    filenames = os.listdir(res_path)
    #cluster result file list
    namelist = []
    for filename in filenames:
        if filename.endswith('.tsv'):
            namelist.append(filename)

    namelist.sort()
    #key: cluster file name. value:cluster file path
    bin_dirs = {}
    for res in namelist:
        bin_dirs[res] =  (res_path + res)

    """
    bins: key1 method_id key2 cluster_id value
    contigs_in_bins: key1:contig name. key2:method_id value:cluster_id
    contig_list: seqid list
    Leiden_result_dict: key:method_id value
    """
    bins, contigs, contiglist, Leiden_result_dict = read_bins_nosequences(bin_dirs, contig_dict)

    methods_sorted = sorted(bins.keys())

    return  contiglist, Leiden_result_dict, bins, contigs, methods_sorted

def _compute_cluster_stats(args):
    c_method_id, c_cluster_id, c_contig_set, contig2marker, c_weight, c_weight_list = args
    if c_weight < 200000:
        return (c_method_id, c_cluster_id, None)
    marker_list = []
    for c_contig in c_contig_set:
        marker_list.extend(contig2marker[c_contig])
    len_marker_list = len(marker_list)
    len_set_marker_list = len(set(marker_list))
    if len_marker_list == 0:
        return (c_method_id, c_cluster_id, None)
    recall = len_set_marker_list / 107
    contamination = (len_marker_list - len_set_marker_list) / len_marker_list
    F1 = 2 * recall * (1 - contamination) / (recall + (1 - contamination))
    return (c_method_id, c_cluster_id, {
        "F1": F1,
        "cont": contamination,
        "comp": recall,
        "weight": c_weight,
        "weight_list": c_weight_list
    })

def recompute_all_clusters(all_methods, bins, contig2marker, contig_dict):
    tasks = []
    for method_id in all_methods:
        for cluster_id in bins[method_id]:
            c_contig_set = bins[method_id][cluster_id]
            c_weight = sum(len(contig_dict[contig]) for contig in c_contig_set)
            c_weight_list = [len(contig_dict[contig]) for contig in c_contig_set]
            tasks.append((method_id, cluster_id, c_contig_set, contig2marker, c_weight, c_weight_list))

    with Pool(processes=max(cpu_count() - 1, 1)) as pool:
        results = pool.map(_compute_cluster_stats, tasks)

    methodid_2_clusterlist_F1list_contalist = defaultdict(dict)
    for method_id, cluster_id, stats in results:
        if stats:
            methodid_2_clusterlist_F1list_contalist[method_id][cluster_id] = stats
    return methodid_2_clusterlist_F1list_contalist


def compute_quality_summary(methodid_2_clusterlist_F1list_contalist):
    bin_quality_dict = defaultdict(lambda: {})
    for method_id, clusters in methodid_2_clusterlist_F1list_contalist.items():
        num_5010 = num_7010 = num_9010 = 0
        num_505 = num_705 = num_905 = 0
        for cluster_id, info in clusters.items():
            comp = info["comp"]
            cont = info["cont"]
            if comp > 0.5 and cont < 0.1:
                num_5010 += 1
            if comp > 0.7 and cont < 0.1:
                num_7010 += 1
            if comp > 0.9 and cont < 0.1:
                num_9010 += 1
            if comp > 0.5 and cont < 0.05:
                num_505 += 1
            if comp > 0.7 and cont < 0.05:
                num_705 += 1
            if comp > 0.9 and cont < 0.05:
                num_905 += 1
        bin_quality_dict[method_id]['num_5010'] = num_5010
        bin_quality_dict[method_id]['num_7010'] = num_7010
        bin_quality_dict[method_id]['num_9010'] = num_9010
        bin_quality_dict[method_id]['num_505'] = num_505
        bin_quality_dict[method_id]['num_705'] = num_705
        bin_quality_dict[method_id]['num_905'] = num_905
        bin_quality_dict[method_id]['sum'] = num_5010 + num_7010 + num_9010 + num_505 + num_705 + num_905
        bin_quality_dict[method_id]['sum_cont5'] = num_505 + num_705 + num_905
    return bin_quality_dict

def run_get_final_result(logger, args, num_threads):
    """
    Run the final step to get the best clustering result based on estimated bin quality.

    :param seed_num: The seed number.
    :param num_threads: The number of threads to use (default: 40).
    :param res_name: The name of the result (default: None).
    :param ignore_kmeans_res: Whether to ignore K-means results (default: True).
    """
    #load fasta file
    contig_dict = {record.id: str(record.seq) for record in SeqIO.parse(args.contig_file, 'fasta')}
    ## load single_copy file
    contig2marker = defaultdict(list)
    with open(args.output_path + '/fasta.SEED', 'r') as f:
        lines = f.readlines()
        for line in lines:
            cur_id = line.rstrip().split('\t')
            marker_id = cur_id[0]
            seqids = cur_id[1:]
            for seqid in seqids:
                contig2marker[seqid].append(marker_id)

    """
    bins: key1 method_id key2 cluster_id value
    contigs_in_bins: key1:contig name. key2:method_id value:cluster_id
    contig_list: seqid list
    Leiden_result_dict: key:method_id value:
    """
    #Leiden_result_dict: key:method_id value:
    contig_list, Leiden_result_dict_new, bins, contigs_in_bins, all_methods = estimate_bins_quality_nobins(contig_dict,  args.output_path + '/cluster_res/')

    # logger.info('Final result:\t'+args.output_path + '/cluster_res/'+best_method)
    # filter_small_bins(logger, args.contig_file, args.output_path + '/cluster_res/'+best_method, args)
    ########################################
    ####generate ensemble result
    '''
    contig_list: contig name list
    contig_dict: key: contig name, value: dna seq
    Leiden_result_dict: key:cluster file (top 16), value: labels for contig_list
    '''
    namelist = contig_list[:]
    # Leiden_result_dict = {}
    # for item in top_10_methods:
    #     Leiden_result_dict[item] = Leiden_result_dict_new[item]
    ##test
    # print(contig_list)
    # print(len(contig_list))
    # for key,value in Leiden_result_dict.items():
    #     print(key)
    #     print(len(value))
    #     print(value)
    minfasta = 200000
    extracted = []

    ##########################
    methodid_2_clusterlist_F1list_contalist = recompute_all_clusters(all_methods, bins, contig2marker, contig_dict)
    bin_quality_dict = compute_quality_summary(methodid_2_clusterlist_F1list_contalist)

    #
    columns = ['num_905', 'num_9010', 'sum', 'sum_cont5', 'num_705', 'num_7010', 'num_505', 'num_5010']
    df_c = pd.DataFrame(columns=columns)
    index_methods = []
    data_to_add = []
    for c_method in bin_quality_dict.keys():
        df_c.loc[c_method] = {
            'num_905': bin_quality_dict[c_method]['num_905'],
            'num_9010': bin_quality_dict[c_method]['num_9010'],
            'sum': bin_quality_dict[c_method]['sum'],
            'sum_cont5': bin_quality_dict[c_method]['sum_cont5'],
            'num_705': bin_quality_dict[c_method]['num_705'],
            'num_7010': bin_quality_dict[c_method]['num_7010'],
            'num_505': bin_quality_dict[c_method]['num_505'],
            'num_5010': bin_quality_dict[c_method]['num_5010']
        }

    result_row = df_c.sort_values(by=['sum_cont5','num_905', 'num_9010',  'num_705', 'num_7010' \
        , 'num_505', 'num_5010'], ascending=False).iloc[0]
    best_method = result_row.name
    logger.info("\n" + df_c.to_string())
    logger.info('Final result:\t'+args.output_path + '/cluster_res/'+best_method)
    #filter_small_bins(logger, args.contig_file, args.output_path + '/cluster_res/' + best_method, args)

    ################################
    while sum(len(contig_dict[contig]) for contig in contig_list) >= minfasta:
        if len(contig_list) == 1:
            extracted.append(contig_list)
            break
        Top_methods = get_top_10_methods(bin_quality_dict)
        max_bin, max_method_id, max_cluster_id = get_best_bin(Top_methods, bins, methodid_2_clusterlist_F1list_contalist)
        if not max_bin:
            break
        extracted.append(max_bin)
        del methodid_2_clusterlist_F1list_contalist[max_method_id][max_cluster_id]
        del bins[max_method_id][max_cluster_id]
        """
        bins: key1 method_id key2 cluster_id value 
        contigs_in_bins: key1:contig name. key2:method_id value:cluster_id
        """
        #print('contig_list',contig_list)
        max_bin_set = set(max_bin)
        contig_list = [c for c in contig_list if c not in max_bin_set]

        for c_method_id in Top_methods:
            if c_method_id == max_method_id:
                continue
            #
            deled_cluster_list = set()
            #
            for temp in max_bin:
                # logger.info("contigs_in_bins:{}".format(contigs_in_bins[temp]))
                # logger.info("bins:{}".format(bins[c_method_id]))
                # logger.info(c_method_id)
                # logger.info(c_cluster_id)
                #c_cluster_id = contigs_in_bins[temp][c_method_id]
                c_cluster_id = contigs_in_bins.get(temp, {}).get(c_method_id)
                if not c_cluster_id:
                    continue
                deled_cluster_list.add(c_cluster_id)
                bins[c_method_id][c_cluster_id].discard(temp)

                # temp_index = contig_list.index(temp)
                # contig_list.pop(temp_index)
            #
            for deled_cluster in deled_cluster_list:
                if deled_cluster not in methodid_2_clusterlist_F1list_contalist[c_method_id]:
                    continue
                else:
                    deled_cluster_contigs = bins[c_method_id][deled_cluster]
                    if len(deled_cluster_contigs) == 0:
                        #
                        #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        #print(f" {deled_cluster}")
                        #print(f"method {c_method_id} 的: {bins[c_method_id].keys()}")
                        #print(f"method {c_method_id} cluster: {methodid_2_clusterlist_F1list_contalist[c_method_id].keys()}")
                        del methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]
                        del bins[c_method_id][deled_cluster]
                    else:
                        if methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]['weight'] < 200000 :
                            del methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]
                            #del bins[c_method_id][deled_cluster]
                            continue
                        marker_list = []
                        tem_weight = 0
                        tem_weight_list = []
                        for temp_contig in deled_cluster_contigs:
                            marker_list.extend(contig2marker[temp_contig])
                            cur_len = len(contig_dict[temp_contig])
                            tem_weight += cur_len
                            tem_weight_list.append(cur_len)
                        len_marker_list = len(marker_list)
                        len_set_marker_list = len(set(marker_list))
                        if len_marker_list == 0:
                            del methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]
                            #del bins[c_method_id][deled_cluster]
                            continue
                        elif tem_weight < 200000:
                            del methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]
                            #del bins[c_method_id][deled_cluster]
                            continue
                        else:
                            tem_recall = len_set_marker_list / 107
                            tem_contamination = (len_marker_list - len_set_marker_list) / len_marker_list
                            tem_F1 = 2 * tem_recall * (1 - tem_contamination) / (
                                    tem_recall + (1 - tem_contamination))


                        methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]['F1'] = tem_F1
                        methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]['cont'] = tem_contamination
                        methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]['comp'] = tem_recall
                        methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster]['weight'] = tem_weight
                        methodid_2_clusterlist_F1list_contalist[c_method_id][deled_cluster][
                            "weight_list"] = tem_weight_list

    for c_cluster_id in  bins[best_method]:
        extracted.append(bins[best_method][c_cluster_id])

    contig2ix = {}
    for i, cs in enumerate(extracted):
        for c in cs:
            contig2ix[c] = i
    contig_labels = [contig2ix.get(c, -1) for c in namelist]
    write_bins(namelist, contig_labels,
                    args.output_path + '/SCGBINNER_result/', args.output_path + '/SCGBINNER_result.tsv',
                    contig_dict,
                    minfasta=minfasta,
                    output_tag=None)



def write_bins(namelist, contig_labels, output, output_file, contig_seqs,
               minfasta = 200000, output_tag=None):
    '''
    Write binned FASTA files

    Returns: DataFrame with information on the bins
    '''
    from collections import defaultdict
    import pandas as pd

    res = defaultdict(list)
    for label, name in zip(contig_labels, namelist):
        if label != -1:
            res[label].append(name)

    os.makedirs(output, exist_ok=True)
    re_str = ""
    written = []
    for label, contigs in res.items():
        sizes = [len(contig_seqs[contig]) for contig in contigs]
        whole_bin_bp = sum(sizes)

        if whole_bin_bp >= minfasta:
            if output_tag is None:
                ofname = f'bin.{label}.fa'
            else:
                ofname = f'{output_tag}_{label}.fa'
            ofname = os.path.join(output, ofname)
            with open(ofname, 'w') as ofile:
                for contig in contigs:
                    re_str = re_str + f"{contig}" + "\t" + f"{label}" + "\n"
                    ofile.write(f'>{contig}\n{contig_seqs[contig]}\n')
    with open(output_file, "w" ) as f:
        f.write(re_str[:-1])


def compute_n50(lengths):
    if not lengths:
        return 0
    lengths.sort(reverse=True)
    total = sum(lengths)
    cumsum = 0
    for l in lengths:
        cumsum += l
        if cumsum >= total / 2:
            return l

def get_best_bin(Top_methods, bins, methodid_2_clusterlist_F1list_contalist):

    # There is room for improving the loop below to avoid repeated computation
    # but it runs very fast in any case
    for max_contamination  in [0.05]:
        for max_comp in [0.9, 0.7, 0.5]:
            max_F1 = 0
            weight_of_max = 1e9
            max_bin = None
            max_method_id = None
            max_cluster_id = None
            for c_method_id in Top_methods:
                for c_cluster_id, info in methodid_2_clusterlist_F1list_contalist[c_method_id].items():
                    c_F1 = info["F1"]
                    c_cont = info["cont"]
                    c_comp = info["comp"]
                    c_weight = info["weight"]
                    c_weight_list = info["weight_list"]
                    if c_weight < 200000:
                        continue
                    if c_cont <= max_contamination and c_comp >= max_comp:
                        if c_F1 > max_F1:
                            max_F1 = c_F1
                            weight_of_max = c_weight
                            max_bin = bins[c_method_id][c_cluster_id]
                            max_N50 = compute_n50(c_weight_list)
                            max_method_id = c_method_id
                            max_cluster_id = c_cluster_id
                        elif c_F1 == max_F1:
                            c_N5 = compute_n50(c_weight_list)
                            if c_N5 > max_N50:
                                weight_of_max = c_weight
                                max_bin = bins[c_method_id][c_cluster_id]
                                max_method_id = c_method_id
                                max_cluster_id = c_cluster_id
        if max_F1 > 0:  # if there is a bin with F1 > 0
            return max_bin.copy(), max_method_id, max_cluster_id
    return None, None, None

