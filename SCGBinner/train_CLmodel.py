import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import sys

from get_augfeature import get_ContrastiveLearningDataset
from utils import get_length

from models.mlp import EmbeddingNet
from simclr import SimCLR

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

class SingleCopyDataset(Dataset):
    def __init__(self, data_list, batch_indices):
        self.data_list = data_list
        self.batch_indices = batch_indices

    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index + 1]
        batch_data = [data[start_idx:end_idx] for data in self.data_list]
        return batch_data

    def __len__(self):
        return len(self.batch_indices) - 1


def train_CLmodel(logger, args):
    """
    Train the Contrastive Learning model.
    """
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.device_number))
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    logger.info("Generate features for the data.")

    if os.path.exists(args.output_path+'/embeddings.tsv'):
        logger.info("The embeddings file has been generated before, please check the output commands.")
        sys.exit()

    #dataset:(n_views, n_contigs, n_features)
    dataset, namelist = get_ContrastiveLearningDataset(args.data, args.n_views,
                                                       args.kmer_model_path, args.device, args.nokmer, args.cov_meannormalize,
                                                       args.cov_minmaxnormalize, args.cov_standardization,args.addvars,args.vars_sqrt, args.kmer_l2_normalize, args.kmerMetric_notl2normalize)

    ##########################################################################
    #remove part of SCP
    Qn = args.remove_single_copy
    #build SCP dataloader
    cur_mapObj = dict(zip(namelist, range(len(namelist))))
    single_copy_seqid_list = []
    single_copy_seqid_len_list = []
    flagn = 0
    with open(args.output_path + '/fasta.SEED', 'r') as f:
        lines = f.readlines()
        for line in lines:
            seqids = line.rstrip().split('\t')[1:]
            filtered_seqids = [seqid for seqid in seqids if seqid in cur_mapObj]
            seqids_len = len(filtered_seqids)
            if seqids_len >= Qn:
                flagn+=1
                if not args.scg_batch_size:
                    single_copy_seqid_list.extend(filtered_seqids)
                    single_copy_seqid_len_list.append(seqids_len)
                else:
                    single_copy_seqid_list.extend(filtered_seqids)
                    import math
                    n_part = math.ceil(seqids_len / args.scg_batch_size)
                    chunk_size = math.ceil(seqids_len / n_part)
                    cur_len_list = [chunk_size] * (n_part - 1)
                    cur_len_list.append(seqids_len - chunk_size * (n_part - 1))
                    single_copy_seqid_len_list.extend(cur_len_list)

    logger.info(f"Single copy number: {flagn}")

    for i in range(len(single_copy_seqid_len_list)):
        if i > 0:
            single_copy_seqid_len_list[i] = single_copy_seqid_len_list[i] + single_copy_seqid_len_list[i-1]
    batch_indices = [0] + single_copy_seqid_len_list
    #
    #cur_mapObj = dict(zip(namelist, range(len(namelist))))
    s_IdxArr = np.empty(len(single_copy_seqid_list), dtype=int)
    for i, seq in enumerate(single_copy_seqid_list):
        s_IdxArr[i] = cur_mapObj[seq]
    s_dataset = [cur_mat[s_IdxArr] for cur_mat in dataset]
    print(s_dataset[0].shape)

    #s_Data = [[x] for x in single_copy_seqid_list]
    single_copy_dataset = SingleCopyDataset(s_dataset ,batch_indices)
    s_loader = torch.utils.data.DataLoader(
        single_copy_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    #####################################################################################

    contig_file = args.data + '/aug0/sequences_aug0.fasta'
    lengths = get_length(contig_file)
    length_weight = []
    for seq_id in namelist:
        length_weight.append(lengths[seq_id])


    if args.notuseaug0:
        args.n_views = args.n_views - 1
        train_dataset = torch.utils.data.TensorDataset(*[dataset[i+1][np.array(length_weight) >= args.contig_len]
                                                         for i in range(args.n_views)])
    else:
        # train_dataset = torch.utils.data.TensorDataset(*dataset)
        #
        train_dataset = torch.utils.data.TensorDataset(*[dataset[i][np.array(length_weight) >= args.contig_len]
                                                         for i in range(args.n_views)])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Set embedder model.
    if args.kmer_model_path == 'empty':
        cov_dim = len(dataset[0][0]) - 136
        input_size = args.out_dim_forcov + 136
    else:
        cov_dim = len(dataset[0][0]) - 128
        input_size = args.out_dim_forcov + 128
        print('cov_dim:\t' + str(cov_dim) + '\n')


    emb_szs_list = [args.emb_szs_forcov] * args.n_layer_forcov

    cov_model = EmbeddingNet(
        in_sz=cov_dim,
        out_sz=args.out_dim_forcov,
        emb_szs=emb_szs_list,
        ps=[args.dropout_value] * (args.n_layer_forcov - 1),
        use_bn=True,
        actn=nn.LeakyReLU(),
    )

    from models.mlp2 import EmbeddingNet as EmbeddingNet2


    emb_szs_list = [args.emb_szs] * args.n_layer

    model = EmbeddingNet2(
        in_sz=input_size,
        out_sz=args.out_dim,
        emb_szs=emb_szs_list,
        ps=[args.dropout_value] * (args.n_layer - 1),
        use_bn=True,
        actn=nn.LeakyReLU(),
        cov_model=cov_model,
        covmodel_notl2normalize=args.covmodel_notl2normalize,
    )
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train_addpretrain(train_loader, s_loader, dataset, namelist)

    logger.info("Finish training.")
