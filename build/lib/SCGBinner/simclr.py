# modified from https://github.com/sthalles/SimCLR
import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_config_file, accuracy, save_checkpoint
import pandas as pd
import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        """
        Initialize the SimCLR model and related components.

        :param kwargs: Keyword arguments including 'args', 'model', 'optimizer', 'scheduler'.
        """
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.output_path)
        logging.basicConfig(filename=os.path.join(self.args.output_path, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        """
        Calculate the InfoNCE loss for SimCLR.

        :param features: Input features.
        :return: Logits and labels for the loss.
        """
        #contrastive labels in a batch (n_view*batch, n_view*batch)
        cur_batch = int(features.shape[0] / self.args.n_views)
        labels = torch.cat([torch.arange(cur_batch) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        #(n*batch, 128)
        features = F.normalize(features, dim=1)
        # (n, batch, 128)
        features_reshaped = features.view(self.args.n_views, cur_batch, features.shape[1])
        # (n, batch, 128)
        features_sum = (features_reshaped.sum(dim=0, keepdim=True) - features_reshaped) / (self.args.n_views - 1)

        #(n*batch, 128)
        features_sum = features_sum.view(self.args.n_views * cur_batch, features.shape[1])

        similarity_matrix = torch.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        # (n_view*batch, n_view*batch-1)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives (n*batch, 1)
        positives = torch.sum(features * features_sum, dim=1, keepdim=True)
        # select only the negatives the negatives (n_view*batch, n_view*batch-n_view)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def divide_equally(self, total_batches, parts):
        quotient, remainder = divmod(total_batches, parts)
        return [quotient + 1] * remainder + [quotient] * (parts - remainder)

    def train_addpretrain(self, train_loader,s_loader, data, namelist):
        """
        Train the SimCLR model with an additional pre-trained k-mer model.

        :param train_loader: Data loader for training.
        :param data: Input data.
        :param namelist: List of sequence names.
        """
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.output_path, self.args)

        earlystop_epoch=0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        # logging.info(f"Training with cpu: {self.args.disable_cuda}.")
        if self.args.kmer_model_path == 'empty':
            kmer_len = 136
        else:
            kmer_len = 128
        logging.info('kmer_len:\t' + str(kmer_len) + '\n')
        #split single copy
        total_batches = len(s_loader)


        logging.info(f"total_SCG_batches: {total_batches}")
        #
        for epoch_counter in range(self.args.epochs):
            epoch_loss_sum = 0
            epoch_top1_sum = 0
            num_batches = 0

            # modified
            for contig_features in tqdm(train_loader):
                contig_features = torch.cat(contig_features, dim=0)

                contig_features = contig_features.to(self.args.device)
                # print(contig_features.shape)

                with autocast(enabled=self.args.fp16_precision):
                    features, covemb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                    # print(contig_features[:2, :-kmer_len])
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                epoch_loss_sum += loss.item()
                epoch_top1_sum += top1[0].item()
                num_batches += 1

            #single_copy_constraint
            if total_batches != 0:
                for i, contig_features in enumerate(tqdm(s_loader)):
                    #print(contig_features[0].shape)
                    #print(contig_features)
                    contig_features = torch.cat(contig_features, dim=0)
                    contig_features = contig_features.view(-1, contig_features.shape[-1])
                    contig_features = contig_features.to(self.args.device)
                    with autocast(enabled=self.args.fp16_precision):
                        # print(contig_features.shape)
                        # print(contig_features[:, -kmer_len:].shape)
                        # print(contig_features[:, :-kmer_len].shape)
                        features, covemb = self.model(contig_features[:, -kmer_len:], contig_features[:, :-kmer_len])
                        # print(contig_features[:2, :-kmer_len])
                        logits_single, labels_single = self.info_nce_loss(features)
                        loss_single_copy = self.criterion(logits_single, labels_single)
                    self.optimizer.zero_grad()
                    scaler.scale(loss_single_copy).backward()
                    scaler.step(self.optimizer)
                    scaler.update()



            if not self.args.notuse_scheduler:
                # warmup for the first 10 epochs
                if epoch_counter >= 10:
                    self.scheduler.step()
            avg_loss = epoch_loss_sum / num_batches
            avg_top1 = epoch_top1_sum / num_batches
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss:.4f}\tTop1 accuracy: {avg_top1:.2f}")

            if self.args.earlystop:
                if epoch_counter >= 10 and avg_top1 > 99.0:
                    earlystop_epoch +=1
                else:
                    earlystop_epoch = 0

                if earlystop_epoch >=3:
                    break

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            # 'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.args.output_path, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.args.output_path}.")

        # ckpt = torch.load('/checkpoint_0200.pth.tar')

        with torch.no_grad():
            self.model.eval()
            bs_ = 1024

            print(len(data))
            all_data = data[0].to(self.args.device)
            out = np.concatenate([self.model(all_data[i:i + bs_, -kmer_len:],
                                             all_data[i:i + bs_, :-kmer_len])[0].to('cpu').numpy()
                                  for i in range(0, len(data[0]), bs_)], axis=0)
            embeddings_df = pd.DataFrame(out, index=namelist)

            outfile = self.args.output_path + '/embeddings.tsv'
            embeddings_df.to_csv(outfile, sep='\t', header=True)


            # covout = np.concatenate([self.model(data[0][i:i + bs_, -kmer_len:].to(self.args.device),
            #                                     data[0][i:i + bs_, :-kmer_len].to(self.args.device))[1].to('cpu').numpy()
            #                          for i in range(0, len(data[0]), bs_)], axis=0)
            # embeddings_df = pd.DataFrame(covout, index=namelist)
            # outfile = self.args.output_path + '/covembeddings.tsv'
            # embeddings_df.to_csv(outfile, sep='\t', header=True)

