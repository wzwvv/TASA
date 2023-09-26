# -*- coding: utf-8 -*-
# @Time    : 2023
# @Author  : zwwang
# @File    : maml.py
import sys
import time

import numpy as np
import argparse
import os
import random
import copy

import pandas as pd
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from os import walk
from scipy.io import loadmat

import learn2learn as l2l

from utils.dataloader import read_mi_combine_tar, read_seed_combine_tar, read_ch_combine_sbmada, read_ch_seperated_sbmada
from utils.utils import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_auc_f1_bca_comb, data_loader_multisource, cal_auc_f1_bca_comb_twomodel
from utils.loss import ClassConfusionLoss, CELabelSmooth_raw, lmmd

class EarlyStopping:
    def __init__(self, patience=10, path=None):
        self.patience = patience
        self.counter = 0
        self.val_min_loss = 1e2
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        print(self.counter)
        if val_loss >= self.val_min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping coun9jter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.val_min_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        netF, netC = model
        tr.save(netF, self.path.replace('.ckpt', 'netF.ckpt'))
        tr.save(netC, self.path.replace('.ckpt', 'netC.ckpt'))

def train_target(args):
    if args.data in ['SEED', 'SEED4']:
        X_src, y_src, X_tar, y_tar = read_seed_combine_tar(args)
    elif args.data in ['seizure'] and args.method == 'MAML':
        #X_src, y_src, X_tar, y_tar, X_sb, y_sb, trade_off = read_ch_combine_sbmada(args)
        X_src, y_src, X_tar, y_tar, X_sb, y_sb, trade_off, lens = read_ch_seperated_sbmada(args)
    else:
        X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    args.input_dim = X_src.shape[1]

    dset_loaders = data_loader_multisource(X_src, y_src, X_tar, y_tar, args, lens)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='xy')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='xy')

    source_domain_num = len(lens)
    batch_nums = []
    iter_sources = []

    for s in range(source_domain_num):
        curr_iter = iter(dset_loaders["source_tr" + str(s)])
        batch_num = len(curr_iter)
        batch_nums.append(batch_num)
        iter_sources.append(curr_iter)
    batch_cnts = np.zeros(source_domain_num, dtype=int)

    max_iter = int(args.max_epoch * np.sum(batch_nums) / 2)
    args.max_iter = max_iter
    patience = 10
    model_path_final = 'Feature_based/models/'
    model_path = model_path_final + f'{args.dataset}_{args.app}_id{args.tar_id}_seed{args.SEED}.ckpt'
    EarlyStop = EarlyStopping(patience=patience, path=model_path)

    netF.train()
    netC.train()

    meta_modelC = l2l.algorithms.MAML(netC, lr=args.lr)

    all_parameters = list(netF.parameters()) + list(meta_modelC.parameters())
    opt = tr.optim.Adam(all_parameters, lr=args.lr)

    auc_e_list = []
    f1_e_list = []
    bca_e_list = []
    auc_t_list = []
    f1_t_list = []
    bca_t_list = []
    sen_e_list = []
    spec_e_list = []
    sen_t_list = []
    spec_t_list = []

    for iter_num in range(max_iter):  # max_iter = max_epoch * num of double minimatches

        opt.zero_grad()

        # print(iter_num)
        source_domain_num = len(lens)
        batch_nums = []
        iter_sources = []

        for s in range(source_domain_num):
            curr_iter = iter(dset_loaders["source_tr" + str(s)])
            batch_num = len(curr_iter)
            batch_nums.append(batch_num)
            iter_sources.append(curr_iter)
        batch_cnts = np.zeros(source_domain_num, dtype=int)

        # print('batch_nums', np.sum(batch_nums))

        num_batches_optimized = 0
        iteration_error = 0
        # loop num of double minimatches for ONE step of update
        for i in range(int(np.sum(batch_nums) / 2)):

            not_found = 0
            while True:
                ind1 = random.randint(0, source_domain_num - 1)
                ind2 = random.randint(0, source_domain_num - 1)
                if not_found == 500:
                    #print('cannot find more required double minibatches')
                    break
                not_found += 1
                if batch_cnts[ind1] == batch_nums[ind1] or batch_cnts[ind2] == batch_nums[ind2]:
                    continue
                elif ind1 == ind2:
                    continue
                else:
                    break
            if not_found == 500:
                break

            batch_cnts[ind1] += 1
            batch_cnts[ind2] += 1

            try:
                (xi, yi), (xj, yj) = next(iter_sources[ind1]), next(iter_sources[ind2])
            except:
                print('no next batch')
                continue

            num_batches_optimized += 1

            xi, yi, xj, yj = xi.cuda().float(), yi.cuda(
            ).long(), xj.cuda().float(), yj.cuda().long()

            inner_netC = meta_modelC.clone()

            _, out = inner_netC(netF(xi))
            inner_loss = tr.nn.functional.cross_entropy(out, yi)

            inner_netC.adapt(inner_loss)
            inner_loss.backward()

            inner_netC = meta_modelC.clone()

            _, predictions = inner_netC(netF(xj))
            valid_error = tr.nn.functional.cross_entropy(predictions, yj)
            iteration_error += valid_error
        # print('num_batches_optimized', num_batches_optimized)
        iteration_error /= num_batches_optimized
        # print('Loss : {:.3f}'.format(iteration_error.item()))

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(1.0 / num_batches_optimized)
        opt.step()

        if iter_num % int(np.sum(batch_nums) / 2) == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            # early stopping
            acc_s_te, sen_s_te, spec_s_te, auc_s_te, f1_s_te, bca_s_te, clf_loss = cal_auc_f1_bca_comb_twomodel(dset_loaders["source_te"], (netF, netC))
            EarlyStop(clf_loss, (netF, netC))
            model_testF = tr.load(model_path.replace('.ckpt', 'netF.ckpt'))
            model_testC = tr.load(model_path.replace('.ckpt', 'netC.ckpt'))
            acc_t_te, sen_t_te, spec_t_te, auc_t_te, f1_t_te, bca_t_te, _ = cal_auc_f1_bca_comb_twomodel(dset_loaders["Target"], (model_testF, model_testC))
            log_str = 'Task: {}, Iter:{}/{}; Val_loss = {:.4f}; Val_auc = {:.2f}%; Val_acc = {:.2f}%; Test_acc = {:.2f}%; Test_sen = {:.2f}%; Test_spec = {:.2f}%; Test_auc = {:.2f}%; Test_f1 = {:.2f}%; Test_bca = {:.2f}%'.format(args.task_str, iter_num, max_iter, clf_loss, auc_s_te, acc_s_te, acc_t_te, sen_t_te, spec_t_te, auc_t_te, f1_t_te, bca_t_te)
            args.log.record(log_str)
            print(log_str)
            auc_t_list.append(auc_t_te)
            bca_t_list.append(bca_t_te)
            f1_t_list.append(f1_t_te)
            sen_t_list.append(sen_t_te)
            spec_t_list.append(spec_t_te)
            if EarlyStop.early_stop:
                break

    args.log.record('Sen_Eval: ' + str(sen_e_list))
    args.log.record('Spec_Eval: ' + str(spec_e_list))
    args.log.record('AUC_Eval: ' + str(auc_e_list))
    args.log.record('F1_Eval: ' + str(f1_e_list))
    args.log.record('BCA_Eval: ' + str(bca_e_list))
    args.log.record('Sen_Test: ' + str(sen_t_list))
    args.log.record('Spec_Test: ' + str(spec_t_list))
    args.log.record('AUC_Test: ' + str(auc_t_list))
    args.log.record('F1_Test: ' + str(f1_t_list))
    args.log.record('BCA_Test: ' + str(bca_t_list))
    return sen_t_te, spec_t_te, auc_t_te, f1_t_te, bca_t_te, acc_t_te



def get_n_target(target_id):
    domains = next(walk('./data/fts_labels/'), (None, None, []))[2]
    for i in range(len(domains)):
        tar = loadmat('./data/fts_labels/' + domains[target_id])
        tar_data = tar['data']
        tar_num = tar_data.shape[0]
    return tar_num


if __name__ == '__main__':
    seed_arr = [2020, 2021, 2022]
    for seed in seed_arr:
        df = pd.DataFrame(columns=['lamda1', 'lamda2', 'sen', 'spec', 'auc', 'f1', 'bca', 'acc', 'duration'])
        data_name_list = ['seizure']
        data_idx = 0
        data_name = data_name_list[data_idx]
        domain = next(walk('./data/fts_labels/'), (None, None, []))[2]
        n_subject = len(domain)

        sub_sen_all = np.zeros(n_subject)
        sub_spec_all = np.zeros(n_subject)
        sub_auc_all = np.zeros(n_subject)
        sub_f1_all = np.zeros(n_subject)
        sub_bca_all = np.zeros(n_subject)
        sub_acc_all = np.zeros(n_subject)
        duration_all = np.zeros(n_subject)
        for idt in range(n_subject):
            n_tar = get_n_target(idt)
            if data_name == 'seizure': N, chn, class_num, trial_num = n_subject, 18, 2, n_tar
            if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
            if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
            if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
            if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851

            args = argparse.Namespace(bottleneck=64, lr=0.001, lr_decay1=0.1, lr_decay2=1.0,
                                      epsilon=1e-05, layer='wn', cov_type='oas', trial=trial_num,
                                      N=N, chn=chn, class_num=class_num, smooth=0)

            args.rate = 0.9
            args.lamda1 = 0.1  # 0.1,1 is the best group on validation set (10%)
            args.lamda2 = 1
            args.tar_id = domain[idt][1:-4]
            args.sn = 26
            args.dataset = 'CHSZ'
            args.smote = True
            args.data = data_name
            args.app = 'sbmada'
            args.method = 'MAML'
            args.mldg_beta = 1
            args.feasel = 'pca95'
            args.backbone = 'Net_ln2'
            if args.data in ['SEED', 'SEED4', 'seizure']:
                    args.batch_size = 32
                else:
                    args.batch_size = 16
                args.max_epoch = 100  # 10
                args.norm = 'zscore'
                args.validation = 'random'
            else:
                args.batch_size = 8  # 8 对于DANN和CDAN合适的
                args.max_epoch = 5  # 10
                args.input_dim = int(args.chn * (args.chn + 1) / 2)
                args.validation = 'last'
            args.eval_epoch = args.max_epoch / 10

            os.environ["CUDA_VISIBLE_DEVICES"] = '5'
            args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
            args.SEED = seed
            fix_random_seed(args.SEED)
            tr.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args)

            args.result_dir = './results/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            args.idt = idt
            source_str = 'Except_S' + domain[idt][1:-4]
            target_str = 'S' + domain[idt][1:-4]
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            t1 = time.time()
            sub_sen_all[idt], sub_spec_all[idt], sub_auc_all[idt], sub_f1_all[idt], sub_bca_all[idt], sub_acc_all[idt] = train_target(args)
            duration_all[idt] = time.time() - t1
            print(f'Sub:{idt:2d}, [{duration_all[idt]:5.2f}], Acc: {sub_auc_all[idt]:.4f}')
        print('Sub sen: ', np.round(sub_sen_all, 3))
        print('Sub sepc: ', np.round(sub_spec_all, 3))
        print('Sub auc: ', np.round(sub_auc_all, 3))
        print('Sub f1: ', np.round(sub_f1_all, 3))
        print('Sub bca: ', np.round(sub_bca_all, 3))
        print('Sub acc: ', np.round(sub_acc_all, 3))
        print('Avg sen: ', np.round(np.mean(sub_sen_all), 3))
        print('Avg spec: ', np.round(np.mean(sub_spec_all), 3))
        print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))
        print('Avg f1: ', np.round(np.mean(sub_f1_all), 3))
        print('Avg bca: ', np.round(np.mean(sub_bca_all), 3))
        print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
        print('Avg duration: ', np.round(np.mean(duration_all), 3))

        df = df.append(
            {'lamda1': args.lamda1,
             'lamda2': args.lamda2,
             'sen': np.round(np.mean(sub_sen_all), 3),
             'spec': np.round(np.mean(sub_spec_all), 3),
             'auc': np.round(np.mean(sub_auc_all), 3),
             'f1': np.round(np.mean(sub_f1_all), 3),
             'bca': np.round(np.mean(sub_bca_all), 3),
             'acc': np.round(np.mean(sub_acc_all), 3),
             'duration': np.round(np.mean(duration_all), 3)}, ignore_index=True)
        df_name = '/results/' + str(args.max_epoch) + '_' + args.app + '_' + args.method + '_' + args.feasel + '_' + str(
            args.lamda1) + '_' + str(args.lamda2) + '_' + str(args.sn) + '_' + str(args.rate) + '_' + str(args.SEED) + '.xlsx'
        df.to_excel(df_name, sheet_name='sheet_info')

        sen_sub_str = str(np.round(sub_sen_all, 3).tolist())
        sen_mean_str = str(np.round(np.mean(sub_sen_all), 3).tolist())
        spec_sub_str = str(np.round(sub_spec_all, 3).tolist())
        spec_mean_str = str(np.round(np.mean(sub_spec_all), 3).tolist())
        auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
        auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())
        f1_sub_str = str(np.round(sub_f1_all, 3).tolist())
        f1_mean_str = str(np.round(np.mean(sub_f1_all), 3).tolist())
        bca_sub_str = str(np.round(sub_bca_all, 3).tolist())
        bca_mean_str = str(np.round(np.mean(sub_bca_all), 3).tolist())
        acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
        acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
        args.log.record("\n===================sen====================")
        args.log.record(sen_sub_str)
        args.log.record(sen_mean_str)
        args.log.record("\n===================spec====================")
        args.log.record(spec_sub_str)
        args.log.record(spec_mean_str)
        args.log.record("\n===================auc====================")
        args.log.record(auc_sub_str)
        args.log.record(auc_mean_str)
        args.log.record("\n===================f1====================")
        args.log.record(f1_sub_str)
        args.log.record(f1_mean_str)
        args.log.record("\n===================bca===================")
        args.log.record(bca_sub_str)
        args.log.record(bca_mean_str)
        args.log.record("\n===================acc====================")
        args.log.record(acc_sub_str)
        args.log.record(acc_mean_str)
