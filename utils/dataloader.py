# -*- coding: utf-8 -*-
# @Time    : 2023
# @Author  : zwwang
# @File    : dataloader.py
import torch as tr
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.autograd import Variable
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, TomekLinks
from utils.data_augment import data_aug
from utils.demo_mada import estimate_sim_mada
from scipy.io import loadmat, savemat
from os import walk


def read_mi_all(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        src_data = np.squeeze(Data_raw[s, :, :, :])
        src_label = Label[s, :].reshape(-1, 1)

        if args.aug:
            sample_size = src_data.shape[2]
            # mult_flag, noise_flag, neg_flag, freq_mod_flag
            flag_aug = [True, True, True, True]
            src_data = np.transpose(src_data, (0, 2, 1))
            src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
            src_data = np.transpose(src_data, (0, 2, 1))

        covar = Covariances(estimator=args.cov_type).transform(src_data)
        fea_tsm = TangentSpace().fit_transform(covar)
        src_label = src_label.reshape(-1, 1)

        data.append(fea_tsm)
        label.append(src_label)

    return data, label


def read_mi_train(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    src_data = np.squeeze(Data_raw[args.ids, :, :, :])
    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (288, 22, 750)

    if args.aug:
        sample_size = src_data.shape[2]
        # mult_flag, noise_flag, neg_flag, freq_mod_flag
        flag_aug = [True, True, True, True]
        # flag_aug = [True, False, False, False]
        src_data = np.transpose(src_data, (0, 2, 1))
        src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
        src_data = np.transpose(src_data, (0, 2, 1))
        src_label = tr.from_numpy(src_label).long()
    # print(src_data.shape, src_label.shape)  # (288*7, 22, 750)

    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, src_label.shape)

    return fea_tsm, src_label


def read_mi_test(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()

    # 288 * 22 * 750
    covar_src = Covariances(estimator=args.cov_type).transform(tar_data)
    fea_tsm = TangentSpace().fit_transform(covar_src)

    # covar = Covariances(estimator=cov_type).transform(tar_data)
    # tmp_ref = TangentSpace().fit(covar[:ntu, :, :])
    # fea_tsm = tmp_ref.transform(covar)

    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, tar_label.shape)
    return fea_tsm, tar_label


def read_mi_test_aug(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data)
    X_tar = TangentSpace().fit_transform(covar_tar)
    X_tar = Variable(tr.from_numpy(X_tar).float())
    y_tar = tr.from_numpy(tar_label).long()

    sample_size = tar_data.shape[2]
    flag_aug = [True, True, True, True]
    tar_data_tmp = np.transpose(tar_data, (0, 2, 1))
    tar_data_tmp, tar_label_aug = data_aug(tar_data_tmp, tar_label, sample_size, flag_aug)
    tar_data_aug = np.transpose(tar_data_tmp, (0, 2, 1))

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data_aug)
    X_tar_aug = TangentSpace().fit_transform(covar_tar)
    X_tar_aug = Variable(tr.from_numpy(X_tar_aug).float())
    y_tar_aug = tr.from_numpy(tar_label_aug).long()

    # X.shape - (#samples, # feas)
    print(y_tar.shape, y_tar.shape)
    print(X_tar_aug.shape, y_tar_aug.shape)
    return X_tar, y_tar, X_tar_aug, y_tar_aug


def read_mi_combine(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    # print('raw data shape', Data_raw.shape, Label.shape)

    Data_new = Data_raw.copy()
    n_sub = len(Data_raw)

    # MTS transfer
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_new[ids[i]]))
        src_label.append(np.squeeze(Label[ids[i]]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    # final label
    src_label = np.squeeze(src_label)
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # final features
    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    src_data = Variable(tr.from_numpy(fea_tsm).float())

    return src_data, src_label


def read_mi_combine_tar(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    print('raw data shape', Data_raw.shape, Label.shape)

    Data_new = Data_raw.copy()
    n_sub = len(Data_raw)

    # combine multiple source data
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_new[ids[i]]))
        src_label.append(np.squeeze(Label[ids[i]]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    # final label
    src_label = np.squeeze(src_label)
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (n_src, chns, fts)
    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)  # tangent space transform
    src_data = Variable(tr.from_numpy(fea_tsm).float())  # (n_src, low_dim_fts)

    # single target domain data
    tar_data = np.squeeze(Data_new[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)
    covar = Covariances(estimator=args.cov_type).transform(tar_data)
    tmp_ref = TangentSpace().fit(covar)  # tangent space transform
    fea_tsm = tmp_ref.transform(covar)
    tar_data = Variable(tr.from_numpy(fea_tsm).float())
    tar_label = tr.from_numpy(tar_label).long()
    print(src_data.shape, src_label.shape)  # (n_src, low_dim_fts)
    print(tar_data.shape, tar_label.shape)  # (n_tar, low_dim_fts)

    return src_data, src_label, tar_data, tar_label


def read_ch_combine_tar(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/'), (None, None, []))[2]
        i = args.idt
        src_x = []
        src_y = []
        for j in range(len(domains)):
            if i != j:
                src = loadmat('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/' + domains[j])
                src0, src1 = src['data'], src['label']
                src1 = src1.squeeze()
                src0 = data_normalize(src0, args.norm)  # 分开zscore
                src_x.append(src0)
                src_y.append(src1)
        src_data = np.concatenate(src_x, axis=0)
        src_label = np.concatenate(src_y, axis=0)
        if args.smote:
            # 先combine再smote
            oversample = SMOTE(random_state=42, sampling_strategy=0.8)
            src_data, src_label = oversample.fit_resample(src_data, src_label)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持99%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/' + domains[i])
        tar_data, tar_label = tar['data'], tar['label']
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()

    src_data = Variable(tr.from_numpy(src_data).float())
    src_l = tr.from_numpy(src_label).long()
    tar_d = Variable(tr.from_numpy(tar_data).float())
    tar_l = tr.from_numpy(tar_label).long()

    return src_data, src_l, tar_d, tar_l


def read_ch_combine_srcsel(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/CHSZ/CHSZ_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        idt = args.idt
        for i in range(tar_num):
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:args.sn]  # 选择20个源域
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    src0 = data_normalize(src0, args.norm)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)
        # src_data = data_normalize(src_data, args.norm)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/CHSZ/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()

    if args.smote:
        # 先combine再smote
        oversample = SMOTE(random_state=42, sampling_strategy=0.3)
        src_data, src_label = oversample.fit_resample(src_data, src_label)

    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()

    return src_data, src_label, tar_data, tar_label


def read_seizure_combine_tar(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        i = args.idt
        src_x = []
        src_y = []
        for j in range(len(domains)):
            if i != j:
                src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[j])
                src0, src1 = src['data'], src['label']
                src1 = src1.squeeze()
                src_x.append(src0)
                src_y.append(src1)
        src_data = np.concatenate(src_x, axis=0)
        src_label = np.concatenate(src_y, axis=0)
        # 源域标准化
        src_data = data_normalize(src_data, args.norm)
        if args.smote:
            oversample = SMOTE(random_state=42)
            src_data, src_label = oversample.fit_resample(src_data, src_label)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[i])
        tar_data, tar_label = tar['data'], tar['label']
        # 目标域标准化
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()

    src_data = Variable(tr.from_numpy(src_data).float())
    src_l = tr.from_numpy(src_label).long()
    tar_d = Variable(tr.from_numpy(tar_data).float())
    tar_l = tr.from_numpy(tar_label).long()

    return src_data, src_l, tar_d, tar_l


def read_seizure_combine_srcsel(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        idt = args.idt
        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        for i in range(tar_num):
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:args.sn]  # 选择源域数量 n/2 or 25
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    src0 = data_normalize(src0, args.norm)
                    if args.smote:
                        # 先smote再combine
                        oversample = SMOTE(random_state=42)
                        src0, src1 = oversample.fit_resample(src0, src1)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)

        # src_data = data_normalize(src_data, args.norm)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()

    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()

    return src_data, src_label, tar_data, tar_label


def read_ch_combine_sbmada(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('./data/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('./data/CHSZ/CHSZ_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        # input trade off
        corre_pd = pd.read_excel('./data/CHSZ_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
                # print('this is the ', trade_off)
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:args.sn]  # 选择源域数量[n/2]
                sb = loadmat('./data/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('./data/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    src0 = data_normalize(src0, args.norm)  # 分开zscore
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)
        if args.smote:
            # 先combine再smote
            oversample = SMOTE(random_state=42, sampling_strategy=0.8)
            src_data, src_label = oversample.fit_resample(src_data, src_label)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        sb_data = data_normalize(sb_data, args.norm)
        sb_data = pca.transform(sb_data)

        tar = loadmat('./data/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def read_nicu_seperated_sbmada(args):  # no data augment

    lens = []
    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        i = args.idt
        src_x = []
        src_y = []
        for j in range(len(domains)):
            if i != j:
                src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[j])
                src0, src1 = src['data'], src['label']
                src1 = src1.squeeze()
                src_x.append(src0)
                src_y.append(src1)
                lens.append(len(src1))
        src_data = np.concatenate(src_x, axis=0)
        src_label = np.concatenate(src_y, axis=0)
        # 源域标准化
        src_data = data_normalize(src_data, args.norm)
        if args.smote:
            oversample = SMOTE(random_state=42)
            src_data, src_label = oversample.fit_resample(src_data, src_label)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[i])
        tar_data, tar_label = tar['data'], tar['label']
        # 目标域标准化
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
        tar_label = tar_label.squeeze()

    src_data = Variable(tr.from_numpy(src_data).float())
    src_l = tr.from_numpy(src_label).long()
    tar_d = Variable(tr.from_numpy(tar_data).float())
    tar_l = tr.from_numpy(tar_label).long()

    return src_data, src_l, tar_d, tar_l, lens

def read_seizure_combine_srcbest(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        # input trade off
        corre_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:26]  # 选择源域数量[all]
                sb = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    src0 = data_normalize(src0, args.norm)
                    if args.smote:
                        # 先smote再combine
                        oversample = SMOTE(random_state=42)
                        src0, src1 = oversample.fit_resample(src0, src1)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)

        # src_data = data_normalize(src_data, args.norm)
        pca = PCA(n_components=0.95)  # 保证降维后的数据保持95%的信息
        pca.fit(src_data)  # 使用源域数据fit pca
        src_data = pca.transform(src_data)

        sb_data = data_normalize(sb_data, args.norm)
        sb_data = pca.transform(sb_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)

    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def read_seizure_combine_sbmada(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]

        # input trade off
        corre_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:args.sn]  # 选择源域数量[n/2]、25
                sb = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)
        # 数据标准化
        src_data = data_normalize(src_data, args.norm)
        # 数据平衡
        if args.smote:
            oversample = SMOTE(random_state=42)
            src_data, src_label = oversample.fit_resample(src_data, src_label)
        # 特征降维
        pca = PCA(n_components=0.95)
        pca.fit(src_data)
        src_data = pca.transform(src_data)

        # 数据标准化+数据平衡+特征降维
        sb_data = data_normalize(sb_data, args.norm)
        if args.smote:
            oversample = SMOTE(random_state=42)
            sb_data, sb_label = oversample.fit_resample(sb_data, sb_label)
        sb_data = pca.transform(sb_data)

        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()

        # 数据标准化+特征降维
        tar_data = data_normalize(tar_data, args.norm)
        tar_data = pca.transform(tar_data)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def read_seizure_combine_sbmada_singal_zscore(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        # input trade off
        corre_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
                # print('this is the ', trade_off)
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:args.sn]  # 选择源域数量[n/2]、25
                sb = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                if args.smote:
                    # 先smote再combine
                    oversample = SMOTE(random_state=42)
                    sb_data, sb_label = oversample.fit_resample(sb_data, sb_label)  # TODO
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    if args.smote:
                        # 先smote再combine
                        oversample = SMOTE(random_state=42)
                        src0, src1 = oversample.fit_resample(src0, src1)
                    # 每个源域单独zscore
                    src0 = data_normalize(src0, args.norm)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)

    # src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    # single target domain data
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()

    # single src_best domain data
    sb_data = data_normalize(sb_data, args.norm)
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def read_seizure_combine_srcbest_M(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        # input trade off
        corre_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
                # print('this is the ', trade_off)
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:]  # 选择源域数量[all]
                sb = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    # 先smote再combine
                    oversample = SMOTE(random_state=42)
                    src0, src1 = oversample.fit_resample(src0, src1)
                    # 每个源域单独zscore
                    # src0 = data_normalize(src0, args.norm)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)

    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    # single target domain data
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()

    # single src_best domain data
    sb_data = data_normalize(sb_data, args.norm)
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def read_seizure_combine_sbmada_M(args):  # no data augment

    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        src_sel_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_DSSS_pca95.xlsx',
                                   index_col=0)
        src_sel_numpy = src_sel_pd.to_numpy()
        tar_num = src_sel_numpy[:, 0].shape[0]
        # input trade off
        corre_pd = pd.read_excel('/mnt/data2/zwwang/TASA_code/Data/NICU/NICU_tradeoff_pca95.xlsx', index_col=0)
        corre_numpy = corre_pd.to_numpy()
        idt = args.idt
        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[idt])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        for i in range(tar_num):
            if str(int(corre_numpy[i][0])) == domains[idt][1:-4]:
                trade_off = corre_numpy[i][2]
                # print('this is the ', trade_off)
            if str(src_sel_numpy[:, 0][i]) == domains[idt][1:-4]:
                d = src_sel_numpy[i][1:20]  # 选择源域数量[n/2]、25
                sb = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[0]) + '.mat')
                sb_data, sb_label = sb['data'], sb['label']
                sb_label = sb_label.squeeze()
                src_num = d.shape[0]
                src_x = []
                src_y = []
                for j in range(src_num):
                    print('this is the ' + str(j + 1) + ' src: ', str(d[j]))
                    src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + 'S' + str(d[j]) + '.mat')
                    src0, src1 = src['data'], src['label']
                    src1 = src1.squeeze()
                    # 先smote再combine
                    oversample = SMOTE(random_state=42)
                    src0, src1 = oversample.fit_resample(src0, src1)
                    # 每个源域单独zscore
                    # src0 = data_normalize(src0, args.norm)
                    src_x.append(src0)
                    src_y.append(src1)
                src_data = np.concatenate(src_x, axis=0)
                src_label = np.concatenate(src_y, axis=0)

    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    # single target domain data
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()

    # single src_best domain data
    sb_data = data_normalize(sb_data, args.norm)
    sb_data = Variable(tr.from_numpy(sb_data).float())
    sb_label = tr.from_numpy(sb_label).long()
    return src_data, src_label, tar_data, tar_label, sb_data, sb_label, trade_off


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)
    if norm_type == 'min_max':
        min_max_scaler = preprocessing.MinMaxScaler()
        fea_de = min_max_scaler.fit_transform(fea_de)

    return fea_de


def read_seizure_train(args):  # no data augment
    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        i = args.ids
        # load target domain
        src = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[i])
        src_data, src_label = src['data'], src['label']
        src_label = src_label.squeeze()

    # single target domain data
    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    # print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)

    return src_data, src_label


def read_seizure_test(args):  # no data augment
    if args.data_env == 'local':
        file = '../data/fts_labels/' + args.data + '.npz'
    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        i = args.idt
        # load target domain
        tar = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[i])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        # print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)

    # single target domain data
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()
    # print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)

    return tar_data, tar_label


def read_seizure_all(args):
    if args.data_env == 'gpu':
        domains = next(walk('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/'), (None, None, []))[2]
        data, label = [], []
        for j in range(len(domains)):
            subj = loadmat('/mnt/data2/zwwang/TASA_code/Data/NICU/fts_labels/' + domains[j])
            all_data, all_label = subj['data'], subj['label']
            data.append(all_data)
            label.append(all_label)

    return data, label


def read_seed_all(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        fea_de = np.squeeze(Data_raw[s, :, :])
        src_label = Label[s, :].reshape(-1, 1)
        data.append(fea_de)
        label.append(src_label)

    return data, label


def read_seed_train(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    fea_de = np.squeeze(Data_raw[args.ids, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_test(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    fea_de = np.squeeze(Data_raw[args.idt, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(fea_de.shape, tar_label.shape)

    return fea_de, tar_label


def read_seed_combine(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    print(Data_raw.shape, Label.shape)

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))

    fea_de = np.concatenate(src_data, axis=0)
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.concatenate(src_label, axis=0)
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_combine_tar(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :])
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(tar_data.shape, tar_label.shape)

    return src_data, src_label, tar_data, tar_label


def obtain_train_val_source(y_array, trial_ins_num, val_type):
    y_array = y_array.numpy()
    ins_num_all = len(y_array)
    src_idx = range(ins_num_all)

    if val_type == 'random':
        # 随机打乱会导致结果偏高，不管是MI还是SEED数据集
        num_train = int(0.9 * len(src_idx))
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    if val_type == 'last':
        # 按顺序划分，一般情况来说没问题，但是如果源数据类别是按顺序排的，会有问题
        num_train = int(0.9 * trial_ins_num)
        id_train = np.array(src_idx).reshape(-1, trial_ins_num)[:, :num_train].reshape(1, -1).flatten()
        id_val = np.array(src_idx).reshape(-1, trial_ins_num)[:, num_train:].reshape(1, -1).flatten()

    return id_train, id_val
