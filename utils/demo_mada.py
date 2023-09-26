# -*- coding: utf-8 -*-
# @Time    : 2023
# @Author  : zwwang
# @File    : demo_mada.py
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression


def estimate_sim_mada(Xs_all, ys_all, Xt):
    # Xs_all: [N_1*d_fea, N_2*d_fea, ..., N_m*d_fea]
    # print(Xs_all.shape, ys_all.shape, Xt.shape)
    Xs_all = Xs_all.tolist()
    ys_all = ys_all.tolist()
    Xt = Xt.tolist()
    # print(len(Xs_all), len(ys_all), len(Xt))
    num_Ds = len(ys_all)
    nC = len(np.unique(ys_all))  # class_num
    ys_all = [ys_all[i] + nC * i for i in range(num_Ds)]
    Xs_new = np.array(Xs_all)
    ys_new = np.array(ys_all)
    ys_new = ys_new.flatten()
    # print(Xs_new.shape, ys_new.shape)

    # mdl = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    mdl = LogisticRegression(C=200, random_state=2020)
    mdl.fit(Xs_new, ys_new)
    y_pred = mdl.predict(Xt)

    # I: 预测目标域每个样本属于哪个域，编号从1~n_Ds
    nt = Xt.shape[0]
    d_pred = [math.floor(y / nC) for y in y_pred]
    corr_pred = np.array([d_pred.count(i) / nt for i in range(num_Ds)])

    # II: 另一种思路，看目标域m个样本在各类上预测概率的之和，然后算平均，是各个域的概率
    # y_pred = lr.predict_proba(Xt)
    # prob_pred = [np.mean(y_pred[:, 2 * i:2 * i + 1]) for i in range(num_Ds)]
    # corr_pred = np.array(prob_pred) / sum(prob_pred)

    return corr_pred


