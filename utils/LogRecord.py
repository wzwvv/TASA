# -*- coding: utf-8 -*-
# @Time    : 2023
# @Author  : zwwang
# @File    : LogRecord.py

import torch as tr
import os.path as osp
from datetime import datetime
from datetime import timedelta, timezone

from utils.utils import create_folder


class LogRecord:
    def __init__(self, args):
        self.args = args
        self.result_dir = args.result_dir
        # self.data_env = 'gpu' if tr.cuda.get_device_name(0) != 'GeForce GTX 1660 Ti' else 'local'\
        self.data_env = 'local'
        self.data_name = args.data
        self.method = args.method
        self.app = args.app
        self.batch_size = args.batch_size
        self.epoch = args.max_epoch
        self.tar_id = args.tar_id
        self.seed = args.SEED
        self.lamda1 = args.lamda1
        self.lamda2 = args.lamda2

    def log_init(self):
        create_folder(self.result_dir, self.args.data_env, self.args.local_dir)

        if self.data_env in ['local', 'mac']:
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=8), name='Asia/Shanghai')).strftime("%Y-%m-%d_%H_%M_%S")
        if self.data_env == 'gpu':
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d_%H_%M_%S")
        file_name_head = self.app + '_' + self.method + '_' + str(self.tar_id) + '_' + str(self.batch_size) + '_' + str(self.epoch) + '_' + str(self.seed) + '_' + str(self.lamda1) + '_' + str(self.lamda2)
        self.args.out_file = open(osp.join(self.args.result_dir, file_name_head + '.txt'), 'w')
        self.args.out_file.write(self._print_args() + '\n')
        self.args.out_file.flush()
        return self.args

    def record(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        return self.args

    def _print_args(self):
        s = "==========================================\n"
        for arg, content in self.args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s
