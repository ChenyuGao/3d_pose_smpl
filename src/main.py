""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from config import get_config, prepare_dirs, save_config
from data_loader import DataLoader
from trainer import Trainer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        dataset = DataLoader(config)
        train_iter = dataset.load_train()
        valid_iter = dataset.load_test()
    print("load data over")

    trainer = Trainer(config, train_iter, valid_iter)
    save_config(config)
    trainer.train()


if __name__ == '__main__':
    config = get_config()
    main(config)
