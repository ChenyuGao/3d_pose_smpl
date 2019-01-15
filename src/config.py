"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json
import numpy as np
import tensorflow as tf

flags = tf.app.flags
curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '../models')
if not osp.exists(model_dir):
    makedirs(model_dir)
SMPL_MODEL_PATH = osp.join(model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH, 'path to the neurtral smpl model')

# Default pred-trained model path for the demo.
PRETRAINED_RESNET = osp.join(model_dir, 'resnet_v2_50', 'resnet_v2_50.ckpt')
flags.DEFINE_string('pretrained_resnet_model_path', PRETRAINED_RESNET,
                    'if not None, fine-tunes from this resnet50 ckpt')
flags.DEFINE_string('pretrained_model_path', None,
                    'if not None, fine-tunes from this ckpt')

# Don't change if testing:
flags.DEFINE_integer('tar_img_size', 224, 'Input image size to the network after preprocessing')

# Training settings:
DATA_DIR = osp.join(curr_path, '..', '..', 'data')
LOG_DIR = osp.join(curr_path, '..', 'logs')
flags.DEFINE_string('data_dir', DATA_DIR, 'data dir')
flags.DEFINE_string('log_dir', LOG_DIR, 'Where to save training models')
flags.DEFINE_string('model_dir', None, 'Where model will be saved -- filled automatically')
flags.DEFINE_integer('epoch', 100, '# of epochs to train')
flags.DEFINE_integer('train_images_num', 160000, 'number of images to train')
flags.DEFINE_integer('valid_images_num', 10000, 'number of images to valid')

# Hyper parameters:
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('batch_size', 32, 'Input image size to the network after preprocessing')

# Data augmentation
flags.DEFINE_integer('trans_max', 20, 'Value to jitter translation')
flags.DEFINE_float('scale_max', 1.23, 'Max value of scale jitter')
flags.DEFINE_float('scale_min', 0.8, 'Min value of scale jitter')


def get_config():
    config = flags.FLAGS
    config(sys.argv)

    return config


# ----- For training ----- #

def save_config(config):
    param_path = osp.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    config_dict = {}
    for k in dir(config):
        config_dict[k] = config.__getattr__(k)

    with open(param_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)


def prepare_dirs(config):
    # Continue training from a load_path
    postfix = []

    if config.lr != 0.001:
        postfix.append("lr%1.e" % config.lr)
    if config.trans_max != 20:
        postfix.append("transmax-%d" % config.trans_max)
    if config.scale_max != 1.23:
        postfix.append("scmax_%.3g" % config.scale_max)
    if config.scale_min != 0.8:
        postfix.append("scmin-%.3g" % config.scale_min)

    postfix = '_'.join(postfix)
    time_str = datetime.now().strftime("%m_%d_%H_%M")
    save_name = "%s_%s" % (time_str, postfix)
    config.model_dir = osp.join(config.log_dir, save_name)

    for path in [config.log_dir, config.model_dir]:
        if not osp.exists(path):
            print('making %s' % path)
            makedirs(path)
