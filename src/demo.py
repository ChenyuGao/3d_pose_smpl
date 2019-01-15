from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import os
import json
import tensorflow as tf
import cv2
from tqdm import tqdm
from util import renderer as vis_util
from util import image as img_util
from RunModel import RunModel

flags.DEFINE_string('img_path', '../demo/image/valid_1', 'Image path to run')
flags.DEFINE_string('save_path', '../demo/smpl_param/valid_1', 'Result path to save')
flags.DEFINE_string('pretrained_model_path', '../logs/01_08_23_28_/model.ckpt-2', 'pretrained model path')
flags.DEFINE_integer('tar_img_size', 224, 'Input image size to the network after preprocessing')
flags.DEFINE_integer('batch_size', 1, 'Input image size to the network after preprocessing')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def image_preprocessing(image, size):
    h, w = image.shape[0], image.shape[1]
    m = max(w, h)
    ratio = size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    resized = cv2.resize(image, (new_w, new_h))
    target_h = size
    target_w = size
    top = (target_h - new_h) // 2
    bottom = (target_h - new_h) // 2
    if top + bottom + h < target_h:
        bottom += 1
    left = (target_w - new_w) // 2
    right = (target_w - new_w) // 2
    if left + right + w < target_w:
        right += 1
    pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return pad_image


def compute_loss(pre_pose, gt_pose):
    return ((pre_pose - gt_pose) ** 2).mean()


def main(img_path, save_path):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = RunModel(config, sess=sess)

    if os.path.isdir(img_path):
        img_list = sorted(os.listdir(img_path))
    else:
        img_list = [os.path.basename(img_path)]
        img_path = os.path.dirname(img_path)
    for img_name in img_list:
        if img_name[-3:] != 'jpg':
            continue
        image = cv2.imread(os.path.join(img_path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_preprocessing(image, config.tar_img_size)
        # image = cv2.resize(image, (224, 224))
        image = image * 1.0 / 127.5 - 1.0

        # Add batch dimension: 1 x D x D x 3
        image = np.expand_dims(image, 0)
        theta = model.predict(image)
        np.savetxt(os.path.join(save_path, img_name[:-3]+'txt'), theta, fmt='%1.6f')

        with open(os.path.join(img_path, '../../label/valid_1', img_name[:-3]+'json')) as f:
            label = json.load(f)
        gt_pose = label['pose']
        loss = compute_loss(theta, gt_pose)
        print(loss)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    main(config.img_path, config.save_path)
