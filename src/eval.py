from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tqdm import tqdm
from util import renderer as vis_util
from util import image as img_util
from util import data_utils
from RunModel import RunModel

flags.DEFINE_string('img_path', '../eval/data/', 'Tfrecords path to run')
flags.DEFINE_string('save_path', '../eval/smpl_param', 'Result path to save')
flags.DEFINE_string('pretrained_model_path', '../logs/12_28_21_12_/model.ckpt-2', 'pretrained model path')
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
    pad_image = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return pad_image


def main(img_path, save_path):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    img_path = os.path.join(img_path, '*.tfrecord')
    files = sorted(glob(img_path))

    all_images, all_centers, all_pose, all_shape, all_gt2d, all_gt3d = [], [], [], [], [], []
    image_data_pl = tf.placeholder(dtype=tf.string)
    decode_op = tf.image.decode_jpeg(image_data_pl)
    decode_op = tf.image.convert_image_dtype(decode_op, dtype=tf.float32)
    decode_op = tf.image.resize_image_with_crop_or_pad(decode_op, 224, 224)

    for serialized_ex in tf.python_io.tf_record_iterator(files[0]):
        example = tf.train.Example()
        example.ParseFromString(serialized_ex)
        image_data = example.features.feature['image'].bytes_list.value[0]
        image = sess.run(decode_op, feed_dict={image_data_pl:  image_data})
        image = image_preprocessing(image, 224)

        pose = example.features.feature['pose'].float_list.value
        shape = example.features.feature['shape'].float_list.value
        center = example.features.feature['center'].int64_list.value

        pose = np.array(pose).reshape((-1, 3))
        shape = np.array(shape)
        center = np.array(center)

        gt2d = example.features.feature['gt2d'].float_list.value
        gt2d = np.array(gt2d).reshape(-1, 2)
        gt3d = example.features.feature['gt3d'].float_list.value
        gt3d = np.array(gt3d).reshape(-1, 3)

        all_images.append(image)
        all_centers.append(center)
        all_pose.append(pose)
        all_shape.append(shape)
        all_gt2d.append(gt2d)
        all_gt3d.append(gt3d)

    model = RunModel(config, sess=sess)

    # TODO
    image = np.expand_dims(image, 0)
    theta = model.predict(image)
    theta = np.round(theta.reshape((-1, 24, 3)), 6)
    print(theta[0, 0, :])
    # np.savetxt(os.path.join(save_path, img_name[:-3]+'txt'), theta)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)

    main(config.img_path, config.save_path)
