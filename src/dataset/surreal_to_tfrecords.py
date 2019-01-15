"""
Convert golf dataset to TFRecords.
All of images(num=10000) is training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import makedirs
from os.path import join, exists

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from .common import convert_to_example, ImageCoder
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.app.flags.DEFINE_string('img_dir',
                           '/home/windward/gcy/Golf/proj/data/SURREAL/summary/image',
                           'image data directory')
tf.app.flags.DEFINE_string('label_dir',
                           '/home/windward/gcy/Golf/proj/data/SURREAL/summary/labels',
                           'label data json directory')
tf.app.flags.DEFINE_string('output_dir', '/home/windward/gcy/Golf/proj/data/tfrecords_surreal',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1000,
                            'Number of shards in training TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def add_to_tfrecord(image_path, seg_path, pose, shape, gt2d, gt3d, coder, writer):
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()
    with tf.gfile.FastGFile(seg_path, 'rb') as f:
        seg_data = f.read()

    image = coder.decode_jpeg(image_data)
    height, width = image.shape[:2]
    assert image.shape[2] == 3

    min_pt = np.min(gt2d, axis=0)
    max_pt = np.max(gt2d, axis=0)
    center = (min_pt + max_pt) / 2.

    label = (seg_data, pose, shape, gt2d, gt3d)

    example = convert_to_example(image_data, image_path, height, width, label, center)

    writer.write(example.SerializeToString())


def package(all_images, labels, out_path, train_num, num_shards):
    """
    packages the images and labels into multiple tfrecords.
    """
    train_dir = join(out_path, 'train')
    if not exists(train_dir):
        makedirs(train_dir)
        os.system('chmod -R 777 %s' % train_dir)
    valid_dir = join(out_path, 'valid')
    if not exists(valid_dir):
        makedirs(valid_dir)
        os.system('chmod -R 777 %s' % valid_dir)
    train_out = join(train_dir, 'train_%03d.tfrecord')
    valid_out = join(valid_dir, 'valid_%03d.tfrecord')

    coder = ImageCoder()
    seg_paths = labels[0]
    pose = labels[1]
    shape = labels[2]
    gt2d = labels[3]
    gt3d = labels[4]

    i = 0
    fidx = 0
    while i < len(all_images):
        # Open new TFRecord file.
        if i < train_num:
            tf_filename = train_out % fidx
        else:
            tf_filename = valid_out % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(all_images) and j < num_shards:
                if i % 100 == 0:
                    print('Converting image %d/%d' % (i, len(all_images)))
                add_to_tfrecord(
                    all_images[i],
                    seg_paths[i],
                    pose[i, :],
                    shape[i, :],
                    gt2d[i, :, :],
                    gt3d[i, :, :],
                    coder,
                    writer)
                i += 1
                j += 1

        fidx += 1


def process_surreal(img_dir, label_dir, out_dir, num_shards_train):
    # all_names = os.listdir(img_dir)

    train_names = pd.read_csv(join('/home/windward/gcy/Golf/proj/data', 'surreal_train_names.csv'))['name'][:160000]
    train_num = len(train_names)
    print(train_num)
    val_names = pd.read_csv(join('/home/windward/gcy/Golf/proj/data', 'surreal_valid_names.csv'))['name'][:10000]
    print(len(val_names))
    all_names = pd.concat([train_names, val_names]).values

    num = len(all_names)
    pose = np.zeros((num, 72), dtype='float')
    shape = np.zeros((num, 10), dtype='float')
    gt2d = np.zeros((num, 24, 2), dtype='int')
    gt3d = np.zeros((num, 24, 3), dtype='int')

    all_images = []
    all_segs = []
    for i, name in enumerate(all_names):
        print(i, end='\r')
        all_images.append(join(img_dir, name))
        all_segs.append(join(label_dir, '../bodyseg', name[:-3] + 'png'))
        label_path = join(label_dir, name[:-3] + 'json')
        with open(label_path) as f:
            data = json.load(f)
        pose[i, :] = np.array(data['pose']).reshape((72,))
        shape[i, :] = np.array(data['shape']).reshape((10, ))
        gt2d[i, :, :] = np.array(data['joints2D']).reshape((2, 24)).T
        gt3d[i, :, :] = np.array(data['joints3D']).reshape((3, 24)).T

    labels = (all_segs, pose, shape, gt2d, gt3d)
    package(all_images, labels, out_dir, train_num, num_shards_train)


def main(argv):
    print('Saving results to %s' % FLAGS.output_dir)

    if not exists(FLAGS.output_dir):
        makedirs(FLAGS.output_dir)
        os.system('chmod -R 777 %s' % FLAGS.output_dir)
    process_surreal(FLAGS.img_dir, FLAGS.label_dir, FLAGS.output_dir, FLAGS.train_shards)


if __name__ == '__main__':
    tf.app.run()
