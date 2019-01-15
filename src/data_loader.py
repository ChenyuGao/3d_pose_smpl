"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob
import tensorflow as tf
from tf_smpl.batch_lbs import batch_rodrigues
from util import data_utils


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.max_epoch = config.epoch
        self.dataset_dir = config.data_dir
        self.batch_size = config.batch_size
        self.output_size = config.tar_img_size
        # Jitter params:
        self.trans_max = config.trans_max
        self.scale_range = [config.scale_min, config.scale_max]

    def load_train(self):
        files_path = join(self.config.data_dir, 'tfrecords', 'train/*.tfrecord')
        files = tf.train.match_filenames_once(files_path)
        shuffle_buffer = 5000
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: self.read_data(x, True))
        dataset = dataset.shuffle(shuffle_buffer).batch(self.batch_size)
        dataset = dataset.repeat(self.max_epoch)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def load_test(self):
        files_path = join(self.config.data_dir, 'tfrecords_surreal', 'valid/*.tfrecord')
        files = tf.train.match_filenames_once(files_path)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(lambda x: self.read_data(x, False))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.max_epoch)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def read_data(self, file, is_traing):
        with tf.name_scope(None, 'read_data', [file]):
            image, image_size, center, fname, pose, shape, gt2d, gt3d, seg = \
                data_utils.parse_example_proto(file)
            if is_traing:
                # image, gt2d = self.train_image_preprocessing(image, image_size, center, gt2d, pose=None, gt3d=None)
                image, gt2d = self.train_image_preprocessing(seg, image_size, center, gt2d, pose=None, gt3d=None)
            else:
                # image, gt2d = self.test_image_preprocessing(image, center, gt2d)
                image, gt2d = self.test_image_preprocessing(seg, center, gt2d)
            gt2d = tf.reshape(gt2d, [-1])
            gt3d = tf.reshape(gt3d, [-1])

            # Convert pose to rotation.
            rotations = batch_rodrigues(tf.reshape(pose, [-1, 3]))
            rotations = tf.reshape(rotations, [-1])

            return image, rotations, pose, shape, gt2d, gt3d

    def train_image_preprocessing(self, image, image_size, center, gt2d, pose=None, gt3d=None):
        margin = tf.to_int32(self.output_size / 2)
        with tf.name_scope(None, 'train_image_preprocessing', [image, center, gt2d]):
            keypoints = tf.transpose(gt2d[:, :])

            # Randomly shift center.
            center = data_utils.jitter_center(center, self.trans_max)
            # randomly scale image.
            image, keypoints, center = data_utils.jitter_scale(
                image, image_size, keypoints, center, self.scale_range)

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)
            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop])

            if pose is not None:
                crop, crop_kp, new_pose, new_gt3d = data_utils.random_flip(    # TODO
                    crop, crop_kp, pose, gt3d)
            else:
                crop, crop_kp = data_utils.random_flip(crop, crop_kp)

            # Normalize kp output to [-1, 1]
            final_label = 2.0 * (crop_kp / self.output_size) - 1.0

            # rescale image from [0, 1] to [-1, 1]
            crop = data_utils.rescale_image(crop)

            if pose is not None:
                return crop, tf.transpose(final_label), new_pose, new_gt3d
            else:
                return crop, tf.transpose(final_label)

    def test_image_preprocessing(self, image, center, gt2d):
        margin = tf.to_int32(self.output_size / 2)
        with tf.name_scope(None, 'test_image_preprocessing', [image, center, gt2d]):
            keypoints = tf.transpose(gt2d[:, :])
            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)
            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop])
            # Normalize kp output to [-1, 1]
            final_label = 2.0 * (crop_kp / self.output_size) - 1.0
            return crop, tf.transpose(final_label)
