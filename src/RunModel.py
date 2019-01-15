from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import resnet50

import tensorflow as tf
import numpy as np
from os.path import join, dirname
from util import renderer as vis_util
from tf_smpl.batch_smpl import SMPL
from tf_smpl.batch_lbs import batch_rodrigues


class RunModel(object):
    def __init__(self, config, sess=None):
        self.config = config
        self.pretrained_model_path = config.pretrained_model_path

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.tar_img_size

        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)
        # Model Settings
        self.pose_params = 72
        self.build_test_model()
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver()
        print('Restoring checkpoint %s..' % self.pretrained_model_path)
        self.saver.restore(self.sess, self.pretrained_model_path)

    def build_test_model(self):
        self.out, _ = resnet50(self.images_pl, self.pose_params, is_training=True, reuse=False)

    def predict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        fetch_dict = {
            'pose': self.out,
        }

        results = self.sess.run(fetch_dict, feed_dict)

        return results['pose']