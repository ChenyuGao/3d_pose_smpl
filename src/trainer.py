from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import resnet50

import tensorflow as tf
from tqdm import tqdm
from os.path import join, dirname
from util import renderer as vis_util
from tf_smpl.batch_smpl import SMPL
from ops import keypoint_l1_loss, compute_3d_loss, align_by_pelvis
from tf_smpl.batch_lbs import batch_rodrigues


class Trainer(object):
    def __init__(self, config, train_iter, valid_iter):
        self.train_iter = train_iter
        self.valid_iter = valid_iter

        self.config = config
        self.image_size = config.tar_img_size
        self.model_dir = config.model_dir

        # self.smpl_model_path = config.smpl_model_path
        self.pretrained_resnet_model_path = config.pretrained_resnet_model_path
        # Data size
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch
        train_images_num = config.train_images_num
        valid_images_num = config.valid_images_num
        self.train_num_itr_per_epoch = train_images_num / self.batch_size
        self.valid_num_itr_per_epoch = valid_images_num / self.batch_size
        self.lr = config.lr
        self.Var = []
        self.pose_params = 72

        # self.smpl = SMPL(self.smpl_model_path)

    def build_model(self, image_batch, rot_batch, pose_batch, shape_batch, gt2d_batch, gt3d_batch):
        self.out, self.Var = resnet50(image_batch, self.pose_params, reuse=False)
        pred_pose = self.out
        # shapes = self.train_loader['shape']
        # pred_rot = batch_rodrigues(tf.reshape(self.out, [self.batch_size, -1, 3]))
        self.loss_pose = self.get_3d_loss(pose_batch, pred_pose)

    def get_3d_loss(self, pose_batch, pred_pose):
        """
        pred_Rot is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        pred_js_3d is N x 24 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (24*3)
        """
        # pred_rot = tf.reshape(pred_rot, [self.batch_size, -1])
        pred_pose = tf.reshape(pred_pose, [self.batch_size, -1])
        # pred_params = tf.concat([pred_Rot, pre_shape], 1, name="params_pred")
        # 24*9+10 = 226
        # gt_rot = tf.reshape(loader['rot'], [self.batch_size, -1])
        gt_pose = tf.reshape(pose_batch, [self.batch_size, -1])
        # gt_shape = loader['shape']
        # gt_params = tf.concat([loader['rot'], loader['shape']], 1, name="params_gt")
        # loss_smpl = compute_3d_loss(pred_params, gt_params)
        # loss_rot = compute_3d_loss(pred_rot, gt_rot)
        loss_pose = compute_3d_loss(pred_pose, gt_pose)

        # gt_joints = loader['gt3d']
        # pred_joints = pred_js_3d[:, :, :]
        # # Align the joints by pelvis.
        # pred_joints = align_by_pelvis(pred_joints)
        # pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])
        # gt_joints = tf.reshape(gt_joints, [self.batch_size, 24, 3])
        # gt_joints = align_by_pelvis(gt_joints)
        # gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])
        #
        # loss_joints_3d = compute_3d_loss(pred_joints, gt_joints)

        return loss_pose

    def train(self):
        best_loss_pose = 100
        early_stopping_patience = 5
        reduce_lr_patience = 3
        count = 0
        factor = 0.1
        min_lr = 0.0001

        is_training = tf.placeholder(tf.bool)
        data_iter = tf.cond(is_training, lambda: self.train_iter.get_next(), lambda: self.valid_iter.get_next())
        image_batch, rot_batch, pose_batch, shape_batch, gt2d_batch, gt3d_batch = data_iter
        self.build_model(image_batch, rot_batch, pose_batch, shape_batch, gt2d_batch, gt3d_batch)
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_pose)

        train_step = 0
        valid_step = 0
        summary_train = [tf.summary.scalar("train_loss/loss_pose", self.loss_pose)]
        summary_train_op = tf.summary.merge(summary_train)
        summary_valid = [tf.summary.scalar("valid_loss/loss_pose", self.loss_pose)]
        summary_valid_op = tf.summary.merge(summary_valid)

        pre_train_saver = tf.train.Saver(self.Var)

        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options)

        with tf.Session(config=sess_config) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)

            print('Restoring checkpoint %s..' % self.model_dir)
            pre_train_saver.restore(sess, self.pretrained_resnet_model_path)

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            for epoch in range(self.max_epoch):
                epoch += 1
                # train
                train_loss_poses = []
                sess.run(self.train_iter.initializer)
                pbar = tqdm(range(int(self.train_num_itr_per_epoch)))
                for step in pbar:
                    step += 1
                    # r = sess.run(batch_loader['image'], feed_dict={is_training: True})
                    # print(r.shape)
                    train_step += 1
                    fetch_dict_train = {
                        "summary": summary_train_op,
                        "loss_pose": self.loss_pose,
                        # "loss_rot": self.loss_rot,
                        # "loss_joints_3d": self.loss_joints_3d,
                        "opt": optimizer,
                    }

                    train_result = sess.run(fetch_dict_train, feed_dict={is_training: True})

                    summary_writer.add_summary(train_result['summary'], global_step=train_step)
                    summary_writer.flush()
                    loss_pose = train_result['loss_pose']
                    # loss_rot = result['loss_rot']
                    # loss_joints_3d = result['loss_joints_3d']
                    train_loss_poses.append(loss_pose)
                    pbar.set_description("[itr %d/epoch %d]: loss_pose: %.4f" % (step, epoch, loss_pose))

                train_loss_pose = sum(train_loss_poses) / len(train_loss_poses)

                # valid
                valid_loss_poses = []
                sess.run(self.valid_iter.initializer)
                for step in range(int(self.valid_num_itr_per_epoch)):
                    valid_step += 1
                    fetch_dict_valid = {
                        "summary": summary_valid_op,
                        "loss_pose": self.loss_pose,
                        # "loss_rot": self.loss_rot,
                        # "loss_joints_3d": self.loss_joints_3d,
                    }
                    valid_result = sess.run(fetch_dict_valid, feed_dict={is_training: False})
                    summary_writer.add_summary(valid_result['summary'], global_step=valid_step)
                    summary_writer.flush()
                    loss_pose = valid_result['loss_pose']
                    valid_loss_poses.append(loss_pose)

                valid_loss_pose = sum(valid_loss_poses) / len(valid_loss_poses)

                print("train_loss_pose: %.4f    valid_loss_pose: %.4f" % (train_loss_pose, valid_loss_pose))

                f = open(join(self.config.model_dir, 'logs.txt'), 'a+')
                f.write("[epoch %d]: train_loss_pose: %.4f    valid_loss_pose: %.4f\n" %
                        (epoch, train_loss_pose, valid_loss_pose))

                if best_loss_pose > valid_loss_pose:
                    saver.save(sess, join(self.model_dir, 'model.ckpt'), global_step=epoch)
                    best_loss_pose = valid_loss_pose
                    count = 0
                else:
                    count += 1
                if count == early_stopping_patience:
                    break
                if count == reduce_lr_patience:
                    if self.lr * factor >= min_lr:
                        self.lr *= factor
                        print("lr reduce to: %f" % self.lr)
                        f.write("lr reduce to: %f\n" % self.lr)

                f.close()

            # coord.request_stop()
            # coord.join(threads)

        print('Finish training on %s' % self.model_dir)
