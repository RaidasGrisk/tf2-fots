import tensorflow as tf
import numpy as np


class Detection(tf.keras.Model):

    """
    Input: shared convs from ResNet50 [1, 120, 160, 64]
    Output: text detection [1, 120, 160, 1] and [1, 120, 160, 5] (f_score and f_geometry)
    Loss: classification (f_score) + regression (f_geometry)

    'The first channel computes the probability of each
    pixel being a positive sample (f_score). <...>
    For each positive sample, the following 4 channels
    predict its distances to top, bottom, left, right sides of
    the bounding box that contains this pixel, and the last channel
    predicts the orientation of the related bounding box (f_geometry)'
    """

    def __init__(self):
        super(Detection, self).__init__()

        self.f_score = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=tf.nn.sigmoid)
        self.geo_map = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), padding='same', activation=tf.nn.sigmoid)
        self.angle_map = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=tf.nn.sigmoid)

    def __call__(self, input):

        f_score = self.f_score(input)
        geo_map = self.geo_map(input) * 512
        angle_map = (self.angle_map(input) - 0.5) * np.pi / 2
        f_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return f_score, f_geometry

    @staticmethod
    def loss_classification(f_score, f_score_, training_mask):
        """
        :param f_score: ground truth of text
        :param f_score_: prediction os text
        :param training_mask: mask used in training, to ignore some text annotated by ###
                :return:
        """
        eps = 1e-5
        intersection = tf.reduce_sum(f_score * f_score_ * training_mask)
        union = tf.reduce_sum(f_score * training_mask) + tf.reduce_sum(f_score_ * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    @staticmethod
    def loss_classification2(f_score, f_score_, training_mask):
        """
        cross enthropy
        :param f_score: ground truth of text
        :param f_score_: prediction os text
        :param training_mask: mask used in training, to ignore some text annotated by ###
                :return:
        """

        loss = tf.keras.losses.binary_crossentropy(f_score * training_mask,
                                                   f_score_ * training_mask,
                                                   from_logits=False,
                                                   label_smoothing=0)
        return tf.reduce_mean(loss)
    
    @staticmethod
    def loss_regression(geo_score, geo_score_, f_score, training_mask):
        """
        :param geo_score: ground truth of geometry
        :param geo_score_: prediction of geometry
        """

        # -------- #
        box_params, angle = tf.split(value=geo_score, num_or_size_splits=[4, 1], axis=3)
        box_params_, angle_ = tf.split(value=geo_score_, num_or_size_splits=[4, 1], axis=3)

        loss_angle = 1 - tf.cos(angle_ - angle)
        loss_angle = tf.reduce_mean(loss_angle * 20 * f_score * training_mask)
        loss_iou = tf.reduce_mean(tf.square((box_params - box_params_) * f_score * training_mask))
        # -------- #

        # # d1 -> top, d2->right, d3->bottom, d4->left
        # d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=geo_score, num_or_size_splits=5, axis=3)
        # d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=geo_score_, num_or_size_splits=5, axis=3)
        # area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        # area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        # w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        # h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        # area_intersect = w_union * h_union
        # area_union = area_gt + area_pred - area_intersect
        # L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
        # L_theta = 1 - tf.cos(theta_pred - theta_gt)
        # # L_g = L_AABB + 20 * L_theta  # 10 in paper
        # loss_iou = tf.reduce_mean(L_AABB * f_score * training_mask)
        # loss_angle = tf.reduce_mean(L_theta * 20 * f_score * training_mask)

        return loss_iou, loss_angle

    def loss_detection(self, f_score, f_score_, geo_score, geo_score_, training_mask):

        loss_clasification = self.loss_classification2(f_score, f_score_, training_mask)
        loss_iou, loss_angle = self.loss_regression(geo_score, geo_score_, f_score, training_mask)
        return loss_iou + loss_angle + loss_clasification
