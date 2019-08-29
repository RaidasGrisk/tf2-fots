import tensorflow as tf
import cv2
import numpy as np
from icdar import generator


class SharedConv(tf.keras.Model):

    def __init__(self, input_shape=(480, 640, 3)):
        super(SharedConv, self).__init__()

        self.resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
        self.resnet.trainable = False
        self.layer_ids = [174, 142, 80, 12]
        self.resnet_layers = tf.keras.models.Model(
            inputs=self.resnet.input,
            outputs=[self.resnet.get_layer(index=i).output for i in self.layer_ids])

        self.l1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=tf.nn.relu)

        self.h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

        self.g1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

    def call(self, input):

        """
        Example of what is actually happening here.
        No loop for it makes it easier to understand.
        We are extracting 4 different layers from ResNet50.
        Dims will depend on the input size, but the relative
        sizes will stay the same and that is what matters.

        input       [1, 480, 640, 3]
        layer_1     [1, 15, 20, 2048]   (one of the last layers, 174th op in the model)
        layer_2     [1, 30, 40, 1024]   (142)
        layer_3     [1, 60, 80, 512]    (80)
        layer_4     [1, 120, 160, 64]   (one of the first layers, 12th op in the model)

        step 1: double the size of layer_1 -> [1, 30, 40, 2048]

        step 2: concat layer_1 [1, 30, 40, 2048] and layer_2 [1, 30, 40, 1024] -> [1, 30, 40, 3072]
                conv with stride 1 to -> [1, 30, 40, 128] this is just to decrease the num of filters (last dim)
                conv with stride 3 to -> [1, 30, 40, 128]
                double the size of this layer -> [1, 60, 80, 128]

        step 3: [1, 60, 80, 128] + [1, 60, 80, 512] -> [1, 60, 80, 640]
                conv to -> [1, 60, 80, 64]
                conv to -> [1, 60, 80, 64]
                resize  -> [1, 120, 160, 64]

        step 4: [1, 120, 160, 64] + [1, 120, 160, 64] -> [1, 120, 160, 128]
                conv to -> [1, 120, 160, 32]
                conv to -> [1, 120, 160, 32]
                conv to -> [1, 120, 160, 32]

        """

        # layers extracted from Resnet:
        # 1st is the farthest one (near the end of the net),
        # 4th is the closest one (near the beggining)
        layer_1, layer_2, layer_3, layer_4 = self.resnet_layers(input)

        # step 1
        # layer_1 -> layer_1
        layer_shape = tf.shape(layer_1)
        layer_1 = tf.image.resize(layer_1, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 2
        # layer_1 + layer_2 -> layer_12
        layer_12_conc = self.l1(tf.concat([layer_1, layer_2], axis=-1))
        layer_12_conv = self.h1(layer_12_conc)
        layer_shape = tf.shape(layer_2)
        layer_12 = tf.image.resize(layer_12_conv, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 3
        # layer_12 + layer_3 -> layer_123
        layer_123_conc = self.l2(tf.concat([layer_12, layer_3], axis=-1))
        layer_123_conv = self.h2(layer_123_conc)
        layer_shape = tf.shape(layer_3)
        layer_123 = tf.image.resize(layer_123_conv, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # step 4
        # layer_123 + layer_4 -> layer_1234
        layer_1234_conc = self.l3(tf.concat([layer_123, layer_4], axis=-1))
        layer_1234_conv = self.h3(layer_1234_conc)
        layer_1234 = self.g1(layer_1234_conv)

        return layer_1234


class DetectionModel(tf.keras.Model):

    def __init__(self):
        super(DetectionModel, self).__init__()

        self.f_score = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid)
        self.geo_map = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation=tf.nn.sigmoid)
        self.angle_map = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid)

    def call(self, input):

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
    def loss_regression(geo_score, geo_score_):
        """
        :param geo_score: ground truth of geometry
        :param geo_score_: prediction of geometry
        :return:
        """

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=geo_score, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=geo_score_, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return L_g

    def loss_detection(self, f_score, f_score_, geo_score, geo_score_, training_mask):
        """
        :param f_score:
        :param f_score_:
        :param geo_score:
        :param geo_score_:
        :param training_mask:
        :return:
        """

        loss_clasification = self.loss_classification(f_score, f_score_, training_mask)
        loss_regression = self.loss_regression(geo_score, geo_score_)
        return tf.reduce_mean(loss_regression * f_score * training_mask) + loss_clasification * 0.01


class RoIRotateModel(tf.keras.Model):
    def __init__(self):
        super(RoIRotateModel, self).__init__()


class RecognitionModel(tf.keras.Model):
    def __init__(self):
        super(RecognitionModel, self).__init__()


# -------- #

model_sharedconv = SharedConv(input_shape=(320, 320, 3))
model_detection = DetectionModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=5)
[print(i.name) for i in model_sharedconv.trainable_variables + model_detection.trainable_variables]

max_iter = 100
iter = 0
for x_batch in generator(input_size=320, batch_size=1):

    with tf.GradientTape() as tape:
        sharedconv = model_sharedconv(np.array(x_batch['images']))
        f_score_, geo_score_ = model_detection(sharedconv)
        loss = model_detection.loss_detection(np.array(x_batch['score_maps']), f_score_,
                                              np.array(x_batch['geo_maps']), geo_score_,
                                              np.array(x_batch['training_masks']))

    grads = tape.gradient(loss, model_sharedconv.trainable_variables + model_detection.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_sharedconv.trainable_variables + model_detection.trainable_variables))
    print(loss.numpy() * 100)

    iter += 1
    if iter == max_iter:
        break


# -------- #
for x_batch in generator(input_size=160, batch_size=2):
    break

cv2.imshow('a', cv2.resize(x_batch['images'][0], (512, 512)).astype(np.uint8))
cv2.imshow('b', cv2.resize(x_batch['score_maps'][0]*255, (512, 512)).astype(np.uint8))
cv2.imshow('c', cv2.resize(f_score_[0, ::].numpy()*255, (512, 512)).astype(np.uint8))
cv2.imshow('d', cv2.resize(x_batch['training_masks'][0]*255, (512, 512)).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
