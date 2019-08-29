import tensorflow as tf
import cv2
import numpy as np
from icdar import generator

# init model
layers = [174, 142, 80, 12]
resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=(480, 640, 3))
resnet_layers = tf.keras.models.Model(inputs=resnet.input, outputs=[resnet.get_layer(index=i).output for i in layers])

names = [i.name for i in resnet.layers]
[names.index(i) for i in layers]

# input
image = cv2.imread('test_images/1_pontifically_58805.jpg', 3)
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

# preprocess
x = tf.keras.applications.resnet50.preprocess_input(image)
x = x[np.newaxis, :, :]

# ------- #
f = resnet_layers(x)
g = [None, None, None, None]
h = [None, None, None, None]
num_outputs = [None, 128, 64, 32]
for i in range(4):
    if i == 0:
        h[i] = f[i]
    else:
        c1_1 = tf.keras.layers.Conv2D(filters=num_outputs[i], kernel_size=1, padding='same', activation=tf.nn.relu)(tf.concat([g[i - 1], f[i]], axis=-1))
        h[i] = tf.keras.layers.Conv2D(filters=num_outputs[i], kernel_size=3, padding='same', activation=tf.nn.relu)(c1_1)
    if i <= 2:
        g[i] = tf.image.resize(h[i], size=[tf.shape(h[i])[1]*2,  tf.shape(h[i])[2]*2])
    else:
        g[i] = tf.keras.layers.Conv2D(filters=num_outputs[i], kernel_size=3, padding='same', activation=tf.nn.relu)(h[i])
    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))


class SharedConv(tf.keras.Model):

    """
    res_1 = self.resnet.layers[174].output
    res_2 = self.resnet.layers[142].output
    res_3 = self.resnet.layers[80].output
    res_4 = self.resnet.layers[12].output

    """

    def __init__(self, input_shape=(360, 360, 3)):
        super(SharedConv, self).__init__()

        self.resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
        self.resnet.trainable = False
        self.inner_layers = [174, 142, 80, 12]
        print([i.name for i in self.resnet.layers])
        self.resnet_layers = tf.keras.models.Model(inputs=self.resnet.input,
                                                   outputs=[self.resnet.get_layer(index=i).output for i in self.inner_layers])

        # concatinating resnet layers
        self.l1 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=tf.nn.relu)
        self.l3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation=tf.nn.relu)

        self.h1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.h3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

        self.g1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

        # f scores and geometry
        self.f_score = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid)
        self.geo_map = tf.keras.layers.Conv2D(filters=4, kernel_size=1, padding='same', activation=tf.nn.sigmoid)
        self.angle_map = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid)

    def call(self, input):

        resnet_layers = self.resnet_layers(input)

        # 1
        layer_shape = tf.shape(resnet_layers[0])
        g1 = tf.image.resize(resnet_layers[0], size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # 2
        c1_1 = self.l1(tf.concat([g1, resnet_layers[1]], axis=-1))
        h1 = self.h1(c1_1)
        layer_shape = tf.shape(resnet_layers[1])
        g2 = tf.image.resize(h1, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # 3
        c1_2 = self.l2(tf.concat([g2, resnet_layers[2]], axis=-1))
        h2 = self.h2(c1_2)
        layer_shape = tf.shape(resnet_layers[2])
        g3 = tf.image.resize(h2, size=[layer_shape[1] * 2, layer_shape[2] * 2])

        # 4
        c1_3 = self.l3(tf.concat([g3, resnet_layers[3]], axis=-1))
        h3 = self.h3(c1_3)
        g4 = self.g1(h3)

        # f_score and f_geometry
        f_score = self.f_score(g4)
        geo_map = self.geo_map(g4) * 512
        angle_map = (self.angle_map(g4) - 0.5) * np.pi / 2
        f_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return g4, f_score, f_geometry

    def dice_coefficient(self, f_score, f_score_, training_mask):
        '''
        dice loss
        :param f_score:
        :param f_score_:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = tf.reduce_sum(f_score * f_score_ * training_mask)
        union = tf.reduce_sum(f_score * training_mask) + tf.reduce_sum(f_score_ * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def loss(self, f_score, f_score_, geo_score, geo_score_, training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param f_score: ground truth of text
        :param f_score_: prediction os text
        :param geo_score: ground truth of geometry
        :param geo_score_: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = self.dice_coefficient(f_score, f_score_, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

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

        return tf.reduce_mean(L_g * f_score * training_mask) + classification_loss


model_sharedconv = SharedConv(input_shape=(160, 160, 3))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
[print(i.name) for i in model_sharedconv.trainable_variables]

max_iter = 10
iter = 0
for x_batch in generator(input_size=160, batch_size=2):

    with tf.GradientTape() as tape:
        sharedconv, f_score_, geo_score_ = model_sharedconv(np.array(x_batch['images']))
        loss = model_sharedconv.loss(np.array(x_batch['score_maps']), f_score_,
                                     np.array(x_batch['geo_maps']), geo_score_,
                                     np.array(x_batch['training_masks']))

    grads = tape.gradient(loss, model_sharedconv.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_sharedconv.trainable_variables))
    print(loss.numpy() * 100)

    print((prev - model_sharedconv.trainable_variables[0]).numpy().sum())
    prev = model_sharedconv.trainable_variables[0]

    # if loss.numpy() *100 == 0.9999999776482582:
    #     break

    iter += 1
    if iter == max_iter:
        break

# -------- #

f_score = np.array(x_batch['score_maps'])
geo_score = np.array(x_batch['geo_maps'])
training_mask = np.array(x_batch['training_masks'])