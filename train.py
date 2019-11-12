"""
Basic idea of FOTS:
0. Input: [1, 480, 640, 3]
1. Extract features from ResNet50/Mobilenet (sharedconv) --> [1, 120, 160, 32]
2. Input sharedconv to DetectionModel to detect text regions --> [1, 120, 160, 1] and [1, 120, 160, 5]
3. RoIRotate to crop and rotate sharedconv for text recognition --> [5, 8, 64, 32] [text_regions, height, width, chan]
4. Input RoIRotate to CRNN and recognize chars in every text region --> [5, 64, 78] [text_regions, width, chars]

logits 64
logit_length 64
blank_index 62

TODO
4. detection kernel size 3? originally it was 1 Keep 1
5. ctc loss mask?: https://github.com/yu20103983/FOTS/blob/master/FOTS/recognizer.py
7. recognition max pool 2d pool_size https://github.com/yu20103983/FOTS/blob/master/FOTS/recognizer.py keep as in paper
10. rotate expand_w = 60?
11. CRNN more same convs..? if i do this, the rnn outputs same fore each batch, how does that make sence?
12. Expand y_true boundries in classification - shrink_poly R param
13. There is no gradient link between backbone and recognition..? !!!!!

Sparse to dense etc non differentiable
https://github.com/Mainak431/List-of-Differentiable--OPs-and-Non-differentiable-OPs--in-Tensorflow
It looks like will have to do that on linux

https://www.tensorflow.org/graphics/api_docs/python/tfg/geometry/transformation/rotation_matrix_3d/from_axis_angle?authuser=2&hl=lt
https://databricks.com/tensorflow/tensorflow-in-3d
"""

import tensorflow as tf
import cv2
import numpy as np
# from icdar import generator
from synthtext import generator
from config import CHAR_VECTOR
from model_backbone import Backbone
from model_detection import Detection
from model_roirotate import RoIRotate, tfa_enabled
from model_recognition import Recognition
from utils import decode_to_text

# init
cpkt_dir = 'checkpoints/'
load_models = True
loss_hist = []
input_shape = [640, 640, 3]  # 160 / 480 / 640 / 800
model_sharedconv = Backbone(backbone='mobilenet', input_shape=input_shape)
model_detection = Detection()
model_RoIrotate = RoIRotate(tfa_enabled=tfa_enabled)
model_recognition = Recognition(num_classes=len(CHAR_VECTOR)+1, training=True, drop_prob=0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
save_iter = 100
iter = 0

if load_models:
    model_sharedconv.load_weights(cpkt_dir + 'sharedconv')
    model_detection.load_weights(cpkt_dir + 'detection')
    model_recognition.load_weights(cpkt_dir + 'recognition')

# -------- #
data_gen = generator(input_size=input_shape[0],
                     batch_size=1,
                     min_text_size=8,
                     random_scale=np.array([1.0]))

for x_batch in data_gen:

    with tf.GradientTape() as tape:

        # forward-prop
        sharedconv = model_sharedconv(x_batch['images'].copy())
        f_score_, geo_score_ = model_detection(sharedconv)
        features, ws = model_RoIrotate(sharedconv, x_batch['rboxes'], expand_px=1)
        logits = model_recognition(features)

        # loss
        loss_cls = model_detection.loss_classification2(x_batch['score_maps'], f_score_, x_batch['training_masks'])
        loss_iou, loss_angle = model_detection.loss_regression(x_batch['geo_maps'], geo_score_, x_batch['score_maps'], x_batch['training_masks'])
        loss_recongition = model_recognition.loss_recognition(y=x_batch['text_labels_sparse'], logits=logits, ws=ws)

        loss_cls *= 100  # 2
        loss_iou *= 100  # 2
        loss_angle *= 100  # 60
        loss_recongition /= 1  # 10
        model_loss = (loss_cls + loss_iou + loss_angle + loss_recongition)

    # backward-prop
    grads = tape.gradient(model_loss,
                          model_sharedconv.trainable_variables +
                          model_detection.trainable_variables +
                          model_recognition.trainable_variables)
    optimizer.apply_gradients(zip(grads,
                                  model_sharedconv.trainable_variables +
                                  model_detection.trainable_variables +
                                  model_recognition.trainable_variables))

    print(iter, loss_cls.numpy(), loss_iou.numpy(), loss_angle.numpy(),
          loss_recongition.numpy(), sum([len(i[0]) for i in x_batch['rboxes']]))

    iter += 1
    loss_hist.append([loss_cls.numpy(), loss_iou.numpy(), loss_angle.numpy(), loss_recongition.numpy()])

    # recognition results
    y_true = tf.sparse.to_dense(tf.SparseTensor(*x_batch['text_labels_sparse'])).numpy()
    decoded, _ = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)), sequence_length=[logits.shape[1]] * logits.shape[0])
    decoded = tf.sparse.to_dense(decoded[0]).numpy()
    print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in decoded[:4, :]],
          [decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in y_true[:4, :]])

    # save
    if iter % save_iter == 0:
        model_sharedconv.save_weights(cpkt_dir + 'sharedconv')
        model_detection.save_weights(cpkt_dir + 'detection')
        model_recognition.save_weights(cpkt_dir + 'recognition')

        with open('loss_test.txt', ['a' if load_models else 'w'][0]) as file:
            [file.write(str(s) + '\n') for s in loss_hist]
        loss_hist = []


# -------- #
# -------- #
# -------- #
data_gen = generator(input_size=input_shape[0], batch_size=1, min_text_size=10, random_scale=np.array([1.0]))
x_batch = data_gen.__next__()
sharedconv = model_sharedconv(x_batch['images'].copy())
f_score_, geo_score_ = model_detection(sharedconv)
features, ws = model_RoIrotate(sharedconv, x_batch['rboxes'], expand_px=2)
logits = model_recognition(features)

cv2.imshow('img', cv2.resize(x_batch['images'][0, ::], (640, 640)).astype(np.uint8))
cv2.imshow('scr', cv2.resize(x_batch['score_maps'][0, ::]*255, (640, 640)).astype(np.uint8))
cv2.imshow('pre', cv2.resize(f_score_[0, ::].numpy()*255, (640, 640)).astype(np.uint8))
cv2.imshow('msk', cv2.resize(x_batch['training_masks'][0, ::]*255, (640, 640)).astype(np.uint8))
[cv2.imshow(str(i), cv2.resize(geo_score_[0, :, :, i].numpy()*255, (512, 512)).astype(np.uint8)) for i in [0, 1, 2, 3, 4]]
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------- #
cv2.imshow('i', cv2.resize(x_batch['images'][0, ::], (512, 512)).astype(np.uint8))
cv2.waitKey(1)
for i in range(32):
    cv2.imshow(str(i), cv2.resize(sharedconv[-1, :, :, i].numpy()*255, (512, 512)).astype(np.uint8))
    cv2.imshow(str(i)+'_crop', (features.numpy() * 255).astype(np.uint8)[0, :, :, i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------- #
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)), sequence_length=[64]*logits.shape[0])
decoded = tf.sparse.to_dense(decoded[0]).numpy()
print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in decoded])

y_true = tf.sparse.to_dense(tf.SparseTensor(*x_batch['text_labels_sparse'])).numpy()
print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in y_true])
# ------------------------- #

# --------- #
# DEBUGGING #
# --------- #
data_gen = generator(input_size=input_shape[0], batch_size=1, min_text_size=5)
x_batch = next(data_gen)
stride = 1
shape = tuple([int(i / stride) for i in x_batch['images'].shape[1:3]])
dummy_input = cv2.resize(x_batch['images'][0, :, :, :].copy(), shape)[np.newaxis, :, :, :]
RoIrotate_test = RoIRotate(tfa_enabled=tfa_enabled, features_stride=stride)
features, ws = RoIrotate_test(dummy_input, x_batch['rboxes'], plot=True, expand_px=1)

# --- #
data_gen = generator(input_size=input_shape[0], batch_size=1, min_text_size=4)
x_batch = data_gen.__next__()
y_true = tf.sparse.to_dense(tf.SparseTensor(*x_batch['text_labels_sparse'])).numpy()
print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in y_true[:4, :]])

cv2.imshow('img', cv2.resize(x_batch['images'][0, ::], (512, 512)).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------- #
# check overlapping
orig_img = x_batch['images'][0].copy()
shrd_con = cv2.resize(sharedconv.numpy().copy()[0, :, :, :], orig_img.shape[:2])

for i in range(shrd_con.shape[-1]):
    overlay = np.tile(shrd_con[:, :, i].copy()[:, :, np.newaxis], 3) * 255*4
    combined = cv2.addWeighted(orig_img, 0.4, overlay, 0.3, 1)

    cv2.imshow('{}'.format(i), combined.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------- #
# check mobilenet layers
model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape)
layer_ids = [154, 119, 57, 30]
backbone_layers = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(index=i).output for i in layer_ids])

input = tf.keras.applications.mobilenet.preprocess_input(x_batch['images'][0].copy()[np.newaxis, :, :, :])
output = backbone_layers(input)

for i in range(36):
    conv = cv2.resize(output[3].numpy().copy()[0, :, :, i], (512, 512)) * 255
    cv2.imshow('{}'.format(i), conv.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
