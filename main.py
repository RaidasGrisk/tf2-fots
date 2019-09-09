"""
# Basic idea of FOTS:
# 0. Input: [1, 480, 640, 3]
# 1. Extract features from ResNet50/Mobilenet (sharedconv) --> [1, 120, 160, 32]
# 2. Input sharedconv to DetectionModel to detect text regions --> [1, 120, 160, 1] and [1, 120, 160, 5]
# 3. RoIRotate to crop and rotate sharedconv for text recognition --> [5, 8, 64, 32] [text_regions, height, width, chan]
# 4. Input RoIRotate to CRNN and recognize chars in every text region --> [5, 64, 78] [text_regions, width, chars]
#
# logits 64
# logit_length 64
# blank_index 62
#
# """


import tensorflow as tf
import cv2
import numpy as np
from icdar import generator
from config import CHAR_VECTOR
from model_backbone import Backbone
from model_detection import Detection
from model_roirotate import RoIRotate
from model_recognition import Recognition
from utils import decode_to_text

# init
cpkt_dir = 'checkpoints/'
load_models = True
loss_hist = []
input_shape = [640, 640, 3]
model_sharedconv = Backbone(backbone='mobilenet', input_shape=input_shape)
model_detection = Detection()
model_RoIrotate = RoIRotate()
model_recognition = Recognition(num_classes=len(CHAR_VECTOR)+1, training=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

if load_models:
    model_sharedconv.load_weights(cpkt_dir + 'sharedconv')
    model_detection.load_weights(cpkt_dir + 'detection')
    model_recognition.load_weights(cpkt_dir + 'recognition')

# -------- #
max_iter = 10000000
save_iter = 100
iter = 0
data_gen = generator(input_size=input_shape[0], batch_size=1)  # 160 / 480 / 640 / 800
for x_batch in data_gen:

    with tf.GradientTape() as tape:

        # forward-prop
        sharedconv = model_sharedconv(x_batch['images'].copy())
        f_score_, geo_score_ = model_detection(sharedconv)
        features, ws = model_RoIrotate(sharedconv, x_batch['rboxes'])
        logits = model_recognition(features)

        # loss
        loss_detection = model_detection.loss_detection(x_batch['score_maps'], f_score_,
                                                        x_batch['geo_maps'], geo_score_,
                                                        x_batch['training_masks'])

        loss_recongition = model_recognition.loss_recognition(y=x_batch['text_labels_sparse'],
                                                              logits=logits,
                                                              ws=ws)
        model_loss = 1 * loss_detection + 1 * loss_recongition

    # backward-prop
    grads = tape.gradient(model_loss,
                          model_sharedconv.trainable_variables +
                          model_detection.trainable_variables +
                          model_recognition.trainable_variables)
    optimizer.apply_gradients(zip(grads,
                                  model_sharedconv.trainable_variables +
                                  model_detection.trainable_variables +
                                  model_recognition.trainable_variables))
    print(iter, loss_detection.numpy(), loss_recongition.numpy(), x_batch['image_fns'], len(x_batch['rboxes'][0][0]))

    iter += 1
    loss_hist.append([loss_detection.numpy(), loss_recongition.numpy(), model_loss.numpy()])

    # save
    if iter % save_iter == 0:
        model_sharedconv.save_weights(cpkt_dir + 'sharedconv')
        model_detection.save_weights(cpkt_dir + 'detection')
        model_recognition.save_weights(cpkt_dir + 'recognition')

        with open('loss_test.txt', 'w') as file:
            [file.write(str(s) + '\n') for s in loss_hist]

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)),
                                                 sequence_length=[64] * logits.shape[0])
    decoded = tf.sparse.to_dense(decoded[0]).numpy()
    print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in decoded[:4, :]])

    # stop
    if iter == max_iter:
        break


# -------- #
# -------- #
# -------- #
x_batch = data_gen.__next__()
cv2.imshow('img', cv2.resize(x_batch['images'][0, ::], (512, 512)).astype(np.uint8))
# cv2.imshow('scr', cv2.resize(x_batch['score_maps'][0, ::]*255, (512, 512)).astype(np.uint8))q
# cv2.imshow('pre', cv2.resize(f_score_[0, ::].numpy()*255, (512, 512)).astype(np.uint8))
# cv2.imshow('msk', cv2.resize(x_batch['training_masks'][0, ::]*255, (512, 512)).astype(np.uint8))
# [cv2.imshow(str(i), cv2.resize(geo_score_[0, :, :, i].numpy()*255, (512, 512)).astype(np.uint8)) for i in [0, 1, 2, 3, 4]]
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------- #
cv2.imshow('i', cv2.resize(x_batch['images'][0, ::], (512, 512)).astype(np.uint8))
cv2.waitKey(1)
for i in range(32):
    cv2.imshow(str(i), cv2.resize(sharedconv[0, :, :, i:i+1].numpy()*255, (512, 512)).astype(np.uint8))
    cv2.imshow(str(i)+'_crop', (features.numpy() * 255).astype(np.uint8)[0, :, :, i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------- #
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)), sequence_length=[64]*logits.shape[0])
decoded = tf.sparse.to_dense(decoded[0]).numpy()
print([decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in decoded[:4, :]])

# ------------------------- #

# --------- #
# DEBUGGING #
# --------- #
data_gen = generator(input_size=input_shape[0], batch_size=1, min_img_box_size=30)
x_batch = data_gen.__next__()
stride = 4
shape = tuple([int(i / stride) for i in x_batch['images'].shape[1:3]])
# shape = x_batch['images'].shape[1:3]
dummy_input = cv2.resize(x_batch['images'][0, :, :, :].copy(), shape)[np.newaxis, :, :, :]
RoIrotate_test = RoIRotate(features_stride=stride)

import copy
temp_ = [copy.deepcopy(x_batch['rboxes'][0])]
# for l, j in zip(temp_[0][0], temp_[0][1]):
#     m1 = np.amax([60 / l[-2], 1])
#     m2 = np.amax([35 / l[-1], 1])
#     l[-1] = int(l[-1] * m1)
#     l[-2] = int(l[-2] * m2)
#     j[2:4] = l[2:4]

features, ws = RoIrotate_test(dummy_input, temp_)

# cv2.imshow('org', x_batch['images'][0, :, :, :].astype(np.uint8))
# cv2.imshow('scl', dummy_input[0, :, :, :].astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in range(features.shape[0]):
    cv2.imshow('org', x_batch['images'][0, :, :, :].astype(np.uint8))
    cv2.imshow('scl', dummy_input[0, :, :, :].astype(np.uint8))
    cv2.imshow(str(i), (features[i, :, :, :].numpy()).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


