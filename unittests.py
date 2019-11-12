"""

"""

import tensorflow as tf
import numpy as np
# from synthtext import generator
from icdar import generator
from config import CHAR_VECTOR
from model_backbone import Backbone
from model_detection import Detection
from model_roirotate import RoIRotate, tfa_enabled
from model_recognition import Recognition

# init
input_shape = [640, 640, 3]  # 160 / 480 / 640 / 800
model_sharedconv = Backbone(backbone='mobilenet', input_shape=input_shape)
model_detection = Detection()
model_RoIrotate = RoIRotate(tfa_enabled=tfa_enabled)
model_recognition = Recognition(num_classes=len(CHAR_VECTOR)+1, training=True, drop_prob=0.05)

# -------- #

data_gen = generator(input_size=input_shape[0],
                     batch_size=1,
                     min_text_size=8,
                     random_scale=np.array([1.0])
                     )
x_batch = data_gen.__next__()

for loss_id in range(4):
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

        if loss_id == 0:
            grads = tape.gradient(loss_cls, model_sharedconv.trainable_variables + model_detection.trainable_variables + model_recognition.trainable_variables)
            grad_stats = [i.numpy().sum() if i is not None else i for i in grads]
            print(sum([i is not None for i in grad_stats]), grad_stats)

        if loss_id == 1:
            grads = tape.gradient(loss_iou, model_sharedconv.trainable_variables + model_detection.trainable_variables + model_recognition.trainable_variables)
            grad_stats = [i.numpy().sum() if i is not None else i for i in grads]
            print(sum([i is not None for i in grad_stats]), grad_stats)

        if loss_id == 2:
            grads = tape.gradient(loss_angle, model_sharedconv.trainable_variables + model_detection.trainable_variables + model_recognition.trainable_variables)
            grad_stats = [i.numpy().sum() if i is not None else i for i in grads]
            print(sum([i is not None for i in grad_stats]), grad_stats)

        if loss_id == 3:
            grads = tape.gradient(loss_recongition, model_sharedconv.trainable_variables + model_detection.trainable_variables + model_recognition.trainable_variables)
            grad_stats = [i.numpy().sum() if i is not None else i for i in grads]
            print(sum([i is not None for i in grad_stats]), grad_stats)
