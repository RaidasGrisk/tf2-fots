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

"""


import tensorflow as tf
import cv2
import numpy as np
from icdar import generator
from config import CHAR_VECTOR
from model_backbone import Backbone
from model_detection import Detection
from model_roirotate import RoIRotate
from model_recognition import Recognition

# init
cpkt_dir = 'checkpoints/'
load_models = False
model_sharedconv = Backbone(backbone='mobilenet')
model_detection = Detection()
model_RoIrotate = RoIRotate(features_stride=1)
model_recognition = Recognition(num_classes=len(CHAR_VECTOR)+1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5)

if load_models:
    model_sharedconv.load_weights(cpkt_dir + 'sharedconv')
    model_detection.load_weights(cpkt_dir + 'detection')
    model_recognition.load_weights(cpkt_dir + 'recognition')

# -------- #
max_iter = 10000
iter = 0
data_gen = generator(input_size=640, batch_size=1)  # 160 / 480 / 640 / 800
for x_batch in data_gen:

    features, ws = model_RoIrotate(x_batch['images'], x_batch['rboxes'])
    img_copy = x_batch['images'].copy()

    _, _, img_shape, _ = x_batch['images'].shape
    for i in range(np.clip(features.shape[0], a_max=15, a_min=0)):
        # print(i, x_batch['images'][0, img_shape - 32*(i+1): img_shape - 32*(i+1-1), img_shape-256: img_shape, :].shape)
        img_copy[0, img_shape - 32*(i+1): img_shape - 32*(i+1-1), img_shape-256: img_shape, :] = features[i, :, :, :].numpy()

    # cv2.imshow('org', img_copy[0, :, :, :].astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(x_batch['image_fns'], len(x_batch['rboxes'][0][0]))
    cv2.imwrite('debug/' + x_batch['image_fns'][0].split('\\')[-1], img_copy[0, :, :, :])

    iter += 1
    if iter == max_iter:
        break
