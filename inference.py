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

from icdar import restore_rectangle


# init
cpkt_dir = 'checkpoints/'
load_models = True
loss_hist = []
input_shape = [640, 640, 3]
model_sharedconv = Backbone(backbone='mobilenet', input_shape=input_shape)
model_detection = Detection()
model_RoIrotate = RoIRotate()
model_recognition = Recognition(num_classes=len(CHAR_VECTOR)+1, training=True)

if load_models:
    model_sharedconv.load_weights(cpkt_dir + 'sharedconv')
    model_detection.load_weights(cpkt_dir + 'detection')
    model_recognition.load_weights(cpkt_dir + 'recognition')

# -------- #
vid = cv2.VideoCapture(0)
while True:

    _, im = vid.read()
    new_h, new_w, _ = im.shape
    max_h_w_i = np.max([new_h, new_w])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    im_padded[:new_h, :new_w, :] = im.copy()
    im = im_padded[np.newaxis, :, :, :]

# data_gen = generator(input_size=input_shape[0], batch_size=1)  # 160 / 480 / 640 / 800
# for _ in range(5):
#     x_batch = data_gen.__next__()

    # detection
    sharedconv = model_sharedconv(im.copy())  # x_batch['images'].shape
    f_score_, geo_score_ = model_detection(sharedconv)

    # -------- #
    score_map_thresh = 0.95
    f_score = f_score_[0, :, :, 0].numpy()
    geo_score = geo_score_[0, :, :, ].numpy()

    # filter out by score map
    xy_text = np.argwhere(f_score > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    if len(xy_text) > 0:

        # restore to coordinates
        text_box_restored = restore_rectangle(origin=xy_text[:, ::-1] * 4,
                                              geometry=geo_score[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2

        # filter out by average score
        # box_thresh = 0.95
        # ids = []
        # for i, box in enumerate(text_box_restored):
        #     mask = np.zeros_like(f_score_[0, :, :, :], dtype=np.uint8)
        #     mask = cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        #     id = cv2.mean(f_score_[0, :, :, :].numpy(), mask)[0]
        #     ids.append(id)
        # text_box_restored = text_box_restored[np.array(ids) > box_thresh]

        # nms
        selected_indices = tf.image.non_max_suppression(boxes=text_box_restored[:, :, 0].astype(np.float32),
                                                        scores=f_score[xy_text[:, 0], xy_text[:, 1]],
                                                        max_output_size=10,
                                                        iou_threshold=0.1)

        print(text_box_restored.shape[0], selected_indices.shape)

        if len(selected_indices) > 0:

            # ----------- #
            # recognition
            indices = xy_text[selected_indices, :]
            height = geo_score[indices[:, 0], indices[:, 1], :][:, 0:2].sum(axis=1)
            width = geo_score[indices[:, 0], indices[:, 1], :][:, 2:4].sum(axis=1)
            angle = geo_score[indices[:, 0], indices[:, 1], :][:, -1]

            text_coords = np.concatenate([indices, height.reshape(-1, 1), width.reshape(-1, 1)], axis=1).astype(np.int)
            text_crop = text_coords.copy()
            text_crop[:, 0:2] = 0
            rboxes = [[text_coords.tolist(), text_crop.tolist(), angle.tolist()]]

            features, ws = model_RoIrotate(sharedconv, rboxes)  # x_batch['rboxes']
            logits = model_recognition(features)

            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)),
                                                         sequence_length=[64] * logits.shape[0])
            decoded = tf.sparse.to_dense(decoded[0]).numpy()
            recognition = [decode_to_text(CHAR_VECTOR, [j for j in i if j != 0]) for i in decoded]
            print(recognition)

        # plot boxes
        for i, box in enumerate(text_box_restored[selected_indices, :, :]):

            im_padded = cv2.polylines(im_padded[:, :, :].copy(), [box.astype(np.int32)], True, color=(255, 255, 0), thickness=1)

            # Draw recognition results area
            if len(selected_indices) > 0:
                text_area = box.copy()
                text_area[2, 1], text_area[3, 1], text_area[0, 1], text_area[1, 1] = text_area[1, 1], text_area[0, 1], text_area[0, 1] - 15, text_area[1, 1] - 15
                im_padded = cv2.fillPoly(im_padded.copy(), [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                im_padded = cv2.putText(im_padded.copy(), recognition[i], (box.astype(np.int32)[0, 0], box.astype(np.int32)[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('a', im_padded.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
