import tensorflow as tf
import numpy as np
import scipy
import cv2
from utils import quick_plot
import math


class RoIRotate(object):

    """
    The basic idea begind this branch is to output croped text regions from sharedconvs
    these cropped regions (boxes) will serve as an input into recongition branch

    For example:
    input   [1, 120, 160, 32] sharedconvs
    output  [7, 8, 64, 32] where [boxes, height, width, channels]

    https://github.com/yu20103983/FOTS/blob/master/FOTS/dataset/dataReader.py
    https://stackoverflow.com/questions/55160136/tensorflow-2-0-and-image-processing
    https://github.com/tensorflow/addons

    https://databricks.com/tensorflow/tensorflow-in-3d
    https://stackoverflow.com/questions/37042748/how-to-create-a-rotation-matrix-in-tensorflow

    """

    # reshape original image to shape of sharedconvs and roi rotate it to see what are the features

    def __init__(self, features_stride=4):
        # self.features = features
        self.features_stride = features_stride
        self.max_RoiWidth = int(256 / features_stride)
        self.fix_RoiHeight = int(32 / features_stride)
        self.ratio = float(self.fix_RoiHeight) / self.max_RoiWidth

    def scanFunc(self, state, b_input, plot=False, expand_px=0):

        ifeatures, outBox, cropBox, angle = b_input
        cropFeatures = tf.image.crop_to_bounding_box(ifeatures, outBox[1]-expand_px, outBox[0]-expand_px, outBox[3]+expand_px*2, outBox[2]+expand_px*2)
        if plot:
            for i in cropFeatures:
                quick_plot(i.numpy())
        # cropFeatures.shape
        # plot(cropFeatures.numpy()[0, ::])
        # rotateCropedFeatures = scipy.ndimage.rotate(cropFeatures, angle*55, axes=(1, 2))
        # rotateCropedFeatures.shape
        # plot(rotateCropedFeatures[0, ::])
        # textImgFeatures = tf.image.crop_to_bounding_box(rotateCropedFeatures, cropBox[1], cropBox[0], cropBox[3], cropBox[2])
        # textImgFeatures.shape
        # plot(textImgFeatures.numpy()[0, ::])

        # ------------- #
        _, h, w, c = cropFeatures.shape
        center = (w/2+expand_px, h/2+expand_px)
        width = cropBox[2]+expand_px*4
        height = cropBox[3]+expand_px*4
        matrix = cv2.getRotationMatrix2D(center=center, angle=math.degrees(math.atan(angle)), scale=1) # https://stackoverflow.com/questions/10057854/inverse-of-tan-in-python-tan-1
        image = cv2.warpAffine(src=cropFeatures.numpy()[0, :, :, :], M=matrix, dsize=(w, h))
        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)
        textImgFeatures = image[y:y + height, x:x + width, :][np.newaxis, :, :, :]
        if plot:
            for i in textImgFeatures:
                quick_plot(i)
        # plot(image)

        # ------------- #

        # resize keep ratio
        w = tf.cast(tf.math.ceil(tf.multiply(tf.divide(self.fix_RoiHeight, cropBox[3]+expand_px*4), tf.cast(cropBox[2]+expand_px*4, tf.float64))), tf.int32)
        resize_textImgFeatures = tf.image.resize(textImgFeatures, (self.fix_RoiHeight, w))
        if plot:
            for i in resize_textImgFeatures:
                quick_plot(i.numpy())
        w = tf.minimum(w, self.max_RoiWidth)

        # crop rotated corners
        pad_or_crop_textImgFeatures = tf.image.crop_to_bounding_box(resize_textImgFeatures, 0, 0, self.fix_RoiHeight, w)
        # plot(pad_or_crop_textImgFeatures[0, ::].numpy())
        pad_or_crop_textImgFeatures = tf.image.pad_to_bounding_box(pad_or_crop_textImgFeatures, 0, 0, self.fix_RoiHeight, self.max_RoiWidth)
        if plot:
            for i in pad_or_crop_textImgFeatures:
                quick_plot(i.numpy())

        return [pad_or_crop_textImgFeatures, w]

    def __call__(self, features, brboxes, expand_w=20, plot=False, expand_px=0):

        # features = x_batch['images']
        # brboxes = x_batch['rboxes']

        # why do we need to pad this?? What is the point?
        paddings = tf.constant([[0, 0], [expand_w, expand_w], [expand_w, expand_w], [0, 0]])
        features_pad = tf.pad(features, paddings, "CONSTANT")
        features_pad = tf.expand_dims(features_pad, axis=1)  # [b, 1, h, w, c]

        btextImgFeatures = []
        ws = []
        # loop over images in batch
        for b, rboxes in enumerate(brboxes):

            outBoxes, cropBoxes, angles = rboxes
            outBoxes = np.array(outBoxes).astype(np.int)
            cropBoxes = np.array(cropBoxes).astype(np.int)
            angles = np.array(angles).astype(np.float)

            # not sure if all is good, maybe +1 width and +1 height????
            outBoxes = tf.cast(tf.math.divide(outBoxes, self.features_stride), tf.int32)
            cropBoxes = tf.cast(tf.math.divide(cropBoxes, self.features_stride), tf.int32)

            outBoxes_xy = outBoxes[:, :2]
            outBoxes_xy = tf.add(outBoxes_xy, expand_w)
            outBoxes = tf.concat([outBoxes_xy, outBoxes[:, 2:]], axis=1)

            ifeatures_pad = features_pad[b]

            # for every box
            croped_ft = []
            croped_ft_w = []
            for outB, cropB, ang in zip(outBoxes, cropBoxes, angles):

                out = self.scanFunc(b_input=(ifeatures_pad, outB, cropB, ang), state=[], plot=plot, expand_px=expand_px)
                croped_ft.append(out[0])
                croped_ft_w.append(out[1])

            textImgFeatures = [tf.concat(croped_ft, axis=0), croped_ft_w]

            # the below code produces OOM because if there are many boxes,
            # tf, tile will produce a massive matrix hence for now the above solution
            # len_crop = tf.shape(outBoxes)[0]
            # ifeatures_tile = tf.tile(ifeatures_pad, [len_crop, 1, 1, 1])  # repeat matrix on 1 axis (for each box)
            # textImgFeatures = tf.scan(self.scanFunc, [ifeatures_tile, outBoxes, cropBoxes, angles],
            #                           [np.zeros((self.fix_RoiHeight, self.max_RoiWidth, channels), np.float32),
            #                            np.array(0, np.int32)])

            btextImgFeatures.append(textImgFeatures[0])
            ws.append(textImgFeatures[1])

        btextImgFeatures = tf.concat(btextImgFeatures, axis=0)
        ws = tf.concat(ws, axis=0)

        return btextImgFeatures, ws
