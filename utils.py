import cv2
import numpy as np


def decode_to_text(char_dict, decoded_out):
    return ''.join([char_dict[i] for i in decoded_out])


def quick_plot(x):
    cv2.imshow('org', x.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
