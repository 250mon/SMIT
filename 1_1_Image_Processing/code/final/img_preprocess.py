import numpy as np
import cv2


class ImgObj:
    def __init__(self, filename):
        # load a image, e.g (232, 230, 3)
        self.in_image = cv2.imread(filename)
        # convert BGR to YCrCb, e.g  (232, 230, 3)
        self.in_YCrCb = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2YCrCb)
        # Y component only, e.g. (232, 230)
        self.in_Y = self.in_YCrCb[:, :, 0].copy()

    # return 3 dim array
    def getYCbCr(self):
        return self.in_YCrCb

    # return 2 dim array of type uint8
    def getY(self):
        return self.in_Y

    # return the size(shape) of the image
    def get_size(self):
        return self.in_Y.shape

    # return a copy which is padded into (src_width x 2, src_height x 2)
    def apply_2x_padding(self, src):
        # number of padding pixels
        top, left = 0, 0
        bottom, right = src.shape
        # border type
        border_type = cv2.BORDER_CONSTANT
        # padding
        return cv2.copyMakeBorder(src, top, bottom, left, right, border_type, value=0)

    # return a copy which is stripped of the padding
    def remove_2x_padding(self, padded):
        h, w = padded.shape
        return padded[:h//2, :w//2]


# apply ln to image
class LogImgObj(ImgObj):
    def __init__(self, filename):
        super().__init__(filename)

    def getY(self):
        log_Y = (np.log(super().getY(), dtype=np.float32))
        log_Y_norm = cv2.normalize(log_Y, None, 1.0, 0, cv2.NORM_MINMAX)
        return log_Y_norm
