import cv2
import numpy as np


class HistEqual:
    def __init__(self, filename):
        self.in_image = cv2.imread(filename)
        self.in_YCrCb = self.getYCrCb(self.in_image)
        self.in_Y = self.in_YCrCb[:, :, 0].copy()

    # convert BGR to YCbCr
    def getYCrCb(self, image):
        YCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return YCbCr

    # convert YCbCr to BGR
    def getBGR(self, image):
        bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        return bgr

    def getY(self):
        return self.in_Y

    def _get_tfvec(self, arr):
        arr_1d = np.concatenate(arr, axis=0)
        # obtain histogram
        hist = np.bincount(arr_1d) / len(arr_1d)
        # return histogram eqaulization tf vector
        return (255. * np.cumsum(hist)).astype(np.uint8)

    def transform_2d(self, arr):
        tf_vec = self._get_tfvec(arr)
        out = np.take(tf_vec, arr)
        return out


if __name__ == '__main__':
    hist_equal = HistEqual("../Images/0010_Low-contrast-image.png")
    cv2.imshow('original', hist_equal.getY())
    out = hist_equal.transform_2d(hist_equal.getY())
    # import pdb
    # pdb.set_trace()
    cv2.imshow('transformed', out)

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
