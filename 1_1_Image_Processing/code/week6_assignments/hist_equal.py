import cv2
import numpy as np
from img_preprocess import ImgObj

# arr: any int type, out: uint8
class HistEqual:
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
    img_obj = ImgObj("../Images/0010_Low-contrast-image.png")
    # hist_equal = HistEqual("../Images/0010_Low-contrast-image.png")
    cv2.imshow('original', img_obj.getY())
    hist_equal = HistEqual()
    out = hist_equal.transform_2d(img_obj.getY())
    cv2.imshow('transformed', out)

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
