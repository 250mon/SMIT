""" Homomorphic filter  """

import numpy as np
import cv2
from img_preprocess import LogImgObj
from fft_module import FFTModule
from fft_module import fft_image
from hist_equal import HistEqual
import pdb


def homomorphic_2d(k_size=(3, 3), cutoff=80.0, gammaL=0.5, gammaH=2.0, power_c=1.0):
    """ Modification of Gaussian filter """
    k_height, k_width = k_size
    # Make a row for width
    # (3,) [-1, 0, 1] **2 -> [1, 0, 1]
    dist_w = np.power((np.arange(k_width) - (k_width-1)/2), 2)
    # (1x3) [[1, 0, 1]]
    dist_w = dist_w.reshape(1, -1)
    # Make a column for hight
    # (3,) [-1, 0, 1] **2 -> [1, 0, 1]
    dist_h = np.power((np.arange(k_height) - (k_height-1)/2), 2)
    # (3x1) [[1],[0],[1]]
    dist_h = dist_h.reshape(-1, 1)
    # Make a kernel size matrix for width
    w_dist = [dist_w] * k_height
    w_dist = np.concatenate(w_dist, axis=0)    # (3x3)
    # Make a kernel size matrix for width
    h_dist = [dist_h] * k_width
    h_dist = np.concatenate(h_dist, axis=1)    # (3x3)

    # Make a Gaussian kernel
    kernel = (gammaH-gammaL) * (1-np.exp(-power_c*(h_dist+w_dist)/(2*cutoff**2))) + gammaL
    return kernel


if __name__ == '__main__':
    # FFT of image
    img_obj = LogImgObj("../Images/backlight.png")
    # pdb.set_trace()
    Fxy = fft_image(img_obj)

    # create homomorphic filter of freq domain
    P, Q = Fxy.shape
    Hxy = homomorphic_2d(Fxy.shape, gammaL=0.25, gammaH=4, power_c=0.1)
    # cv2.imshow('H(u,v)', Hxy)

    fft_mod = FFTModule()
    # apply the filter in freq domain
    Gxy = np.multiply(Fxy, Hxy)
    fft_mod.imshow_complex('G(u,v)', Gxy)

    # Inverse FFT
    iffted = fft_mod.ifft_2d(Gxy)
    # shifting back to corner
    iffted_img = fft_mod.shift_pi(iffted.real)
    # normalize to imshow
    img_recon = cv2.normalize(iffted_img, None, 1.0, 0, cv2.NORM_MINMAX)
    cv2.imshow('gp(x, y)', img_recon)

    # remove padding
    img_recon = img_obj.remove_2x_padding(img_recon)
    cv2.imshow('g(x, y)', img_recon)

    hist_equal = HistEqual()
    out = hist_equal.transform_2d((img_recon * 255).astype(np.int32))
    cv2.imshow('hist equalized', out)


    print("press any key to proceed or press 'q' to quit")
    while (key := cv2.waitKey(0)) != ord('q'):
        print("press any key to proceed or press 'q' to quit")
    exit(0)
