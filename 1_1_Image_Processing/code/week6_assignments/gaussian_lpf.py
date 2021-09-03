""" FFT image and apply low pass filter(Gaussian) in Freq domain """

import numpy as np
import cv2
from img_preprocess import ImgObj
from fft_module import FFTModule
from fft_module import fft_image


def gaussian_2d(k_size=(3, 3), sigma=1.0):
    """ Gaussian filter generation; max value 1 """
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
    kernel = np.exp(-(h_dist + w_dist) / (2 * sigma**2))
    return kernel


if __name__ == '__main__':
    # FFT of image
    img_obj = ImgObj("../Images/small_chocolate.png")
    Fxy = fft_image(img_obj)

    # create Gaussian filter of freq domain
    P, Q = Fxy.shape
    Hxy = gaussian_2d(Fxy.shape, min(P, Q)/16)
    cv2.imshow('H(u,v)', Hxy)

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

    print("press any key to proceed or press 'q' to quit")
    while (key := cv2.waitKey(0)) != ord('q'):
        print("press any key to proceed or press 'q' to quit")
    exit(0)
