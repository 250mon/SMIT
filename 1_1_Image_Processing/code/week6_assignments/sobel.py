""" Apply Sobel filter in Freq domain and Spatial domain """

import numpy as np
import cv2
from img_preprocess import ImgObj
from fft_module import FFTModule
from fft_module import fft_image
import convolution
import pdb


# x is downward direction, y rightward
def get_sobel():
    gx = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
        ])
    gy = gx.T.copy()
    return gx, gy

# place the filter in the center of zero paddings of shape (H,W)
def filter_padding(filter_, size=(3, 3)):
    padded = np.zeros(size)
    # origin(upper left) index where the filter be placed
    org_x_ix = size[0] // 2 - filter_.shape[0] // 2
    org_y_ix = size[1] // 2 - filter_.shape[1] // 2
    # place the filter at the origin ix
    padded[ org_x_ix : org_x_ix + filter_.shape[0], 
            org_y_ix : org_y_ix + filter_.shape[1] ] = filter_
    return padded


if __name__ == '__main__':
    # FFT of image
    img_obj = ImgObj("../Images/small_chocolate.png")
    Fxy = fft_image(img_obj)

    # img_fft = fft.FFTModule("../Images/small_chocolate.png")
    # Fxy = img_fft.fft_2d()

    # create Sobel filter
    hx, hy = get_sobel()
    # apply zero padding to make it the size of (Hx2, Wx2)
    img_size = img_obj.get_size()
    filter_size = (img_size[0] * 2, img_size[1] * 2)
    hx_padded = filter_padding(hx, filter_size)
    hy_padded = filter_padding(hy, filter_size)
    # shifting to the center
    fft_mod = FFTModule()
    centeredx = fft_mod.shift_pi(hx_padded)
    centeredy = fft_mod.shift_pi(hy_padded)
    # fft of filter
    Hx = fft_mod.fft_2d(centeredx)
    Hy = fft_mod.fft_2d(centeredy)
    Hxy = Hx + Hy
    # to decenter the image which enables cropping later
    Hxy = fft_mod.shift_pi(Hxy)

    # apply the filter
    Gxy = np.multiply(Fxy, Hxy)
    fft_mod.imshow_complex('G(u,v)', Gxy)

    # Inverse FFT
    iffted = fft_mod.ifft_2d(Gxy)
    # shifting back to corner
    iffted_img = fft_mod.shift_pi(iffted.real)
    # normalize to imshow
    img_recon = cv2.normalize(iffted_img, None, 1.0, 0, cv2.NORM_MINMAX)
    # cv2.imshow('gp(x, y)', img_recon)
    # remove padding
    img_recon = img_obj.remove_2x_padding(img_recon)
    cv2.imshow('g(x, y)', img_recon)

    # Spatial domain
    conv_output = convolution.cv2conv(img_obj.getY(), hx + hy)
    cv2.imshow('f*h', conv_output)

    print("press any key to proceed or press 'q' to quit")
    while (key := cv2.waitKey(0)) != ord('q'):
        print("press any key to proceed or press 'q' to quit")
    exit(0)
