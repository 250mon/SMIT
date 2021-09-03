import numpy as np
import cv2


class FFTModule:
    """ week6 assignment 1: Freqency Domain Filtering Basic """
    # return a kernel which is -1^(x+y)
    def _get_shift_kenel(self, shape):
        height, width = shape
        # -1**(x, y)
        odd_row = [(-1)**i for i in range(width)]
        even_row = [(-1)**(i+1) for i in range(width)]
        rows = [odd_row, even_row] * (height // 2)
        shift_kernel = np.array(rows, np.float32).reshape((height, width))
        return shift_kernel

    # return a copy which is shifted by pi
    # ; effect of setting the center of the ffted image on zero
    def shift_pi(self, arr):
        shift_kernel = self._get_shift_kenel(arr.shape)
        return np.multiply(arr, shift_kernel)

    def imshow_complex(self, description, complex_img):
        # take an abs of the complex numbers and normalize
        complex_img_abs = cv2.normalize(np.abs(complex_img), None, 1.0, 0, cv2.NORM_MINMAX)
        # raise the dark region to the 0.5 power
        complex_img_abs_pow = np.power(complex_img_abs, 0.5)
        cv2.imshow(description, complex_img_abs_pow)

    def fft_2d(self, arr):
        ffted = np.fft.fft2(arr)
        return ffted

    def ifft_2d(self, arr):
        iffted = np.fft.ifft2(arr)
        return iffted


def fft_image(img_obj):
    # show the original image
    cv2.imshow('f(x, y)', img_obj.getY())
    # padding 
    padded_Y = img_obj.apply_2x_padding(img_obj.getY())
    cv2.imshow('fp(x, y)', padded_Y)
    # shifting to center
    fft_mod = FFTModule()
    centered_Y = fft_mod.shift_pi(padded_Y)
    cv2.imshow('fp(x, y) * -1^(x+y)', centered_Y)
    # FFT
    Fxy = fft_mod.fft_2d(centered_Y)
    fft_mod.imshow_complex('Fp(x, y)', Fxy)
    return Fxy
