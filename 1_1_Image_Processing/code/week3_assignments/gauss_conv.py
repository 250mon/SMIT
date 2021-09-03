import argparse
import cv2
import utils
import gaussian_filter
import convolution
import time


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Implementation of Image Processing 2020")

    parser.add_argument('--img_dir',
                        type=str,
                        default='../Images',
                        help='The directory where the image files are located',
                        required=False)

    return parser.parse_args()


def gauss_conv():
    """ week3 assignment 1: gaussian filter convolution"""

    args = get_arguments()
    ip = utils.ImageProcessing(args)

    # load a image
    image = ip.get_one_image()  # (232, 230, 3)
    cv2.imshow('original_BGR', image)
    # convert BGR to YCrCb
    img_cvt = ip.cvtYCrCb(image)  # (232, 230, 3)
    cv2.imshow('original_cvt', img_cvt)
    # get Y
    old_Y = img_cvt[:, :, 0]  # (232, 230)

    # make a kernel
    ks = 9
    gauss_kernel = gaussian_filter.gaussian_2d(None, k_size=(ks, ks), sigma=max(1.0, (ks / 4)))

    # 1. For npconv
    # start a timer
    st_time = time.time()
    # convolution performed with the kernel
    filtered_Y = convolution.npconv(old_Y, gauss_kernel)
    # stop the timer
    elapsed_time_npconv = time.time() - st_time
    # apply the new Y to the image
    cv2.imshow('npconv of gaussian kernel', filtered_Y)

    # 2. For cv2conv
    # start a timer
    st_time = time.time()
    # convolution performed with the kernel
    filtered_Y = convolution.cv2conv(old_Y, gauss_kernel)
    # stop the timer
    elapsed_time_cv2conv = time.time() - st_time
    # apply the new Y to the image
    cv2.imshow('cv2conv of gaussian kernel', filtered_Y)

    print("Elapsed time...\n")
    print(f'npconv: {elapsed_time_npconv}\n')
    print(f'cv2conv: {elapsed_time_cv2conv}\n')

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
    if True or key == ord('q'):
        return


if __name__ == '__main__':
    gauss_conv()
