import argparse, cv2, utils
import numpy as np
import matplotlib.pyplot as plt
import gaussian_filter
import convolution
import time

def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation of Image Processing 2020")
    
    parser.add_argument('--img_dir', 
            type=str, 
            default='../Images',
            help='The directory where the image files are located',
            required=False)
    
    return parser.parse_args()

def normalize_img(arr, min_val, max_val):
    """ min_val ~ max_val normalization """
    min_arr = np.amin(arr)
    max_arr = np.amax(arr)
    if min_arr != max_arr:
        arr = (arr - min_arr) * max_val / (max_arr - min_arr) + min_val
    return arr

def clip_img(arr, min_val, max_val):
    """ min_val ~ max_val clipping """
    arr = np.where(arr < min_val, min_val, arr)
    arr = np.where(arr > max_val, max_val, arr)
    return arr


def laplace_conv():
    """ week3 assignment 2: laplacian image sharpening filter convolution """
    
    args = get_arguments()
    ip = utils.ImageProcessing(args)
    
    # load a image
    image = ip.get_one_image() # (232, 230, 3)
    cv2.imshow('original', image)
    # convert BGR to YCbCr
    img_cvt = ip.cvtYCrCb(image) # (232, 230, 3)
    # get Y
    old_Y_org = img_cvt[:, :, 0] # (232, 230)
    # because image processing affects the original image, img_cvt
    old_Y = old_Y_org.copy()

    # make a kernel
    laplacian_kernel = [[1.0,  1.0,  1.0],
                        [1.0, -8.0,  1.0],
                        [1.0,  1.0,  1.0]]

    # convolution performed with the kernel
    # cv2conv returns an output shape of 2 dimension
    filtered_Y = convolution.cv2conv(old_Y, laplacian_kernel)
    # npconv returns an output shape of 3 dimension
    #filtered_Y = filtered_Y[:, :, 0]

    # multiplication of constant c
    for c_val in range(1, 6):
        c_val *= -1
        # adding filtered_Y and old_Y
        new_Y = old_Y + c_val * filtered_Y
        # clipping
        new_Y = clip_img(new_Y, 0.0, 255.0)
        # type conversion to uint8
        new_Y = new_Y.astype(np.uint8)

        # change the image to a filtered one
        img_cvt[:, :, 0] = new_Y
        image = ip.getBGR(img_cvt)
        img_title = f'filtered; c={c_val}'
        cv2.imshow(img_title, image)

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
    if True or key == ord('q'):
        return


if __name__ == '__main__':
    laplace_conv()
