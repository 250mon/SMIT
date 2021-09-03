import argparse, cv2, utils
import numpy as np
import matplotlib.pyplot as plt
import gaussian_filter
import convolution
import time
import pdb


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


def unsharp():
    """ week3 assignment 2: unsharp masking and high boost filtering """
    
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

    for ks in [5, 9, 13]:
        # make a kernel
        gauss_kernel = gaussian_filter.gaussian_2d(None, k_size=(ks,ks), sigma=max(1.0, (ks/4)))

        # convolution performed with the kernel
        # cv2conv returns an output shape of 2 dimension
        filtered_Y = convolution.cv2conv(old_Y, gauss_kernel)
        # unsharp mask
        mask_Y = old_Y - filtered_Y

        # multiplication of boost k
        for k_val in [2, 5]:
            # adding filtered_Y and old_Y
            new_Y = old_Y + k_val * mask_Y
            # clipping
            new_Y = clip_img(new_Y, 0.0, 255.0)
            # type conversion to uint8
            new_Y = new_Y.astype(np.uint8)

            # change the image to a filtered one
            img_cvt[:, :, 0] = new_Y
            image = ip.getBGR(img_cvt)
            img_title = f'unsharp; kernel={ks} / k={k_val}'
            cv2.imshow(img_title, image)

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
    if True or key == ord('q'):
        return


if __name__ == '__main__':
    unsharp()
