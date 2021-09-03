import numpy as np
import cv2

def npconv(image, kernel):
    """ Convolution using numpy (direct numpy implementation) """
    # kernel shape (height and width)
    ker_h, ker_w = np.shape(kernel)
    # padding size for height axis
    pad_h1, pad_h2 = ker_h // 2, (ker_h - 1) // 2
    # padding size for width axis
    pad_w1, pad_w2 = ker_w // 2, (ker_w -1) // 2
    
    if len(np.shape(image)) == 2:
        # if the image is (H, W) shape, reshape to (H, W, 1)
        image = np.reshape(image, np.shape(image) + (1,))

    img_h, img_w, img_c = np.shape(image)
    # padding
    pad_image = np.pad(image, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0,0)), mode='reflect')
    # kernel flipping (left-right and up-down)
    flip_ker = np.flipud(np.fliplr(kernel)) 
    # convolution output initialization -> image size
    out = np.zeros(np.shape(image))  

    # convolution
    # for each channel
    for id_c in range(img_c):
        # for each width
        for id_w in range(img_w):
            # for each heigbht
            for id_h in range(img_h):
                # crop kernel size padded image pixels (output location)
                out[id_h, id_w, id_c] = np.sum(flip_ker * pad_image[id_h:id_h+ker_h, id_w:id_w+ker_w, id_c])
                                    
    return out.astype(np.uint8)


def cv2conv(image, kernel):
    """ Convolution using opencv filter2D function """
    # kernel flipping (left-right and up-down)
    flip_ker = np.flipud(np.fliplr(kernel))
    # 2D filtering
    # data , offset (output position: -1 means center), flipped-kernel, border-processing
    return cv2.filter2D(image, -1, flip_ker, borderType=cv2.BORDER_REFLECT_101)
