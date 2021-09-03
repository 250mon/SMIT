import scipy.fft as sci_tx
import numpy as np
import cv2
from img_preprocess import ImgObj
from fft_module import FFTModule
import pdb


def image_transform(data, name='DFT', direction='Forward'):
    if name == 'DFT':
        if direction == 'Forward':
            res = sci_tx.fft2(data)
        elif direction == 'Backward':
            res = sci_tx.ifft2(data)
    elif name == 'DCT':
        if direction == 'Forward':
            res = sci_tx.dct(data, axis=0)
            res = sci_tx.dct(res, axis=1)
        elif direction == 'Backward':
            res = sci_tx.idct(data, axis=0)
            res = sci_tx.idct(res, axis=1)
    else:
        raise NotImplementedError('undefined transform')

    return res

# transform the image(uint8) and then inverse transform
# returns a enhanced image
def tf_enhancement(x, tf_name='DFT', param=(0.95, 1.0, 1.0)):
    tr_img = image_transform(x, tf_name, direction='Forward')
    tr_mag = np.abs(tr_img)
    tr_phase = tr_img / tr_mag  # (cos(p,q) + jsin(p,q) or sgn(p,q))

    tr_kernel = np.power(tr_mag, param[0] - 1) * np.power(np.log(np.power(tr_mag, param[2]) + 1.0), param[1])
    tr_out = tr_mag * tr_kernel * tr_phase
    res_img = np.real(image_transform(tr_out, tf_name, direction='Backward')) # * 255.0
    # pdb.set_trace()
    res_img = (np.clip(res_img, 0., 255.)).astype(np.uint8)
    return res_img

def cal_eme(enh_img):
    return 20. * np.log(np.max(enh_img) / (np.min(enh_img) + 1))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Error: a image file path is needed!")
        exit(0)
    filename = sys.argv.pop(1)
    try:
        img_obj = ImgObj(filename)
    except FileNotFoundError:
        print(f"Error: file({filename}) not found")
        # img_obj = ImgObj("../Images/darkboot.jpg")
        exit(0)

    src_img = img_obj.getY()

    transform_ = ['DFT', 'DCT']
    alpha_ = np.arange(0.5, 1.0, 0.1)
    beta_ = np.arange(0.5, 2.0, 0.3)
    lambda_ = np.arange(0.5, 2.0, 0.3)
    print(f'{alpha_} {beta_} {lambda_}')

    max_eme = 0.
    opt_tf = None
    opt_a = None
    y_max = 0
    y_min = 0

    total_iter = len(transform_) * len(alpha_) * len(beta_) * len(lambda_)
    for i in range(total_iter):
        a_div, a_ix = divmod(i, len(alpha_))
        b_div, b_ix = divmod(a_div, len(beta_))
        l_div, l_ix = divmod(b_div, len(lambda_))
        _, tf_ix = divmod(l_div, len(transform_))
        param = (alpha_[a_ix], beta_[b_ix], lambda_[l_ix])
        enh_img = tf_enhancement(src_img, transform_[tf_ix], param)

        eme = cal_eme(enh_img)
        # print(f'tf={transform_[tf_ix]} a={param} eme={eme}(max:{enh_img.max()}, min:{enh_img.min()})')
        if max_eme < eme:
            opt_tf = transform_[tf_ix]
            opt_a = param
            y_max = enh_img.max()
            y_min = enh_img.min()
            max_eme = eme

    print(f'tf={opt_tf} a={opt_a} max_eme={max_eme}(max:{y_max}, min:{y_min})')
    cv2.imshow('original', src_img)
    cv2.imshow('enhancded', tf_enhancement(src_img, opt_tf, param=opt_a))

    print("press any key to proceed or press 'q' to quit")
    key = cv2.waitKey(0)
