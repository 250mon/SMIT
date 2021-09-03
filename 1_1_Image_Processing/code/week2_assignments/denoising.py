import argparse, cv2, utils
import numpy as np
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation of Image Processing 2020")
    
    parser.add_argument('--img_dir', 
            type=str, 
            default='../Images',
            help='The directory where the image files are located',
            required=False)
    
    parser.add_argument('--std', 
            type=int, 
            default=10.0,
            help='Noise standard deviation',
            required=False)

    return parser.parse_args()



def denoising():
    """ week2 assignment 1: denoising and variance """

    args = get_arguments()
    ip = utils.ImageProcessing(args)
    
    # load a image
    image = ip.get_one_image() # 232x230x3
    #cv2.imshow('original', image)

    est_stds = []
    K_values = [10, 30, 50]
    # make images list which contain 50 pieces of image with noise
    for k_val in K_values:
        # images list which contain 50 pieces of image with noise
        images = []
        for idx in range(k_val):
            # generate a noise image
            noise = np.random.normal(loc=0.0, scale=args.std, size=np.shape(image))
            images.append(np.add(image, noise))

        # denoising
        denoised = np.sum(np.array(images), axis=0) / float(k_val)

        # noise in denoised image
        est_noise = np.subtract(denoised, image)
        # estimated std of the noise
        est_std = np.std(est_noise)
        print(f'K = {k_val} and estimated std = {est_std}')
        est_stds.append(est_std)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    fig, ax = plt.subplots()
    ax.plot(K_values, est_stds)
    ax.set_title("Denoising", fontsize=24)
    ax.set_xlabel("K", fontsize=14)
    ax.set_ylabel("Estimated Std of Noise", fontsize=14)
    plt.show()
    

    
if __name__ == '__main__':
    denoising()
    # args = get_arguments()
    # ip = utils.ImageProcessing(args)
    # image = ip.get_one_image()
    # print(np.shape(image))
