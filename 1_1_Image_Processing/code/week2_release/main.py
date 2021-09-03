import argparse, cv2, utils
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation of Image Processing 2020")
    
    parser.add_argument('--img_dir', 
                        type=str, 
                        default='../Images', 
                        help='The directory where the image files are located', 
                        required = False)
    
    return parser.parse_args()



def main():
    
    args = get_arguments()
    ip = utils.ImageProcessing(args)
    
    for idx in range(ip.img_list_size):
        
        image = ip.get_one_image()
        YCrCb = ip.cvtYCrCb(image)
        
        cv2.imshow('original', image)
        cv2.imshow('gray image', YCrCb[:, :, 0])
        
        print("press any key to proceed or press 'q' to quit")
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break

    
        
    
if __name__ == '__main__':
    main()