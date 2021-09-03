"""
Created on Sat Mar. 09 15:09:17 2019

@author: ygkim

main for mnist

"""

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import argparse, utils, time, cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
        
    parser.add_argument('--net_type', 
                        type=str, 
                        #default='Dense',
                        #default='Conv',
                        #default='VGG16',
                        #default='ResNet34',
                        default = 'PyNet',
                        help='Parameter for Network Selection', 
                        required = False)
    
    parser.add_argument('--ckpt_dir', 
                        type=str, 
                        default='./ckpt',
                        help='The directory where the checkpoint files are located', 
                        required = False)
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs',
                        help='The directory where the Training logs are located', 
                        required = False)
    
    parser.add_argument('--res_dir', 
                        type=str, 
                        default='./res',
                        help='The directory where the Training results are located', 
                        required = False)
        
    return parser.parse_args()

    
def main():
    
    args = get_arguments()
    cfg = utils.Config(args)
    
    print("---------------------------------------------------------")
    print("         Starting Zurich-Data Batch Processing Example")
    print("---------------------------------------------------------")
    
    #mnist_data = utils.MnistReader(cfg)
    #cifar = utils.Cifar100Reader(cfg)
    vgg19 = utils.NeuralLoss(cfg)
    zurich = utils.PyNetReader(cfg)
    
    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    while 1:
        batch_image, batch_label = zurich.next_batch()
        
        image = batch_image[:,:,:,:-1]
        labels = np.zeros(np.shape(image), dtype=np.uint8)
        
        for idx in range(len(image)):
            # dtype should be np.uint8
            # 1st argument 4 channels, 2nd argument 3 channels should be used
            labels[idx] = cv2.resize(batch_label[idx], np.shape(image)[1:-1], interpolation=cv2.INTER_CUBIC)
            
        _loss = vgg19.get_layer_loss(image, labels)
        
        loss = sess.run(_loss)
        print("batch generation with neural loss of {:.4f},   PRESS any key to proceed and 'q' to quit this program".format(loss))
        
        # key = utils.show_zurich(batch_image, batch_label)
        key = utils.show_zurich(batch_image, labels)
        #key = utils.show_cifar(cifar.eval_data[index*256:(index+1)*256, :, :, :], batch_label)
        
        
        if key == ord('q'):
            zurich.close()
            break

            
            
   
if __name__ == '__main__':
       
    main() 
