from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import argparse, cv2
import tensorflow as tf
import numpy as np

def arg_process():
    
    parser = argparse.ArgumentParser('Implementation of VGG16 CIFAR10 Classification 2020')
    
    parser.add_argument('--pb_name', 
                        type=str, 
                        default='./VGG16-CIFAR10-201007.pb',
                        help='Protocol Buffer File Name', 
                        required = False)
    
    parser.add_argument('--data_name', 
                        type=str, 
                        default='../CIFAR/CIFAR-10/test_batch',
                        help='Test Data Name', 
                        required = False)
    
    args, unkowns = parser.parse_known_args()
    
    return args, unkowns


label={0:'airplane',
       1:'automobile',
       2:'bird',
       3:'cat',
       4:'deer',
       5:'dog',
       6:'frog',
       7:'horse',
       8:'ship',
       9:'truck'}

def load_graph(file):
    
    with tf.gfile.GFile(file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # fix nodes    
    for node in graph_def.node:
        
        #if node.op == 'RefSwitch':
            #print(node.name, ' -- ', node.input)
        if node.op == 'RefSwitch':
          node.op = 'Switch'
          for index in range(len(node.input)):
            if 'count' in node.input[index]:
              node.input[index] = node.input[index] + '/read'
              #print(node.input[index])
            elif 'sh_x' in node.input[index]:
              node.input[index] = node.input[index] + '/read'
              #print(node.input[index])
            elif 'sh_x2' in node.input[index]:
              node.input[index] = node.input[index] + '/read'
              #print(node.input[index])
        elif node.op == 'AssignSub':
          node.op = 'Sub'
          if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
          node.op = 'Add'
          if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
          node.op = 'Identity'
          if 'use_locking' in node.attr: del node.attr['use_locking']
          if 'validate_shape' in node.attr: del node.attr['validate_shape']
          if len(node.input) == 2:
            # input0: ref: Should be from a Variable node. May be uninitialized.
            # input1: value: The value to be assigned to the variable.
            node.input[0] = node.input[1]
            del node.input[1]    
        
    _ = tf.graph_util.import_graph_def(graph_def)
    
    graph = tf.get_default_graph()
        
    return graph


def load_image(file):
    
    def _unpickle(file):
        import pickle
        with open(file, 'rb') as f:
            dict = pickle.load(f, encoding='latin1')
        return dict
    
    test = _unpickle(file)
    
    return np.moveaxis(np.reshape(test['data'], (-1, 3, 32, 32)), 1, -1)

    
def main(FLAGS):
    
    sess = tf.Session()
    graph = load_graph(FLAGS.pb_name)
        
    _output = graph.get_tensor_by_name("import/output:0") # <op_name>:<output_index> for tensor
    _input = graph.get_tensor_by_name("import/input:0") # <op_name>:<output_index> for tensor
    _btrain = graph.get_tensor_by_name("import/btrain:0")
    _breset = graph.get_tensor_by_name("import/breset:0")
    _buse_drop = graph.get_tensor_by_name("import/buse_drop:0")
    
    data = load_image(FLAGS.data_name)
    size = len(data) - 1
    
    cv2.namedWindow('Batch Data', cv2.WINDOW_NORMAL)
    
    while 1:
        
        idx = np.random.randint(0, size)
        
        img = data[idx]
        #print('image shape', np.shape(img))
        #print('reshaped image', np.shape(np.reshape(img, (1,)+np.shape(img))))
        feed_dict = {_input:np.reshape(img, (1,)+np.shape(img)), _btrain: False, _breset: False, _buse_drop: False}
        result = label[int(sess.run(_output, feed_dict=feed_dict))]
        
        print("--------------------------------------------------------------")
        if result[0] == 'a':
            print("            This is an - ", result)
        else:
            print("            This is a - ", result)
        print("--------------------------------------------------------------")
        print("\n press any key to continue or press 'q' to quit")
        
        cv2.imshow('Batch Data', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        
    
    sess.close()
    
    return 0

   
if __name__ == '__main__':
    
    FLAGS, unparsed = arg_process()
    
    main(FLAGS) 
