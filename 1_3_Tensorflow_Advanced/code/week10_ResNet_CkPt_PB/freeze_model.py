
import argparse
import sys
import os

import tensorflow as tf


# generator output nodes to be saved
nodes_name=['output']

FLAGS = None
out_file = 'VGG16-CIFAR10-201007.pb'

def main(unused_argv):
        
    if not FLAGS.ckpt_dir: 
      print("Usage: python freeze_model.py --ckpt_dir checkpoint_directory ") 
      sys.exit(1) 
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    
    if ckpt:
        model = ckpt.model_checkpoint_path
    else:
        print("No Checkpoint File Found")
        sys.exit(1)
        
    sess = tf.Session()
        
    saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)
    saver.restore(sess, model)
    
    
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), nodes_name)
    
    out_path = os.path.join(os.path.split(model)[0], out_file)
    
    with tf.gfile.GFile(out_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        
    for index in range(len(output_graph_def.node)):
        print("Saved  - {} - done".format(output_graph_def.node[index].name))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        "--ckpt_dir", 
        type=str, 
        default='.', 
        help="check-point file location.",
        required = False)
    
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
