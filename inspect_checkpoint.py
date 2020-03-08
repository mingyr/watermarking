import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from tensorflow.python import pywrap_tensorflow
from scipy import io

import string

flags.DEFINE_string('file_path', '', 'path of checkpoint file.')
flags.DEFINE_string('output_dir', '', 'path of output directory')
flags.DEFINE_string('tensor_name', '', 'fold number')
flags.DEFINE_boolean('print', False, 'whether or not save filter')
flags.DEFINE_boolean('save', False, 'whether or not save filter')
flags.DEFINE_boolean('analyze', False, 'whether or not analyze filter')

FLAGS = flags.FLAGS

def main(unused_argv):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shape in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
        file_path: Name of the checkpoint file.
        tensor_name: Name of the tensor in the checkpoint file to print.
        all_tensors: Boolean indicating whether to print all tensors.
    """
    from os import path, makedirs

    if not path.exists(FLAGS.file_path):
        print("non-existing path: {}".format(FLAGS.file_path))

    '''
    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not path.exists(FLAGS.output_dir):
        makedirs(FLAGS.output_dir)
    '''

    if path.isdir(FLAGS.file_path):
        glob_path = path.join(FLAGS.file_path, "*.meta")
        file_name_lst = glob(glob_path)
        if len(file_name_lst) == 0:
            print("cannot find checkpoint file under directory {}".format(FLAGS.file_path))
            return False
        elif len(file_name_lst) > 1:
            file_name_lst.sort(key = os.path.getmtime, reverse = True)
            file_name = file_name_lst[0]
        else:
            file_name = file_name_lst[0]

        file_name = path.splitext(file_name)[0]
    else:
        file_name = FLAGS.file_path 
   
    print("checkpoint file {} will be used".format(file_name))
     
    tensor_name_lst = FLAGS.tensor_name.split(",")
    all_tensors = True
    for name in tensor_name_lst:
        all_tensors = all_tensors and name == ''

    np.set_printoptions(threshold = np.nan)
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor name: {}".format(key))
                tensor_val = reader.get_tensor(key)
                print("tensor shape: {}".format(tensor_val.shape))
                # print("tensor value: {}".format(tensor_val))
        else:
            for tensor_name in tensor_name_lst:
                print("tensor name: {}".format(tensor_name))
                tensor_val = reader.get_tensor(tensor_name)
                print("tensor shape: {}".format(tensor_val.shape))

                if FLAGS.print:
                    print("tensor value: {}".format(tensor_val))
 
                if FLAGS.save:
                    if not os.path.exists(FLAGS.output_dir):
                        os.makedirs(FLAGS.output_dir)

                    name = tensor_name.replace('/', '_')
                    filename = os.path.join(FLAGS.output_dir, "{}.mat".format(name))
                    io.savemat(filename, {name: tensor_val})

                if FLAGS.analyze:
                    assert (np.ndim(tensor_val) == 2)

                    import matlab.engine
                    eng = matlab.engine.start_matlab("-nojvm")

                    X = matlab.double(tensor_val.tolist())
                    coeff = eng.pca(X, 'NumComponents', 3)

                    coeff = np.array(coeff)
                    print("shape of original principal component coefficients: {}".format(coeff.shape))

                    coeff = coeff[:, range(3)]
                    print("shape of reduced principal component coefficients: {}".format(coeff.shape))

                    P = np.matmul(X, coeff)
                    print("shape of reduced principal component coefficients: {}".format(P.shape))

                    min_elem = tf.reduce_min(tf.reduce_min(P))
                    max_elem = tf.reduce_max(tf.reduce_max(P))
                    output = (P - tf.reshape(min_elem, [1, 1])) / tf.reshape((max_elem - min_elem), [1, 1])

                    size = eng.sqrt(float(int(output.shape[0])))

                    size = int(size)
                    chans = int(output.shape[1])
                    print("dimension of the image: {}x{}x{}".format(size, size, chans))

                    eng.quit()
                    
                    img = tf.reshape(output, [-1, size, size, chans])
                    
                    add_img_op = tf.summary.image("weights-image-of-{}".format("-".join(tensor_name.split('/'))), img)

                    with tf.Session() as sess:

                        writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)
                        
                        sess.run(tf.global_variables_initializer())

                        summ_img = sess.run(add_img_op)
                        writer.add_summary(summ_img, 0)

                        writer.close()
                         
                # print("tensor value: {}".format(tensor_val))

    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")
        if "Data loss" in str(e) and (any([e in file_name for e in [".index", ".meta", ".data"]])):
            print("It's likely that this is a V2 checkpoint and you need to provide the filename prefix*. "
                  "Try removing the '.' and extension.")

    
if __name__ == "__main__":
    app.run()
