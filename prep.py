import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

flags.DEFINE_string('mode', '', 'color mode, RGB|YUV')
FLAGS = flags.FLAGS

# locate all the image files
image_path = 'images'

assert FLAGS.mode == 'rgb' or FLAGS.mode == 'yuv'

files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

# print(files)


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def decode(image_string):
    
    data = tf.cond(tf.io.is_jpeg(image_string), 
                   lambda: tf.io.decode_jpeg(image_string),
                   lambda: tf.io.decode_png(image_string))

    data = tf.cast(data, tf.float32)
    if FLAGS.mode == 'rgb':
        data = tf.identity(data)
    elif FLAGS.mode == 'yuv':
        data = tf.image.rgb_to_yuv(data / 255) * 255
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    with tf.control_dependencies([tf.assert_equal(tf.shape(data), tf.TensorShape((183, 275, 3)))]):
        data = tf.reshape(data, [-1])

    return data
    
# print(image_string)

image_string_placeholder = tf.placeholder(tf.string, [], 'image_string_placeholder')
data = decode(image_string_placeholder)

tfrecords_filename = 'images.tfr'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for f in files:
        if 'logo' in f:
            print('skipping processing {}'.format(f))
            continue
        else:
            print("processing {}".format(f))

        filename = os.path.join(image_path, f)

        with open(filename, 'rb') as f:
            image_string = f.read()
            values = sess.run(data, feed_dict = {image_string_placeholder: image_string})

        feature = {'image': _float_feature(values)}
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())

writer.close()
    

