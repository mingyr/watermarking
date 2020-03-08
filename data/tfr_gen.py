import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import matlab.engine

FLAGS = flags.FLAGS

eng = matlab.engine.start_matlab("-nojvm")

content = eng.load('image_data.mat')
# train_images = np.array(content['train_images'])
test_images = np.array(content['test_images'])

eng.quit()

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list = tf.train.FloatList(value = value))

height = 240
width = 320
channel = 3

image_placeholder = tf.placeholder(tf.uint8, [height, width, channel], 'image_placeholder')
image_tensor = tf.cast(image_placeholder, tf.float32)
image_tensor = tf.reshape(image_tensor, [-1])

train_count = 0
test_count = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    '''
    tfrecords_filename = 'train_images.tfr'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    total = train_images.shape[-1]
    for i in range(total):
        print('Processing training samples: {}/{}'.format(i, total))
        image = np.squeeze(train_images[:, :, :, i])
        values = sess.run(image_tensor, feed_dict = {image_placeholder: image})
        feature = {'image': _float_feature(values)}
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())

    writer.close()
    '''

    tfrecords_filename = 'test_images.tfr'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    total = test_images.shape[-1]
    for i in range(total):
        print('Processing testing samples: {}/{}'.format(i, total))
        image = np.squeeze(test_images[:, :, :, i])
        values = sess.run(image_tensor, feed_dict = {image_placeholder: image})
        feature = {'image': _float_feature(values)}
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())

    writer.close()


