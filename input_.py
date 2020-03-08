"""
@author: MSUser
"""
import os,sys
import math
import numpy as np
import tensorflow as tf
import sonnet as snt
import scipy.io
import skimage.io
import matlab.engine as engine

from utils import draw_image

class Input(snt.AbstractModule):
    def __init__(self, batch_size, data_shape=None, num_epochs=-1, name='input'):
        """
        Args:
            batch_size: number of tfrecords to dequeue
            data_shape: the expected shape of series of images
            num_enqueuing_threads: enqueuing threads
        """
        super(Input, self).__init__(name=name)
        self._batch_size = batch_size
        self._data_shape = data_shape
        self._num_epochs = num_epochs
        if not data_shape:
            raise AssertionError('invalid data shape')

    def _parse_function(self, example):
        dims = np.prod(self._data_shape)
        features = {'image': tf.FixedLenFeature([dims], dtype=(tf.float32))}
        example_parsed = tf.parse_single_example(serialized=example, features=features)
        image = tf.reshape(example_parsed['image'], self._data_shape)
        return image

    def _build(self, filenames):
        """
        Retrieve tfrecord from files and prepare for batching dequeue
        Args:
            filenames: 
        Returns:
            wave label in batch
        """

        if type(filenames) == list:
            if not os.path.isfile(filenames[0]):
                raise AssertionError(('invalid file path: {}').format(filenames[0]))

            dataset = tf.data.TFRecordDataset(filenames)
        else:
            if type(filenames) == str:
                if not os.path.isfile(filenames):
                    raise AssertionError(('invalid file path: {}').format(filenames))

                dataset = tf.data.TFRecordDataset([filenames])
            else:
                raise ValueError(('wrong type {}').format(type(filenames)))

        dataset = dataset.map(self._parse_function)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat(self._num_epochs)

        iterator = dataset.make_one_shot_iterator()
        images = iterator.get_next()

        return images


class Watermark(snt.AbstractModule):
    def __init__(self, wm_path, name='watermark'):
        super(Watermark, self).__init__(name=name)

        eng = engine.start_matlab('-nojvm')
        self._image = eng.load(wm_path)['watermark']
        eng.quit()

    def _build(self):
        outputs = tf.constant(np.array(self._image), tf.float32)
        outputs = tf.expand_dims(outputs, axis=0)

        return outputs


class Image(snt.AbstractModule):
    def __init__(self, image_path_data, image_seq, name='image'):
        super(Image, self).__init__(name=name)

        if not os.path.isfile(image_path_data):
            raise AssertionError(('non-existing image file {}').format(image_path))

        eng = engine.start_matlab('-nojvm')
        content = eng.load(image_path_data)
        image_paths = content['image_paths']
        if not len(image_paths) > image_seq:
            raise AssertionError('Invalid imamge sequence number')
        self._image = eng.read_image(image_paths[image_seq])
        eng.quit()

    def _build(self):
        outputs = tf.constant(np.array(self._image), tf.float32)
        outputs = tf.expand_dims(outputs, axis=0)

        return outputs
        
    
class Enhance(snt.AbstractModule):   
    def __init__(self, sharpen = False, name = 'enhance'):
        super(Enhance, self).__init__(name = name)
        self._sharpen = sharpen
        
    def _build(self, inputs):       
        min_elem = tf.reduce_min(inputs)
        max_elem = tf.reduce_max(inputs)
        outputs = (inputs - min_elem) / (max_elem - min_elem)
        
        if self._sharpen:
            outputs = tf.square(outputs)
        outputs = outputs * 255.0
        
        return outputs


def test_input():
    from config import FLAGS
    
    input_ = Input(FLAGS.train_batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])
    images = input_('/data/yuming/watermark-data/train_images.tfr')
    
    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        images_val = sess.run(images)
        print(images_val.shape)
        
        images = [{'data':np.squeeze(images_val[0, :, :, :].astype(np.uint8)), 'title':'image0'},
                  {'data':np.squeeze(images_val[1, :, :, :].astype(np.uint8)), 'title':'image1'},
                  {'data':np.squeeze(images_val[2, :, :, :].astype(np.uint8)), 'title':'image2'},
                  {'data':np.squeeze(images_val[3, :, :, :].astype(np.uint8)), 'title':'image3'}]
        image_str = draw_image(images)
        
        writer.add_summary(image_str, global_step=0)
    writer.close()


def test_wm():
    wm = Watermark('/data/yuming/watermark-data/watermark.mat')()
    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        wm_val = sess.run(wm)
        np.set_printoptions(threshold=(np.nan))
        print(wm_val.shape)
        images = [
         {'data':np.squeeze(wm_val[0, :, :, :].astype(np.uint8)), 'title':'watermark'}]
        image_tensor = draw_image(images)
        image_str = draw_image(images)
        writer.add_summary(image_str, global_step=0)
    writer.close()


def test_image():
    image = Image('/data/yuming/watermark-data/image_paths.mat', 1)()
    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        image_val = sess.run(image)
        np.set_printoptions(threshold=(np.nan))
        print(image_val.shape)
        images = [{'data':np.squeeze(image_val[0, :, :, :].astype(np.uint8)), 'title':'test image'}]
        image_str = draw_image(images)
        writer.add_summary(image_str, global_step=0)
    writer.close()


def test_enhance():
    original_image = Image('/data/yuming/watermark-data/image_paths.mat', 10)()
    
    enhance = Enhance(sharpen = True)
    enhanced_image = enhance(original_image)

    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        original_image_val, enhanced_image_val = sess.run([original_image, enhanced_image])

        images = [{'data':np.squeeze(original_image_val[0, :, :, :].astype(np.uint8)), 'title': 'original image'},
                  {'data':np.squeeze(enhanced_image_val[0, :, :, :].astype(np.uint8)), 'title': 'clipped image'}]
        image_str = draw_image(images)
        writer.add_summary(image_str, global_step=0)
    writer.close()
    

if __name__ == '__main__':
    test_image()
    # test_enhance()
    
    
