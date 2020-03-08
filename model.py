import numpy as np
import tensorflow as tf
import sonnet as snt
from utils import Activation, Pooling

class Upsampler(snt.AbstractModule):
    def __init__(self, dim, factor = 2, filter_size = 5, num_filters = 16, act = 'relu', name = 'upsampler'):
        super(Upsampler, self).__init__(name = name)
        self._act = Activation(act = act, verbose = True)

        with self._enter_variable_scope():
            self._conv  = snt.Conv2DTranspose(num_filters, [e * factor for e in dim[1:-1]], filter_size, stride = 2, use_bias = False)

            dim2 = [dim[0], dim[1] * factor, dim[2] * factor, num_filters]
            self._conv2 = snt.Conv2DTranspose(num_filters, [e * factor for e in dim2[1:-1]], filter_size, stride = 2, use_bias = False)

            dim3 = [dim[0], dim[1] * factor * factor, dim[2] * factor * factor, num_filters]
            self._conv3 = snt.Conv2DTranspose(num_filters, [e * factor for e in dim3[1:-1]], filter_size, stride = 2, use_bias = False)

            self._seq = snt.Sequential([self._conv,  self._act,
                                        self._conv2, self._act,
                                        self._conv3, self._act,])
            
    def _build(self, inputs):
        outputs = self._seq(inputs)

        return outputs

class Downsampler(snt.AbstractModule):
    def __init__(self, dim, factor = 2, filter_size = 5, act = 'relu', pool = 'avg', name = 'downsampler'):
        super(Downsampler, self).__init__(name = name)

        self._act = Activation(act, verbose = True)
        self._pool = Pooling(pool, 2, verbose = True)
 
        with self._enter_variable_scope():
            self._conv  = snt.Conv2D(dim[-1] * factor * factor, filter_size, use_bias = False)
            self._conv2 = snt.Conv2D(dim[-1] * factor,          filter_size, use_bias = False)
            self._conv3 = snt.Conv2D(dim[-1],                   filter_size, use_bias = False)

            self._seq = snt.Sequential([self._conv,  self._pool, self._act,
                                        self._conv2, self._pool, self._act,
                                        self._conv3, self._pool, self._act])

    def _build(self, inputs, training = False):
        outputs = self._seq(inputs)

        if not training:
            outputs = tf.clip_by_value(outputs, 0.0, 255.0)

        return outputs
    
class Blender(snt.AbstractModule):
    def __init__(self, dim, factor = 2**3, name = 'blender'):
        super(Blender, self).__init__(name = name)
        self._factor = factor

        with self._enter_variable_scope():
            self._ws = tf.get_variable('ws', [1] + [dim[1] * self._factor, dim[2] * self._factor] + [1], 
                                       tf.float32, initializer = tf.constant_initializer(1.0))
            self._wl = tf.get_variable('wl', [1] + [dim[1] * self._factor, dim[2] * self._factor] + [1], 
                                       tf.float32, initializer = tf.constant_initializer(1.0))

    def _build(self, s, l):
        # with tf.control_dependencies([tf.assert_equal(tf.shape(s), tf.shape(l))]):
        outputs = s * self._ws + l * self._wl

        return outputs

class Extractor(snt.AbstractModule):
    def __init__(self, dim, factor = 2, filter_size = 5, act = 'relu', name = "extractor"):
        super(Extractor, self).__init__(name = name)
        self._act = Activation(act, verbose = True)

        with self._enter_variable_scope():
            self._conv  = snt.Conv2D(dim[-1] * factor * factor, filter_size, use_bias = False)
            self._conv2 = snt.Conv2D(dim[-1] * factor,          filter_size, use_bias = False)
            self._conv3 = snt.Conv2D(dim[-1],                   filter_size, use_bias = False)

            self._seq = snt.Sequential([self._conv,  self._act,
                                        self._conv2, self._act,
                                        self._conv3, self._act])

    def _build(self, inputs, training = False):
        outputs = self._seq(inputs)
        if not training:
            outputs = tf.clip_by_value(outputs, 0.0, 255.0)

        return outputs
            
def test_upsampler():
    from config import FLAGS

    s = tf.constant(1.0, tf.float32, [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])

    upsampler = Upsampler(s.get_shape().as_list())

    t = upsampler(s)

    writer = tf.summary.FileWriter("model-output", tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(t)

        print(v.shape)

    writer.close()

def test_downsampler():
    from config import FLAGS

    s = tf.constant(1.0, tf.float32, [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])

    downsampler = Downsampler(s.get_shape().as_list())

    t = downsampler(s)

    writer = tf.summary.FileWriter("model-output", tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(t)

        print(v.shape)

    writer.close()

def test_blender():
    from config import FLAGS

    s = tf.constant(1.0, tf.float32, [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])
    l = tf.constant(1.0, tf.float32, [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])

    blender = Blender(s.get_shape().as_list())

    t = blender(s, l)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(t)

        print(v.shape)
        print(v)

def test_extrator():
    from config import FLAGS

    s = tf.constant(1.0, tf.float32, [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])

    extrator = Extractor(s.get_shape().as_list())
    t = extrator(s)

    # writer = tf.summary.FileWriter("model-output", tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(t)

        print(v.shape)

    # writer.close()

def test():
    from config import FLAGS
    from input_ import Input, Watermark

    dim = [FLAGS.train_batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans]
    image_upsampler = Upsampler(dim)
    wm_upsampler = Upsampler([1] + dim[1:])
    downsampler = Downsampler(dim, factor = 4)
    blender = Blender(dim)
    extrator = Extractor(dim)
    
    input_ = Input(FLAGS.batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])
    images = input_('/data/yuming/watermark-data/train_images.tfr')
    wm = Watermark('/data/yuming/watermark-data/watermark.mat')()

    image_upsampled = image_upsampler(images)
    wm_upsampled = wm_upsampler(wm)
    image_blended = blender(image_upsampled, wm_upsampled)
    image_downsampled = downsampler(image_blended)
    wm_extracted = extrator(image_downsampled)

    writer = tf.summary.FileWriter("model-output", tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        v = sess.run(wm_extracted)

        print(v.shape)
     
    writer.close()

if __name__ == '__main__':
    # test_upsampler()
    # test_downsampler()
    # test_blender()
    # test_extrator()
    test()

    
