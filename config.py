# Copyright 2017 Yurui Ming (yrming@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for configuring the prediction model."""

import tensorflow as tf
from tensorflow.python.platform import flags

flags.DEFINE_integer('num_iterations', 10000, 'number of training iterations.')

flags.DEFINE_float('learning_rate', 0.001, 'the base learning rate of the generator')
flags.DEFINE_boolean('lr_decay', True, 'whether or not enable learning rate decay')

flags.DEFINE_float('lr_decay_factor', 0.92, 'learning rate decay factor')
flags.DEFINE_float('lr_decay_steps', 400, 'after the specified steps then learning rate decay')

flags.DEFINE_string('gpus', '', 'visible GPU list')

flags.DEFINE_string('data_dir', '', 'directory of data')
flags.DEFINE_string('output_dir', '', 'directory for model outputs.')
flags.DEFINE_string('checkpoint_dir', '', 'directory of checkpoint files.')

flags.DEFINE_integer('batch_size', 8, 'batch size for training.')
flags.DEFINE_integer('test_size', 412, 'number of samples for testing')

flags.DEFINE_integer('img_height', 240, 'height of the image.')
flags.DEFINE_integer('img_width', 320, 'width of the image.')
flags.DEFINE_integer('num_chans', 3, 'number of channels.')

flags.DEFINE_integer('train_batch_size', 32, 'batch size for training.')
flags.DEFINE_integer('xval_batch_size', 0, 'batch size for cross evaluation')
flags.DEFINE_integer('test_batch_size', 0, 'batch size for test')

flags.DEFINE_integer('summary_interval', 5, 'how often to record tensorboard summaries.')
flags.DEFINE_integer('validation_interval', 10, 'how often to run a batch through the validation model')
# flags.DEFINE_integer('save_interval', 2000, 'how often to save a model checkpoint.')

flags.DEFINE_boolean('validate', False, 'whether do cross-validation or not')

flags.DEFINE_integer('image_seq', 0, 'test image sequence number.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


