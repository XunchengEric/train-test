#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d

# create weight variable
def create_var(name, shape, initializer, trainable=True):
    with tf.variable_scope(name):
        return tf.get_variable(name, shape=shape, dtype=tf.float64, initializer=initializer, trainable=trainable)

def conv(input_layer, input_channels, output_channels, kernel_size, stride=1, padding='VALID', bias=True, name="conv"):
    with tf.variable_scope(name):
        weights = create_var("kernel", [kernel_size, kernel_size, input_channels, output_channels], conv2d_initializer())

        layer = tf.nn.conv2d(input=input_layer, filters=weights, strides=[1, stride, stride, 1], padding=padding)
        
        if bias:
            biases = create_var("bias", [output_channels], tf.zeros_initializer())

            layer += biases
        return layer

def maxpool(input_layer, kernel_size, stride=None, padding='VALID', name="maxpool"):
    with tf.variable_scope(name):
        if stride is None:
            stride = kernel_size
        return tf.nn.max_pool1d(input=input_layer, ksize=[1, kernel_size, 1], strides=[1, stride, 1], padding=padding)

def deconv(input_layer, input_channels, output_channels, kernel_size, stride=1, padding='VALID', bias=True, name="deconv"):
    with tf.variable_scope(name):
        weights = create_var("kernel", [kernel_size, output_channels, input_channels], conv2d_initializer())

        batch_size = tf.shape(input_layer)[0]
        input_size = tf.shape(input_layer)[1]
        output_size = (input_size-1) * stride + kernel_size

        layer = tf.nn.conv1d_transpose(input=input_layer, filters=weights, output_shape=[batch_size, output_size, output_channels], strides=[1, stride, 1], padding=padding)
        
        if bias:
            biases = create_var("bias", [output_channels], tf.zeros_initializer())

            layer += biases
        return layer

L = 6
site_number = int(L**2)

def inference_single(x, name='MC'):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #x = tf.transpose(x, [0, 2, 3, 1])
        x = conv(x, 1, 32, 5, name="conv_1")
        x = tf.reshape(x, [-1, site_number, 32])
        x = maxpool(x, 2, name="maxpool_1")
        x = deconv(x, 32, 16, 2, stride=2, name="deconv_1")
        x = tf.reshape(x, [-1, L, L, 16])
        x = tf.tile(x, [1, 3, 3, 1])
        x = tf.slice(x, [0, L-2, L-2, 0], [-1, L+4, L+4, -1])

        x = conv(x, 16, 8, 5, name="conv_2")
        x = tf.reshape(x, [-1, site_number, 8])
        x = maxpool(x, 2, name="maxpool_2")
        x = deconv(x, 8, 16, 2, stride=2, name="deconv_2")
        x = tf.reshape(x, [-1, L, L, 16])
        x = tf.tile(x, [1, 3, 3, 1])
        x = tf.slice(x, [0, L-1, L-1, 0], [-1, L+2, L+2, -1])

        x = conv(x, 16, 16, 3, name="conv_3")
        x = tf.reshape(x, [-1, site_number, 16])
        x = maxpool(x, 2, name="maxpool_3")
        x = deconv(x, 16, 16, 2, stride=2, name="deconv_3")
        x = tf.reshape(x, [-1, L, L, 16])
        x = tf.tile(x, [1, 3, 3, 1])
        x = tf.slice(x, [0, L-1, L-1, 0], [-1, L+2, L+2, -1])
                    
        x = conv(x, 16, 16, 3, name="conv_4")
        x = tf.reshape(x, [-1, site_number, 16])
        x = maxpool(x, 2, name="maxpool_4")
        x = deconv(x, 16, 16, 2, stride=2, name="deconv_4")
        x = tf.reshape(x, [-1, L, L, 16])
        x = tf.tile(x, [1, 3, 3, 1])
        x = tf.slice(x, [0, L-1, L-1, 0], [-1, L+2, L+2, -1])

        x = conv(x, 16, 16, 3, name="conv_5")
        x = tf.reshape(x, [-1, site_number, 16])
        x = maxpool(x, 2, name="maxpool_5")
        x = deconv(x, 16, 16, 2, stride=2, name="deconv_5")
        x = tf.reshape(x, [-1, L, L, 16])
        x = tf.tile(x, [1, 3, 3, 1])
        x = tf.slice(x, [0, L-1, L-1, 0], [-1, L+2, L+2, -1])
                    
        x = conv(x, 16, 16, 3, name="conv_6")
        x = tf.reshape(x, [-1, site_number, 16])
        x = maxpool(x, 2, name="maxpool_6")
        x = deconv(x, 16, 1, 2, stride=2, name="deconv_6")
        x = tf.reshape(x, [-1, site_number])
        x = tf.reduce_prod(x, axis=1)
        x = tf.squeeze(x)
        
        return x


input = tf.placeholder(dtype=tf.float64, shape=[None, 1, L+4, L+4], name='spin_lattice')

x1 = tf.transpose(input, [0, 2, 3, 1])
#logits = inference_single(x1)
#x2 = tf.image.rot90(x1, 1)
#x3 = tf.image.rot90(x1, 2)
#x4 = tf.image.rot90(x1, 3)

y1 = inference_single(x1)
#y2 = inference_single(x2)
#y3 = inference_single(x3)
#y4 = inference_single(x4)
#y = y1 + y2 + y3 + y4
logits = tf.identity(y1, name="logits")

init = tf.global_variables_initializer()
saver_def = tf.train.Saver().as_saver_def()

#print('Run this operation to initialize variables     : ', init.name)
#print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
#print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
#print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters += variable_parameters
#print(total_parameters)

grads_and_vars = tf.train.GradientDescentOptimizer(1.0).compute_gradients(logits, tf.trainable_variables())
print(grads_and_vars)
with open('graph_6x6_1w6_nosym.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
