import numpy as np
import tensorflow as tf
import scipy.misc as scm
import imageio

tf.logging.set_verbosity(tf.logging.INFO)

pretrained_vgg = np.load("./_vgg16.npy", encoding="latin1").item()

#print(np.array(pretrained_vgg['fc6_W']).shape)

#print(np.array(pretrained_vgg['fc8_b']).shape)
'''def weights_init(layer_name):
    if "conv" in layer_name: #conv layer
        if "W" in layer_name: #kernel weights
            vals = pretrained_vgg[layer_name][:,:,0,:]
        else: #bias
            vals = pretrained_vgg[layer_name]
    else: #fc layer
        vals = pretrained_vgg[layer_name][0:4096]
    return tf.constant_initializer(vals)'''

def get_weights(layer_name):
    return tf.constant_initializer(pretrained_vgg[layer_name][0])

def get_bias(layer_name):
    return tf.constant_initializer(pretrained_vgg[layer_name][1])

def load_vgg16():
    input_layer = tf.Variable(tf.truncated_normal([1, 224, 224, 3], mean=125.0, stddev=50.0), dtype=tf.float32, name="input")

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer = get_weights('conv1_1'),
                              bias_initializer = get_bias('conv1_1'), name="conv1")
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv1_2'),
                              bias_initializer=get_bias('conv1_2'), name="conv2")
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, name="pool1")
    #########################

    conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv2_1'),
                              bias_initializer=get_bias('conv2_1'), name="conv3")
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv2_2'),
                              bias_initializer=get_bias('conv2_2'), name="conv4")
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2, name="pool2")
    #########################s

    conv5 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv3_1'),
                              bias_initializer=get_bias('conv3_1'), name="conv5")
    conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv3_2'),
                              bias_initializer=get_bias('conv3_2'), name="conv6")
    conv7 = tf.layers.conv2d(inputs=conv6, filters=256, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv3_3'),
                              bias_initializer=get_bias('conv3_3'), name="conv7")
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2,2], strides=2, name="pool3")
    #########################

    conv8 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv4_1'),
                              bias_initializer=get_bias('conv4_1'), name="conv8")
    conv9 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv4_2'),
                              bias_initializer=get_bias('conv4_2'), name="conv9")
    conv10 = tf.layers.conv2d(inputs=conv9, filters=512, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv4_3'),
                              bias_initializer=get_bias('conv4_3'), name="conv10")
    #pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[1,1], strides=1)
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2,2], strides=2, name="pool4")
    #########################

    conv11 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv5_1'),
                              bias_initializer=get_bias('conv5_1'), name="conv11")
    conv12= tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=[3, 3],
                              padding="same", activation=tf.nn.relu,
                              kernel_initializer=get_weights('conv5_2'),
                              bias_initializer=get_bias('conv5_2'), name="conv12")
    conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3, 3],
                            padding="same", activation=tf.nn.relu,
                            kernel_initializer=get_weights('conv5_3'),
                            bias_initializer=get_bias('conv5_3'), name="conv13")
    #pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[1,1], strides=1)
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2,2], strides=2, name="pool5")
    #########################

    '''pool5_flat = tf.reshape(pool5, [-1, 7*7*512]) #Flatten
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu,
                            kernel_initializer=get_weights('fc6'),
                            bias_initializer=get_bias('fc6'), name="fc6")
    dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu,
                            kernel_initializer=get_weights('fc7'),
                            bias_initializer=get_bias('fc7'), name="fc7")
    logits = tf.layers.dense(inputs=dense2, units=1000, activation=None,
                            kernel_initializer=get_weights('fc8'),
                            bias_initializer=get_bias('fc8'),
                            name="logits")'''

    model = {
    "input":input_layer,
    "conv1_1":conv1,
    "conv1_2":conv2,
    "pool1":pool1,
    "conv2_1":conv3,
    "conv2_2":conv4,
    "pool2":pool2,
    "conv3_1":conv5,
    "conv3_2":conv6,
    "conv3_3":conv7,
    "pool3":pool3,
    "conv4_1":conv8,
    "conv4_2":conv9,
    "conv4_3":conv10,
    "pool4":pool4,
    "conv5_1":conv11,
    "conv5_2":conv12,
    "conv5_3":conv13,
    "pool5":pool5}
    return model

def get_noisy_img(shape):
    img = np.random.uniform(0.0, 1.0, shape).astype('float64')
    return np.reshape(img, shape)

def compute_content_cost(a_C, a_G):

    m, nh, nw, nc = a_G.get_shape().as_list()

    a_C_t = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_t = tf.transpose(tf.reshape(a_G, [-1]))

    J_c = tf.reduce_sum((a_C_t - a_G_t)**2) / (4*nh*nw*nc)

    return J_c

def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))

def compute_layer_style_cost(a_S, a_G):

    m, nh, nw, nc = a_G.get_shape().as_list()

    a_S = tf.reshape(a_S, [nh*nw, nc])
    a_G = tf.reshape(a_G, [nh*nw, nc])

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    J_s_layer = tf.reduce_sum((GS - GG)**2) / (4 * nc**2 * (nw * nh)**2)
    return J_s_layer

def compute_style_cost(session, model, STYLE_LAYERS):
    J_s = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = session.run(out)
        a_G = out
        J_s_layer = compute_layer_style_cost(a_S, a_G)
        J_s += coeff*J_s_layer
    return J_s

def total_cost(Jc, Js, alpha=10, beta=40):
    return alpha*Jc + beta*Js

source_img = imageio.imread('images/source.jpg')
painting_img = imageio.imread('images/van_gogh.jpg')
result_img = get_noisy_img(source_img.shape)

imageio.imsave('output/test_random.jpg', result_img)
