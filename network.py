import tensorflow as tf
import numpy as np

from ops import *
from architecture import *

class network():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.input_channel = args.input_channel

        self.build_model()

        #summary
        self.loss_sum = tf.summary.scalar("loss", self.loss) 
        self.tr_acc_sum = tf.summary.scalar("acc", self.acc) 
        self.val_acc_sum = tf.summary.scalar("val_acc", self.val_acc) 
        self.train_img_sum = tf.summary.image("input_img", self.train_imgs, max_outputs=10)

        self.classmap_sum = tf.summary.image("classmap", self.classmap, max_outputs=10)

    #structure of the model
    def build_model(self):
        self.train_imgs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channel])
        self.train_labels = tf.placeholder(tf.int32, [self.batch_size,])
        train_labels = tf.one_hot(self.train_labels, 200)

        self.val_imgs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channel])
        self.val_labels = tf.placeholder(tf.int32, [self.batch_size,])
        val_labels = tf.one_hot(self.val_labels, 200)

        self.pred_logits, self.end_points = self.VGG16(self.train_imgs, name="VGG16")

        self.vars = tf.trainable_variables()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_logits, labels=train_labels))

        self.pred = tf.argmax(self.end_points, axis=1)
        gt = tf.argmax(train_labels, axis=1)
        correct_prediction = tf.equal(self.pred, gt)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        #self.acc = tf.Print(self.acc, [gt], message="\ngt:", summarize=10)
        #self.acc = tf.Print(self.acc, [self.pred], message="\npred:", summarize=10)

        #Validation Result
        self.val_logits, self.val_points = self.VGG16(self.val_imgs, name="VGG16", reuse=True)
        val_pred = tf.argmax(self.val_points, axis=1)
        val_gt = tf.argmax(val_labels, axis=1)
        val_prediction = tf.equal(val_pred, val_gt)
        self.val_acc = tf.reduce_mean(tf.cast(val_prediction, dtype=tf.float32)) 


        self.CAM_image = tf.image.resize_bilinear(self.last_layer, [224, 224])
        CAM_img = tf.reshape(self.CAM_image, [-1, 224*224, 1024])
        label_w = tf.gather(tf.transpose(self.weights), self.pred)
        label_w = tf.reshape(label_w, [-1, 1024, 1])

        classmap = tf.matmul(CAM_img, label_w)
        self.classmap = tf.tile(tf.reshape(classmap, [-1, 224, 224, 1]), [1,1,1,3])



    def VGG16(self, input, name="VGG16", reuse=False):
      with tf.variable_scope(name, reuse=reuse) as scope:
        net = conv2d(input, 3, 64, 1, 1, padding='VALID', name='conv0')
        net = tf.nn.relu(net)
        net = conv2d(net, 64, 64, 1, 1, padding='VALID', name='conv1')
        net = tf.nn.relu(net)
        net = max_pool(net, 2, 2, padding='VALID', name='pool0')

        net = conv2d(net, 64, 128, 1, 1, padding='VALID', name='conv2')
        net = tf.nn.relu(net)
        net = conv2d(net, 128, 128, 1, 1, padding='VALID', name='conv3')
        net = tf.nn.relu(net)
        net = max_pool(net, 2, 2, padding='VALID', name='pool1')

        net = conv2d(net, 128, 256, 1, 1, padding='VALID', name='conv4')
        net = tf.nn.relu(net)
        net = conv2d(net, 256, 256, 1, 1, padding='VALID', name='conv5')
        net = tf.nn.relu(net)
        net = conv2d(net, 256, 256, 1, 1, padding='VALID', name='conv6')
        net = tf.nn.relu(net)
        net = max_pool(net, 2, 2, padding='VALID', name='pool2')
        
        net = conv2d(net, 256, 512, 1, 1, padding='VALID', name='conv7')
        net = tf.nn.relu(net)
        net = conv2d(net, 512, 512, 1, 1, padding='VALID', name='conv8')
        net = tf.nn.relu(net)
        net = conv2d(net, 512, 512, 1, 1, padding='VALID', name='conv9')
        net = tf.nn.relu(net)
        net = max_pool(net, 2, 2, padding='VALID', name='pool3')

        net = conv2d(net, 512, 512, 1, 1, padding='VALID', name='conv10')
        net = tf.nn.relu(net)
        net = conv2d(net, 512, 512, 1, 1, padding='VALID', name='conv11')
        net = tf.nn.relu(net)
        net = conv2d(net, 512, 512, 1, 1, padding='VALID', name='conv12')
        net = tf.nn.relu(net)
        net = conv2d(net, 512, 1024, 1, 1, padding='VALID', name='conv13')
        
        #net = max_pool(net, 2, 2, padding='VALID', name='pool4')        

        self.last_layer = net
        gap = tf.reduce_mean(net, axis=[1,2])
        
        flattened = tf.reshape(gap, (self.batch_size, -1))
        net, self.weights = linear(flattened, 200, name='linear')

        return net, tf.nn.softmax(net)










