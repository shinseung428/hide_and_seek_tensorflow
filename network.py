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
        self.out_class = args.out_class
        
        self.beta = 0.0001

        self.build_model()

        #summary
        self.loss_sum = tf.summary.scalar("loss", self.loss) 
        self.tr_acc_sum = tf.summary.scalar("acc", self.acc) 
        self.val_acc_sum = tf.summary.scalar("val_acc", self.val_acc) 
        self.train_img_sum = tf.summary.image("tr_img", self.train_imgs, max_outputs=5)
        self.val_img_sum = tf.summary.image("val_img", self.val_imgs, max_outputs=5)

        self.tr_classmap_sum = tf.summary.image("tr_classmap", self.tr_colorized_classmap, max_outputs=5)
        self.val_classmap_sum = tf.summary.image("val_classmap", self.val_colorized_classmap, max_outputs=5)

    #structure of the model
    def build_model(self):
        self.train_imgs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channel])
        self.train_labels = tf.placeholder(tf.int32, [self.batch_size,])
        train_labels = tf.one_hot(self.train_labels, self.out_class)

        self.val_imgs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.input_channel])
        self.val_labels = tf.placeholder(tf.int32, [self.batch_size,])
        val_labels = tf.one_hot(self.val_labels, self.out_class)

        self.pred_logits, self.end_points = self.VGG(self.train_imgs, name="VGG")

        self.vars = tf.trainable_variables()

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_logits, labels=train_labels))
        penalty = 0
        for var in self.vars:
            penalty += tf.nn.l2_loss(var)
        
        self.loss = tf.reduce_mean(loss + self.beta*penalty)

        self.pred = tf.argmax(self.end_points, axis=1)
        gt = tf.argmax(train_labels, axis=1)
        correct_prediction = tf.equal(self.pred, gt)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        #Validation Result
        val_logits, val_points = self.VGG(self.val_imgs, name="VGG", reuse=True)
        val_pred = tf.argmax(val_points, axis=1)
        val_gt = tf.argmax(val_labels, axis=1)
        val_prediction = tf.equal(val_pred, val_gt)
        self.val_acc = tf.reduce_mean(tf.cast(val_prediction, dtype=tf.float32)) 

        #Training Classmap
        CAM_image = tf.image.resize_bilinear(self.last_layer, [self.input_height, self.input_width])
        CAM_img = tf.reshape(CAM_image, [-1, self.input_height*self.input_width, 1024])
        label_w = tf.gather(tf.transpose(self.weights), self.pred)
        label_w = tf.reshape(label_w, [-1, 1024, 1])

        classmap = tf.matmul(CAM_img, label_w)
        classmap = tf.reshape(classmap, [-1, self.input_height, self.input_width, 1])
        
        colorized = []
        for idx in range(self.batch_size):
            colorized.append(colorize(classmap[idx]))

        self.tr_colorized_classmap = tf.convert_to_tensor(colorized)

        #Validation Classmap
        CAM_image = tf.image.resize_bilinear(self.last_layer, [self.input_height, self.input_width])
        CAM_img = tf.reshape(CAM_image, [-1, self.input_height*self.input_width, 1024])
        label_w = tf.gather(tf.transpose(self.weights), val_pred)
        label_w = tf.reshape(label_w, [-1, 1024, 1])

        classmap = tf.matmul(CAM_img, label_w)
        classmap = tf.reshape(classmap, [-1, self.input_height, self.input_width, 1])
        classmap = classmap > 0.2

        colorized = []
        for idx in range(self.batch_size):
            colorized.append(colorize(classmap[idx]))

        self.val_colorized_classmap = tf.convert_to_tensor(colorized)        


    def AlexNet(self, input, name="VGG16", reuse=False):
      with tf.variable_scope(name, reuse=reuse) as scope:
        net = conv2d(input, 3, 96, 3, 1, padding='SAME', name='conv1')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn1")
        net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        net = max_pool(net, 3, 2, padding='VALID', name='pool1')
        
        net = conv2d(net, 96, 256, 3, 1, padding='SAME', name='conv2')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn2")
        net = tf.nn.local_response_normalization(net, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        net = max_pool(net, 3, 2, padding='VALID', name='pool2')
        
        net = conv2d(net, 256, 384, 3, 1, padding='SAME', name='conv3')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn3")

        net = conv2d(net, 384, 384, 3, 1, padding='SAME', name='conv4')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn4")

        net = conv2d(net, 384, 256, 3, 1, padding='SAME', name='conv5')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn5")
        net = max_pool(net, 3, 2, padding='VALID', name='pool3')
        
        #extra conv layers
        net = conv2d(net, 256, 512, 3, 1, padding='SAME', name='conv6')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn6")
        net = conv2d(net, 512, 1024, 3, 1, padding='SAME', name='conv7')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn7")
        
        self.last_layer = net
        #Global Average Pooling
        gap = tf.reduce_mean(net, axis=[1,2])
        
        flattened = tf.reshape(gap, (self.batch_size, -1))
        net, self.weights = linear(flattened, self.out_class, name='linear')

        return net, tf.nn.softmax(net)


    def VGG(self, input, name="VGG16", reuse=False):
      with tf.variable_scope(name, reuse=reuse) as scope:
        # block 1
        net = conv2d(input, 3, 64, 3, 1, padding='SAME', name='conv1')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn1")
        
        net = conv2d(net, 64, 64, 3, 1, padding='SAME', name='conv2')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn2")        
        
        net = max_pool(net, 2, 2, padding='VALID', name='pool1')
                
        # block 2
        net = conv2d(net, 64, 128, 3, 1, padding='SAME', name='conv3')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn3")

        net = conv2d(net, 128, 128, 3, 1, padding='SAME', name='conv4')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn4")

        net = max_pool(net, 2, 2, padding='VALID', name='pool2')

        # block 3
        net = conv2d(net, 128, 256, 3, 1, padding='SAME', name='conv5')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn5")
        
        net = conv2d(net, 256, 256, 3, 1, padding='SAME', name='conv6')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn6")

        net = conv2d(net, 256, 256, 3, 1, padding='SAME', name='conv7')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn7")

        net = max_pool(net, 2, 2, padding='VALID', name='pool3')

        # block 4
        net = conv2d(net, 256, 512, 3, 1, padding='SAME', name='conv8')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn8")
        
        net = conv2d(net, 512, 512, 3, 1, padding='SAME', name='conv9')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn9")

        net = conv2d(net, 512, 1024, 3, 1, padding='SAME', name='conv10')
        net = tf.nn.relu(net)
        net = batch_norm(net, name="bn10")

        self.last_layer = net
        #Global Average Pooling
        gap = tf.reduce_mean(net, axis=[1,2])
        
        flattened = tf.reshape(gap, (self.batch_size, -1))
        net, self.weights = linear(flattened, self.out_class, name='linear')

        return net, tf.nn.softmax(net)







