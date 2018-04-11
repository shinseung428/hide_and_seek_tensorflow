
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import scipy.misc

import matplotlib
import matplotlib.cm


def colorize(value, vmin=None, vmax=None, cmap=None):
	"""
	A utility function for TensorFlow that maps a grayscale image to a matplotlib
	colormap for use with TensorBoard image summaries.
	By default it will normalize the input value to the range 0..1 before mapping
	to a grayscale colormap.
	Arguments:
	  - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
	    [height, width, 1].
	  - vmin: the minimum value of the range used for normalization.
	    (Default: value minimum)
	  - vmax: the maximum value of the range used for normalization.
	    (Default: value maximum)
	  - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
	    (Default: 'gray')
	Example usage:
	```
	output = tf.random_uniform(shape=[256, 256, 1])
	output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
	tf.summary.image('output', output_color)
	```

	Returns a 3D tensor of shape [height, width, 3].
	"""

	# normalize
	vmin = tf.reduce_min(value) if vmin is None else vmin
	vmax = tf.reduce_max(value) if vmax is None else vmax
	value = (value - vmin) / (vmax - vmin) # vmin..vmax

	# squeeze last dim if it exists
	value = tf.squeeze(value)

	# quantize
	indices = tf.to_int32(tf.round(value * 255))

	# gather
	cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
	colors = cm(np.arange(256))[:, :3]
	colors = tf.constant(colors, dtype=tf.float32)
	value = tf.gather(colors, indices)

	return value



def load_image(path, args):
	image = scipy.misc.imread(path, mode='RGB')
	image = scipy.misc.imresize(image, (args.input_width, args.input_height))

	return image / 127.5 - 1

def load_tr_data(args):
	print "Preparing Training Data..."

	folders = os.path.join(args.data, "train")
	folders = glob(folders+"/*")
	
	images = []
	boxes = []
	labels = []
	labels_dict = {}
	for idx, folder in enumerate(folders):
		txt_path = os.path.join(folder,folder.split('/')[-1]+"_boxes.txt")
		fb = open(txt_path, "r")
		data = fb.read().split('\n')
		data = data[:len(data)-1]
		labels_dict[folder.split('/')[-1]] = idx
		
		for line in data:
			line = line.split('\t')
			name = line[0]
			box = line[1:]
			img_path = os.path.join(folder+"/images", name)
			
			images.append(img_path)
			boxes.append(box)
			labels.append(idx)

	return images, boxes, labels, labels_dict, len(labels)

def load_val_data(args, labels_dict):
	print("Preparing Validation Data...")

	path = os.path.join(args.data, "val")
	
	images = []
	boxes = []
	labels = []

	txt_path = os.path.join(path,"val_annotations.txt")
	fb = open(txt_path, "r")
	data = fb.read().split('\n')
	data = data[:len(data)-1]
	
	for line in data:
		line = line.split('\t')
		name = line[0]
		box = line[1:]
		img_path = os.path.join(path+"/images", name)

		images.append(img_path)
		boxes.append(box)
		labels.append(labels_dict[line[1]])
	
	return images, boxes, labels, len(labels)





