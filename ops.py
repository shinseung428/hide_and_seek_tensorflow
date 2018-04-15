
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.ndimage

import matplotlib
import matplotlib.cm


def colorize(value, vmin=None, vmax=None, cmap='plasma'):
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


def load_image(path, args, is_training=True):
	# mean of the pixel values in the whole dataset
	mean = 0.442872096287

	image = scipy.misc.imread(path, mode='RGB')
	image = scipy.misc.imresize(image, (args.input_width, args.input_height))
	
	
	if is_training:
		#probably add data augmentation before randomly blocking image patches		
		if np.random.rand() > 0.5:
			image = np.flip(image, 1)
		if np.random.rand() > 0.5:
			image = scipy.ndimage.interpolation.rotate(image, np.random.randint(-15,15))
			image = scipy.misc.imresize(image, (args.input_width, args.input_height))
		if np.random.rand() > 0.5:
			image = scipy.ndimage.interpolation.zoom(image, (1.5,1.5,1.0))
			image = scipy.misc.imresize(image, (args.input_width, args.input_height))
		if np.random.rand() > 0.5:
			size = 56
			startx = np.random.randint(0,args.input_width-size)
			starty = np.random.randint(0,args.input_height-size)
			new_img = image[startx:startx+size, starty:starty+size, :]
			image = scipy.misc.imresize(image, (args.input_width, args.input_height))

		image = image / 255.0
		# use 13x13 grid
		# grid = np.random.randint(10, 15)
		# grid = np.random.choice([8,13,16], 1)[0]
		# breaks = args.input_width // grid + 1
		# for x in range(breaks):
		# 	for y in range(breaks):
		# 		prob = np.random.rand()
		# 		if prob >= 0.5:
		# 			image[x*grid:x*grid+grid,y*grid:y*grid+grid,:] = mean
	else:
		image = image / 255.0

	return image 

def load_tr_data(args):
	print "Preparing Training Data..."

	folders = os.path.join(args.data, "train")
	folders = glob(folders+"/*")
	
	# mean = 0
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
			#img = load_image(img_path, args)
			#mean += np.sum(img)
			
	
	# print 2176804927.67/(len(labels) * args.input_height * args.input_height * args.input_channel)
	
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





