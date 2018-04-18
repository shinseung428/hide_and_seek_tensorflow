
import os
import tensorflow as tf
import numpy as np

import scipy.misc
import scipy.ndimage

from skimage.measure import label, regionprops
from skimage.morphology import closing, square

import matplotlib
import matplotlib.cm

from glob import glob 

def find_overlap(box1, box2):
	#check if one rect is on the left of the other
	if box1[0] > box2[0] + box2[2] or box2[0] > box1[0] + box1[2]:
		return (0,0,0,0)
	#check if one rect is above the other
	if box1[1] > box2[1] + box2[3] or box2[1] > box1[1] + box1[3]:
		return (0,0,0,0)


	x_left = max(box1[0], box2[0])
	y_top = max(box1[1], box2[1])
	x_right = min(box1[0] + box1[2], box2[0] + box2[2])
	y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

	return (x_left, y_top, x_right-x_left, y_bottom-y_top)

def box_size(box):
	return (box[2] + 1) * (box[3] + 1)

def calc_loc(pred_box, gt_box):
	total = 0.0
	for idx in range(len(pred_box)):		
		# px, py, pw, ph = pred_box[idx]
		# gx, gy, gw, gh = gt_box[idx]
	
		#calculate iou
		overlap_box = find_overlap(pred_box[idx], gt_box[idx])
		overlap_area = box_size(overlap_box)
		pred_box_area = box_size(pred_box[idx])
		gt_box_area = box_size(gt_box[idx])

		total_area = float(overlap_area)/(pred_box_area + gt_box_area - overlap_area)
		
		total += total_area

	return total/len(pred_box)

def find_largest_box(heatmap):
	res = []
	for idx in range(len(heatmap)):
		mask = heatmap[idx] > 0
		masked = label(mask)

		largest_area = 0
		box = []
		for region in regionprops(masked):
			if region.area > largest_area:
				largest_area = region.area
				minr, minc, maxr, maxc = region.bbox
				box = (minc, minr, maxc - minc, maxr-minr)
		
		if largest_area == 0:
			res.append((0,0,0,0))
		else:
			res.append(box)

	return res


def colorize(value, vmin=None, vmax=None, cmap='plasma'):
	# normalize
	vmin = tf.reduce_min(value) if vmin is None else vmin
	vmax = tf.reduce_max(value) if vmax is None else vmax
	value = (value - vmin) / (vmax - vmin) # vmin..vmax

	# squeeze last dim 
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
		# if np.random.rand() > 0.5:
		# 	image = scipy.ndimage.interpolation.rotate(image, np.random.randint(-10,10))
		# 	image = scipy.misc.imresize(image, (args.input_width, args.input_height))
		if np.random.rand() > 0.5:
			size = np.random.choice([50,56,60], 1)[0]
			startx = np.random.randint(0,args.input_width-size)
			starty = np.random.randint(0,args.input_height-size)
			new_img = image[startx:startx+size, starty:starty+size, :]
			image = scipy.misc.imresize(image, (args.input_width, args.input_height))
		if np.random.rand() > 0.5:
			image = scipy.ndimage.interpolation.zoom(image, (1.5,1.5,1.0))
			image = scipy.misc.imresize(image, (args.input_width, args.input_height))

		image = image / 255.0
		# use 13x13 grid
		# grid = np.random.randint(10, 15)
		grid = np.random.choice([8,13,16], 1)[0]
		breaks = args.input_width // grid + 1
		for x in range(breaks):
			for y in range(breaks):
				prob = np.random.rand()
				if prob >= 0.5:
					image[x*grid:x*grid+grid,y*grid:y*grid+grid,:] = mean
	else:
		image = image / 255.0

	return image 

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
			
			int_box = []
			for b in box:
				int_box.append(int(b))
			box = int_box

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
		box = line[2:]
		int_box = []
		for b in box:
			int_box.append(int(b))
		box = int_box
		img_path = os.path.join(path+"/images", name)

		images.append(img_path)
		boxes.append(box)
		labels.append(labels_dict[line[1]])

	return images, boxes, labels, len(labels)





