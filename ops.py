
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2 
import csv

def load_image(path):
	img = cv2.resize(cv2.imread(path), (224,224))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img / 127.5 - 1

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
			# img = cv2.imread(img_path)
			images.append(img_path)
			boxes.append(box)
			labels.append(idx)

	return np.asarray(images), np.asarray(boxes), np.asarray(labels), labels_dict, len(labels)

def load_val_data(args, labels_dict):
	print "Preparing Validation Data..."

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

		# img = cv2.imread(img_path)
		images.append(img_path)
		boxes.append(box)
		labels.append(labels_dict[line[1]])


	return np.asarray(images), np.asarray(boxes), np.asarray(labels), len(labels)

# #function to get training data
# def load_train_data(args, path):
# 	print "Preparing " + path + " data..."

# 	paths = os.path.join(args.data, path + "/*_*.jpg")
# 	data_count = len(glob(paths))

# 	img_list = glob(paths)
# 	label_list = [int(path.split('_')[-1].split('.')[-2]) for path in img_list]
	
# 	if len(label_list) > 0:
# 		num_class = np.array(label_list).max()
# 	else:
# 		print "No " + path + " data read-!"
# 		num_class = 0
	

# 	img_list = tf.convert_to_tensor(img_list, dtype=tf.string)
# 	label_list = tf.convert_to_tensor(label_list, dtype=tf.int32)


# 	filename_queue = tf.train.slice_input_producer([img_list, label_list])
# 	#filename_queue = tf.train.slice_input_producer([img_list, label_list])

# 	label = tf.cast(filename_queue[1], dtype=tf.int32)
# 	label = tf.one_hot(label, num_class)

# 	#image_reader = tf.WholeFileReader()
# 	image_file = tf.read_file(filename_queue[0])
# 	images = tf.image.decode_jpeg(image_file, channels=3)

# 	#input image range from -1 to 1
# 	#center crop 32x32 since raw images are not center cropped.
# 	images = tf.image.central_crop(images, 0.5)
# 	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
# 	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1  


# 	train_batch, train_labels = tf.train.shuffle_batch([images, label],
# 														batch_size=args.batch_size,
# 														capacity=args.batch_size*2,
# 														min_after_dequeue=args.batch_size
# 														)


# 	return train_batch, train_labels, num_class, data_count


def read_test_img(args, path):

	path_ = os.path.join(args.data, "test/"+path+".jpg")
	if not os.path.isfile(path_):
		return np.zeros((args.input_width, args.input_height,3), dtype=np.float32)

	img = cv2.imread(path_)
	img = cv2.resize(img, (args.input_width,args.input_height))
	return img


def read_csv(path):
	print path
	with open(path+"/test.csv", 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
		res = []
		for row in spamreader:
			res.append(row[0])

		return res[1:]






