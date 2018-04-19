import os
import math
import tensorflow as tf
from config import *
from network import *
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



def validate(args, sess, model):
    
    step = 0
    

    #saver
    saver = tf.train.Saver()        
    last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print("Loaded model file from " + ckpt_name)
    


    #Prepare data
    #prepare training data
    _, _, _, labels_dict, _ = load_tr_data(args) 
    valid_imgs, valid_boxes, valid_labels, valid_count = load_val_data(args, labels_dict)    

    zipped_data = zip(valid_imgs, valid_boxes, valid_labels)
    np.random.shuffle(zipped_data)

    new_imgs, new_boxes, new_labels = [], [], []
    for img, box, label in zipped_data:
        new_imgs.append(img)
        new_boxes.append(box)
        new_labels.append(label)
    valid_imgs = np.asarray(new_imgs)
    valid_boxes = np.asarray(new_boxes)
    valid_labels = np.asarray(new_labels)
    

    print "Processing Validation Data..."
    #calculate total validation result
    ave_top_1 = 0.0
    ave_top_5 = 0.0 
    ave_IOU = 0.0
    last_idx = 0
    val_batch_idxs = valid_count // args.batch_size
    for idx in tqdm(range(0, val_batch_idxs)):
        val_img_batch = valid_imgs[args.batch_size*idx:args.batch_size*idx+args.batch_size]
        val_lab_batch = valid_labels[args.batch_size*idx:args.batch_size*idx+args.batch_size]
        val_box_batch = valid_boxes[args.batch_size*idx:args.batch_size*idx+args.batch_size]

        val_batch = [load_image(path, args, is_training=False) for path in val_img_batch]
        val_batch = np.asarray(val_batch)

        #skip if insufficient elements
        if len(val_img_batch) < args.batch_size:
            break

        dictionary = {
                      model.val_imgs:val_batch,
                      model.val_labels:val_lab_batch
                      }
        #Update Network
        val_acc, val_top5, val_cam = sess.run([model.val_acc,
                                               model.val_top5, 
                                               model.val_classmap, 
                                              ],
                                              feed_dict=dictionary
                                             )



        #calculate IOU and other stuff here
        pred_val_boxes = find_largest_box(val_cam)
        IOU_acc = calc_loc(val_box_batch, pred_val_boxes)

        ave_top_1 += val_acc
        ave_top_5 += val_top5
        ave_IOU += IOU_acc

        last_idx = idx
    

    print "Val_Acc_1: [%.4f] Val_Acc_5: [%.4f] IOU_Acc: [%.4f]"%(ave_top_1/last_idx, ave_top_5/last_idx, ave_IOU/last_idx)



      
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network(args)

        print('Process Validation Data...')
        validate(args, sess, model)

main(args)
