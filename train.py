import os
import math
import tensorflow as tf
from config import *
from network import *
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



def train(args, sess, model):
    #optimizers
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
                                                0.01,                # Base learning rate.
                                                batch * args.batch_size,  # Current index into the dataset.
                                                100000,          # Decay step.
                                                0.96,                # Decay rate.
                                                staircase=True)    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(model.loss, var_list=model.vars, global_step=batch)
    # optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer").minimize(model.loss, var_list=model.vars, global_step=batch)


    start_epoch = 0
    step = 0
    global_step = 0

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print("Loaded model file from " + ckpt_name)
        ckpt_numbers = ckpt_name.split('-')
        start_epoch = int(ckpt_numbers[-1])
        global_step = start_epoch*100000
    else:
        tf.global_variables_initializer().run()



    #summary init
    all_summary = tf.summary.merge([model.loss_sum,
                                    model.tr_acc_sum, 
                                    model.val_acc_sum,
                                    model.val_acc5_sum,
                                    model.train_img_sum,
                                    model.tr_classmap_sum,
                                    model.val_img_sum,
                                    model.val_classmap_sum,
                                    ])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)


    #Prepare data
    #prepare training data
    train_imgs, train_boxes, train_labels, labels_dict, train_count = load_tr_data(args) 
    valid_imgs, valid_boxes, valid_labels, valid_count = load_val_data(args, labels_dict)    

    zipped_data = zip(train_imgs, train_boxes, train_labels)
    np.random.shuffle(zipped_data)
    
    new_imgs, new_boxes, new_labels = [], [], []
    for img, box, label in zipped_data:
        new_imgs.append(img)
        new_boxes.append(box)
        new_labels.append(label)
    train_imgs = np.asarray(new_imgs)
    train_boxes = np.asarray(new_boxes)
    train_labels = np.asarray(new_labels)


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


    print("Training Count: %d Class Num: %d "%(train_count, train_labels.max()+1))
    

    batch_idxs = train_count // args.batch_size
    #training starts here
    for epoch in range(start_epoch, args.epochs):
        for idx in range(0, batch_idxs):
            tr_img_batch = train_imgs[args.batch_size*idx:args.batch_size*idx+args.batch_size]
            tr_lab_batch = train_labels[args.batch_size*idx:args.batch_size*idx+args.batch_size]
            tr_box_batch = train_boxes[args.batch_size*idx:args.batch_size*idx+args.batch_size]


            #sample random batch of validation data
            val_idx = valid_count // args.batch_size
            valid_index = np.random.choice(range(val_idx), 1)[0]
            val_img_batch = valid_imgs[args.batch_size*valid_index:args.batch_size*valid_index+args.batch_size]
            val_lab_batch = valid_labels[args.batch_size*valid_index:args.batch_size*valid_index+args.batch_size]
            val_box_batch = valid_boxes[args.batch_size*valid_index:args.batch_size*valid_index+args.batch_size]


            tr_batch = [load_image(path, args, is_training=True) for path in tr_img_batch]
            val_batch = [load_image(path, args, is_training=False) for path in val_img_batch]
            tr_batch = np.asarray(tr_batch)
            val_batch = np.asarray(val_batch)

            if len(tr_img_batch) < args.batch_size:
                break

            dictionary = {model.train_imgs:tr_batch,
                          model.train_labels:tr_lab_batch,
                          model.val_imgs:val_batch,
                          model.val_labels:val_lab_batch
                          }
            #Update Network
            summary, loss, acc, val_acc, val_top5, val_cam, _ = sess.run([all_summary, 
                                                                          model.loss, 
                                                                          model.acc, 
                                                                          model.val_acc,
                                                                          model.val_top5, 
                                                                          model.val_classmap, 
                                                                          optimizer],
                                                                        feed_dict=dictionary
                                                                        )
            writer.add_summary(summary, global_step)


            #calculate IOU and other stuff here
            pred_val_boxes = find_largest_box(val_cam)
            IOU_acc = calc_loc(val_box_batch, pred_val_boxes)


            print("Epoch [%d] Step [%d] Loss: [%.4f] Acc: [%.4f] \nVal(top-1): [%.4f] Val(top-5): [%.4f] IOU_acc: [%.4f]" % (epoch, idx, loss, acc, val_acc, val_top5, IOU_acc))
            global_step += 1
                
        
        saver.save(sess, args.checkpoints_path + "/model-"+str(epoch))
        print("Model saved at /model-" + str(epoch))

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

        print('Start Training...')
        train(args, sess, model)

main(args)

#Still Working....