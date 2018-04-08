import csv
import tensorflow as tf
from config import *
from network import *
from ops import *
from tqdm import tqdm

def test(args, sess, model):
    print "Preparing test data..."
    paths = os.path.join(args.data, "test")
    # img_paths = glob(paths)
    img_paths = read_csv(args.data)
    data_count = len(img_paths)

    #saver
    saver = tf.train.Saver()        
    last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print "Loaded model file from " + ckpt_name
    print "Making predictions..."
    with open('./data/result.csv', 'w') as csvfile:
        fieldnames = ['id', 'landmarks']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        batch_idx = data_count // args.batch_size
        writer.writeheader()
        for idx in tqdm(range(batch_idx)):
            path_batch = img_paths[idx*args.batch_size:idx*args.batch_size+args.batch_size]
            img_batch = [read_test_img(args, path) for path in path_batch]
            
            pred, conf = sess.run([model.test_pred, model.test_conf], feed_dict={model.test_imgs:img_batch})
            
            for i in range(0,len(pred)):
                id_ = path_batch[i].split('/')[-1].split('.')[0]
                conf_ = conf[i][pred[i]]
                
                writer.writerow({'id': id_, 'landmarks': ("%d %.2f")%(pred[i], conf_)})        

        ##for Remaining data
        remaining = data_count - batch_idx*args.batch_size
        last_batch = img_paths[-remaining:]
        last_batch = np.tile(last_batch, [args.batch_size/remaining, 1, 1, 1])
        pred, conf = sess.run([model.test_pred, model.test_conf], feed_dict={model.test_imgs:last_batch})
        for i in range(0,len(pred[:remaining])):
            id_ = path_batch[i].split('/')[-1].split('.')[0]
            conf_ = conf[i][pred[i]]
            writer.writerow({'id': id_, 'landmarks': ("%d %.2f")%(pred[i], conf_)})         

    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network(args)

        print 'Start Testing...'
        test(args, sess, model)

main(args)

#Still Working....