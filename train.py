import numpy as np
import tf_models
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.backend import learning_phase
from tensorflow.contrib.keras.python.keras.layers import concatenate, Conv3D
from nibabel import load as load_nii
import os
import argparse
import keras


def parse_inputs():

    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('-r', '--root-path', dest='root_path', default='/media/lele/Data/spie/Brats17TrainingData/HGG')
    parser.add_argument('-sp', '--save-path', dest='save_path', default='dense24_correction')
    parser.add_argument('-lp', '--load-path', dest='load_path', default='dense24_correction')
    parser.add_argument('-ow', '--offset-width', dest='offset_w', type=int, default=12)
    parser.add_argument('-oh', '--offset-height', dest='offset_h', type=int, default=12)
    parser.add_argument('-oc', '--offset-channel', dest='offset_c', nargs='+', type=int, default=12)
    parser.add_argument('-ws', '--width-size', dest='wsize', type=int, default=38)
    parser.add_argument('-hs', '--height-size', dest='hsize', type=int, default=38)
    parser.add_argument('-cs', '--channel-size', dest='csize', type=int, default=38)
    parser.add_argument('-ps', '--pred-size', dest='psize', type=int, default=12)
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=2)
    parser.add_argument('-e', '--num-epochs', dest='num_epochs', type=int, default=5)
    parser.add_argument('-c', '--continue-training', dest='continue_training', type=bool, default=False)
    parser.add_argument('-mn', '--model_name', dest='model_name', type=str, default='dense24')
    parser.add_argument('-nc', '--n4correction', dest='correction', type=bool, default=False)
    parser.add_argument('-gpu', '--gpu_id', dest='gpu_id', type=str, default='0')
    return vars(parser.parse_args())

options = parse_inputs()

os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_id']
def acc_tf(y_pred, y_true):
    correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, -1), tf.int32), tf.cast(tf.argmax(y_true, -1), tf.int32))
    return 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_patches_3d(data, labels, centers, hsize, wsize, csize, psize, preprocess=True):
    """

    :param data: 4D nparray (h, w, c, ?)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """
    patches_x, patches_y = [], []
    offset_p = (hsize - psize) / 2
    for i in range(len(centers[0])):
        h, w, c = centers[0, i], centers[1, i], centers[2, i]
        h_beg = min(max(0, h - hsize / 2), 240 - hsize)
        w_beg = min(max(0, w - wsize / 2), 240 - wsize)
        c_beg = min(max(0, c - csize / 2), 155 - csize)
        ph_beg = h_beg + offset_p
        pw_beg = w_beg + offset_p
        pc_beg = c_beg + offset_p
        vox = data[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize, :]
        vox_labels = labels[ph_beg:ph_beg + psize, pw_beg:pw_beg + psize, pc_beg:pc_beg + psize]
        patches_x.append(vox)
        patches_y.append(vox_labels)
    return np.array(patches_x), np.array(patches_y)


def positive_ratio(x):
    return float(np.sum(np.greater(x, 0))) / np.prod(x.shape)


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def segmentation_loss(y_true, y_pred, n_classes):
    y_true = tf.reshape(y_true, (-1, n_classes))
    y_pred = tf.reshape(y_pred, (-1, n_classes))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))


def vox_preprocess(vox):
    vox_shape = vox.shape
    vox = np.reshape(vox, (-1, vox_shape[-1]))
    vox = scale(vox, axis=0)
    return np.reshape(vox, vox_shape)


def one_hot(y, num_classes):
    y_ = np.zeros([len(y), num_classes])
    y_[np.arange(len(y)), y] = 1
    return y_


def dice_coef_np(y_true, y_pred, num_classes):
    """

    :param y_true: sparse labels
    :param y_pred: sparse labels
    :param num_classes: number of classes
    :return:
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_true = y_true.flatten()
    y_true = one_hot(y_true, num_classes)
    y_pred = y_pred.flatten()
    y_pred = one_hot(y_pred, num_classes)
    intersection = np.sum(y_true * y_pred, axis=0)
    return (2. * intersection) / (np.sum(y_true, axis=0) + np.sum(y_pred, axis=0))


def vox_generator(all_files, n_pos, n_neg,correction= False):
    path = options['root_path']
    while 1:
        for file in all_files:
            if correction:
                flair = load_nii(os.path.join(path, file, file + '_flair_corrected.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2_corrected.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1_corrected.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce_corrected.nii.gz')).get_data()
            else:

                flair = load_nii(os.path.join(path, file, file + '_flair.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce.nii.gz')).get_data()

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = load_nii(os.path.join(path, file, file+'_seg.nii.gz')).get_data()

            foreground = np.array(np.where(labels > 0))
            background = np.array(np.where((labels == 0) & (flair > 0)))

            # n_pos = int(foreground.shape[1] * discount)
            foreground = foreground[:, np.random.permutation(foreground.shape[1])[:n_pos]]
            background = background[:, np.random.permutation(background.shape[1])[:n_neg]]

            centers = np.concatenate((foreground, background), axis=1)
            centers = centers[:, np.random.permutation(n_neg+n_pos)]

            yield data_norm, labels, centers


def label_transform(y, nlabels):
    return [
            keras.utils.to_categorical(np.copy(y).astype(dtype=np.bool),
                                       num_classes=2).reshape([y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2]),

            keras.utils.to_categorical(y,
                                       num_classes=nlabels).reshape([y.shape[0], y.shape[1], y.shape[2], y.shape[3], nlabels])
            ]


def train():
    NUM_EPOCHS = options['num_epochs']
    LOAD_PATH = options['load_path']
    SAVE_PATH = options['save_path']
    PSIZE = options['psize']
    HSIZE = options['hsize']
    WSIZE = options['wsize']
    CSIZE = options['csize']
    model_name= options['model_name']
    BATCH_SIZE = options['batch_size']
    continue_training = options['continue_training']

    files = []
    num_labels = 5
    with open('train.txt') as f:
        for line in f:
            files.append(line[:-1])
    print '%d training samples' % len(files)

    flair_t2_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))
    t1_t1ce_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))
    flair_t2_gt_node = tf.placeholder(dtype=tf.int32, shape=(None, PSIZE, PSIZE, PSIZE, 2))
    t1_t1ce_gt_node = tf.placeholder(dtype=tf.int32, shape=(None, PSIZE, PSIZE, PSIZE, 5))

    if model_name == 'dense48':
        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=t1_t1ce_node, name='t1')
    elif model_name == 'no_dense':

        flair_t2_15, flair_t2_27 = tf_models.PlainCounterpart(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.PlainCounterpart(input=t1_t1ce_node, name='t1')

    elif model_name == 'dense24':

        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat(input=t1_t1ce_node, name='t1')
    else:
        print' No such model name '

    t1_t1ce_15 = concatenate([t1_t1ce_15, flair_t2_15])
    t1_t1ce_27 = concatenate([t1_t1ce_27, flair_t2_27])

    flair_t2_15 = Conv3D(2, kernel_size=1, strides=1, padding='same', name='flair_t2_15_cls')(flair_t2_15)
    flair_t2_27 = Conv3D(2, kernel_size=1, strides=1, padding='same', name='flair_t2_27_cls')(flair_t2_27)
    t1_t1ce_15 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_15_cls')(t1_t1ce_15)
    t1_t1ce_27 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_27_cls')(t1_t1ce_27)

    flair_t2_score = flair_t2_15[:, 13:25, 13:25, 13:25, :] + \
                     flair_t2_27[:, 13:25, 13:25, 13:25, :]

    t1_t1ce_score = t1_t1ce_15[:, 13:25, 13:25, 13:25, :] + \
                    t1_t1ce_27[:, 13:25, 13:25, 13:25, :]

    loss = segmentation_loss(flair_t2_gt_node, flair_t2_score, 2) + \
           segmentation_loss(t1_t1ce_gt_node, t1_t1ce_score, 5)

    acc_flair_t2 = acc_tf(y_pred=flair_t2_score, y_true=flair_t2_gt_node)
    acc_t1_t1ce = acc_tf(y_pred=t1_t1ce_score, y_true=t1_t1ce_gt_node)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)

    saver = tf.train.Saver(max_to_keep=15)
    data_gen_train = vox_generator(all_files=files, n_pos=200, n_neg=200,correction = options['correction'])

    with tf.Session() as sess:
        if continue_training:
            saver.restore(sess, LOAD_PATH)
        else:
            sess.run(tf.global_variables_initializer())
        for ei in range(NUM_EPOCHS):
            for pi in range(len(files)):
                acc_pi, loss_pi = [], []
                data, labels, centers = data_gen_train.next()
                n_batches = int(np.ceil(float(centers.shape[1]) / BATCH_SIZE))
                for nb in range(n_batches):
                    offset_batch = min(nb * BATCH_SIZE, centers.shape[1] - BATCH_SIZE)
                    data_batch, label_batch = get_patches_3d(data, labels, centers[:, offset_batch:offset_batch + BATCH_SIZE], HSIZE, WSIZE, CSIZE, PSIZE, False)
                    label_batch = label_transform(label_batch, 5)
                    _, l, acc_ft, acc_t1c = sess.run(fetches=[optimizer, loss, acc_flair_t2, acc_t1_t1ce],
                                                   feed_dict={flair_t2_node: data_batch[:, :, :, :, :2],
                                                              t1_t1ce_node: data_batch[:, :, :, :, 2:],
                                                              flair_t2_gt_node: label_batch[0],
                                                              t1_t1ce_gt_node: label_batch[1],
                                                              learning_phase(): 1})
                    acc_pi.append([acc_ft, acc_t1c])
                    loss_pi.append(l)
                    n_pos_sum = np.sum(np.reshape(label_batch[0], (-1, 2)), axis=0)
                    print 'epoch-patient: %d, %d, iter: %d-%d, p%%: %.4f, loss: %.4f, acc_flair_t2: %.2f%%, acc_t1_t1ce: %.2f%%' % \
                          (ei + 1, pi + 1, nb + 1, n_batches, n_pos_sum[1]/float(np.sum(n_pos_sum)), l, acc_ft, acc_t1c)

                print 'patient loss: %.4f, patient acc: %.4f' % (np.mean(loss_pi), np.mean(acc_pi))

            saver.save(sess, SAVE_PATH, global_step=ei)
            print 'model saved'


if __name__ == '__main__':
    
    train()