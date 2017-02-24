import sys, os
# assumes running from main directory
sys.path.append(os.path.abspath("./"))

import time, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from testing import test
from shared_utils.data import *



class ROLO_TF:
    # Buttons
    validate = True
    validate_step = 2#500
    display_validate = True
    save_step = 50
    bidirectional = False
    display_step = 1
    restore_weights = True
    display_coords = False
    iou_with_ground_truth = True
    display_object_loss = True
    display_regu = False

    # Magic numbers
    learning_rate = 0.0001
    lamda = 1.0

    # Path
    rolo_weights_file = 'weights/rolo_weights.ckpt'
    rolo_current_save = 'weights/rolo_weights_temp.ckpt'

    # Vector for very small model
    len_feat = 1080
    # Vector for 4096 features model
    # len_feat = 4096
    len_predict = 6
    len_coord = 4
    len_vec = len_feat + len_predict

    # Batch
    nsteps = 3
    batchsize = 16
    n_iters = 180000
    batch_offset = 0

    # Data
    x = tf.placeholder("float32", [None, nsteps, len_vec])
    y = tf.placeholder("float32", [None, len_coord])
    # istate = tf.placeholder("float32", [None, 2*len_vec])
    list_batch_pairs = []

    # Initializing
    def __init__(self, kwargs):
        # TODO: do this the proper way **kwargs
        print("ROLO Initializing")
        if 'num_layers' in kwargs:
            self.number_of_layers = kwargs['num_layers']
        if 'bidirectional' in kwargs:
            self.bidirectional = kwargs['bidirectional']

        self.ROLO()

    # Routines: Network
    def LSTM(self, name,  _X):
        ''' shape: (batchsize, nsteps, len_vec) '''
        _X = tf.transpose(_X, [1, 0, 2])
        ''' shape: (nsteps, batchsize, len_vec) '''
        _X = tf.reshape(_X, [self.nsteps * self.batchsize, self.len_vec])
        # import pdb; pdb.set_trace()
        ''' shape: n_steps * (batchsize, len_vec) '''
        _X = tf.split(_X, num_or_size_splits=self.nsteps, axis=0)


        cell = tf.contrib.rnn.LSTMCell(self.len_vec, self.len_vec, state_is_tuple = False)

        # TODO: use dropout???
        # cell = DropoutWrapper(cell, output_keep_prob=dropout)

        lstm_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.number_of_layers, state_is_tuple=False)

        state = lstm_cell.zero_state(self.batchsize, tf.float32)

        pred, output_state = tf.contrib.rnn.static_rnn(lstm_cell, _X, state, dtype=tf.float32)

        batch_pred_feats = pred[0][:, 0:self.len_feat]
        batch_pred_coords = pred[0][:, self.len_feat:self.len_feat+self.len_coord]
        batch_pred_confs = pred[0][:, self.len_feat+self.len_coord]
        return batch_pred_feats, batch_pred_coords, batch_pred_confs, output_state


    def iou(self, boxes1, boxes2):
        """
        Note: Modified from https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_net.py
        calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        # import pdb; pdb.set_trace()
        boxes1 = tf.stack([boxes1[:,0] - boxes1[:,2] / 2, boxes1[:,1] - boxes1[:,3] / 2,
                          boxes1[:,0] + boxes1[:,2] / 2, boxes1[:,1] + boxes1[:,3] / 2])
        # # boxes1 = tf.transpose(boxes1)
        boxes2 =  tf.stack([boxes2[:,0] - boxes2[:,2] / 2, boxes2[:,1] - boxes2[:,3] / 2,
                          boxes2[:,0] + boxes2[:,2] / 2, boxes2[:,1] + boxes2[:,3] / 2])
        # Assumes boxes 1 is the
        # boxes1[:,0] = boxes1[:,0] - boxes1[:,2]/2
        # boxes1[:,1] = boxes1[:,1] - boxes1[:,3]/2

        #calculate the left up point
        # import pdb; pdb.set_trace()
        lu = tf.maximum(boxes1[0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[2:], boxes2[2:])

        #intersection
        intersection = rd - lu

        inter_square = tf.multiply(intersection[0],intersection[1])

        mask = tf.cast(intersection[0] > 0, tf.float32) * tf.cast(intersection[1] > 0, tf.float32)

        inter_square = tf.multiply(mask,inter_square)

        #calculate the boxs1 square and boxs2 square
        square1 = tf.multiply((boxes1[2] - boxes1[0]) ,(boxes1[3] - boxes1[1]))
        square2 = tf.multiply((boxes2[2] - boxes2[0]),(boxes2[3] - boxes2[1]))

        return inter_square/(square1 + square2 - inter_square + 1e-6)

    # Routines: Train & Test
    def train(self):
        ''' Network '''
        batch_pred_feats, batch_pred_coords, batch_pred_confs, self.final_state = self.LSTM('lstm', self.x)

        ''' Loss: L2 '''
        loss = tf.reduce_mean(tf.square(self.y - batch_pred_coords)) * 100

        iou_predict_truth = self.iou(batch_pred_coords, self.y[:,0:4])

        ''' confidence loss'''

        object_loss = tf.reduce_mean(tf.nn.l2_loss((batch_pred_confs - iou_predict_truth))) * 100
        # ave_iou = tf.reduce_mean(iou_predict_truth)
        # noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale


        ''' regularization term: L2 '''
        regularization_term = tf.reduce_mean(tf.square(self.x[:, self.nsteps-1, 0:self.len_feat] - batch_pred_feats)) * 100

        minimize_iou = (1.0 - tf.reduce_mean(iou_predict_truth)) * 100

        ''' Optimizer '''
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss + object_loss + minimize_iou + self.lamda * regularization_term) # Adam Optimizer

        ''' Summary for tensorboard analysis '''
        dataset_loss = -1
        dataset_loss_best = 100
        test_writer = tf.summary.FileWriter('summary/test')
        tf.summary.scalar('dataset_loss', dataset_loss)
        summary_op = tf.summary.merge_all()

        ''' Initializing the variables '''
        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        batch_states = np.zeros((self.batchsize, 2*self.len_vec))

        # TODO: make this a command line argument, etc.
        # training set loader
        # batch_loader = BatchLoader("./DATA/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["Human3","Human4", "Human8", "Human9"])
        batch_loader = BatchLoader("./DATA/TRAINING/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["GOPR0005","GOPR0006","GOPR0008","GOPR0008_2","GOPR0009","GOPR0009_2","GOPR0010","GOPR0011","GOPR0012","GOPR0013","GOPR0014","GOPR0015","GOPR0016","MVI_8607","MVI_8609","MVI_8610","MVI_8612","MVI_8614","MVI_8615","MVI_8616"])
        # Validation set loader
        # validation_set_loader = BatchLoader("./VALID/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["Human6" ,"Human7"])
        validation_set_loader = BatchLoader("./DATA/VALID/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["bbd_2017__2017-01-09-21-40-02_cam_flimage_raw","bbd_2017__2017-01-09-21-44-31_cam_flimage_raw","bbd_2017__2017-01-09-21-48-46_cam_flimage_raw","bbd_2017__2017-01-10-16-07-49_cam_flimage_raw","bbd_2017__2017-01-10-16-21-01_cam_flimage_raw","bbd_2017__2017-01-10-16-31-57_cam_flimage_raw","bbd_2017__2017-01-10-21-43-03_cam_flimage_raw","bbd_2017__2017-01-11-20-21-32_cam_flimage_raw","bbd_2017__2017-01-11-21-02-37_cam_flimage_raw"])

        ''' Launch the graph '''
        with tf.Session() as sess:
            if self.restore_weights == True and os.path.isfile(self.rolo_current_save):
                sess.run(init)
                self.saver.restore(sess, self.rolo_current_save)
                print("Weight loaded, finetuning")
            else:
                sess.run(init)
                print("Training from scratch")


            for self.iter_id in range(self.n_iters):
                ''' Load training data & ground truth '''
                batch_id = self.iter_id - self.batch_offset

                batch_xs, batch_ys, _ = batch_loader.load_batch(batch_id)

                # ''' Reshape data '''
                # batch_xs = np.reshape(batch_xs, [self.batchsize, self.nsteps, self.len_vec])
                # batch_ys = np.reshape(batch_ys, [self.batchsize, 4])

                ''' Update weights by back-propagation '''
                # import pdb; pdb.set_trace()
                sess.run(optimizer, feed_dict={self.x: batch_xs,
                                               self.y: batch_ys})

                if self.iter_id % self.display_step == 0:
                    ''' Calculate batch loss '''
                    batch_loss = sess.run(loss,
                                          feed_dict={self.x: batch_xs,
                                                     self.y: batch_ys})
                    print("Batch loss for iteration %d: %.3f" % (self.iter_id, batch_loss))
                if self.display_object_loss:
                    ''' Calculate batch object loss '''
                    batch_o_loss = sess.run(object_loss,
                                          feed_dict={self.x: batch_xs,
                                                     self.y: batch_ys})
                    print("Object loss for iteration %d: %.3f" % (self.iter_id, batch_o_loss))

                if self.iou_with_ground_truth:
                    ''' Calculate batch object loss '''
                    batch_o_loss = sess.run(tf.reduce_mean(iou_predict_truth),
                                          feed_dict={self.x: batch_xs,
                                                     self.y: batch_ys})
                    print("Average with ground for iteration %d: %.3f" % (self.iter_id, batch_o_loss))

                if self.display_regu is True:
                    ''' Caculate regularization term'''
                    batch_regularization = sess.run(regularization_term,
                                                    feed_dict={self.x: batch_xs,
                                                               self.y: batch_ys})
                    print("Batch regu for iteration %d: %.3f" % (self.iter_id, batch_regularization))

                if self.display_coords is True:
                    ''' Caculate predicted coordinates '''
                    coords_predict = sess.run(batch_pred_coords,
                                              feed_dict={self.x: batch_xs,
                                                         self.y: batch_ys})
                    print("predicted coords:" + str(coords_predict[0]))
                    print("ground truth coords:" + str(batch_ys[0]))

                ''' Save model '''
                if self.iter_id % self.save_step == 1:
                    self.saver.save(sess, self.rolo_current_save)
                    print("\n Model saved in file: %s" % self.rolo_current_save)

                ''' Validation '''
                if self.validate == True and self.iter_id % self.validate_step == 0 and self.iter_id > 0:
                    # Run validation set

                    dataset_loss = self.test(sess, loss, validation_set_loader, batch_pred_feats, batch_pred_coords, batch_pred_confs, self.final_state)

                    ''' Early-stop regularization '''
                    if dataset_loss <= dataset_loss_best:
                        dataset_loss_best = dataset_loss
                        self.saver.save(sess, self.rolo_weights_file)
                        print("\n Better Model saved in file: %s" % self.rolo_weights_file)

                    ''' Write summary for tensorboard '''
                    summary = sess.run(summary_op, feed_dict={self.x: batch_xs,
                                                              self.y: batch_ys})
                    test_writer.add_summary(summary, self.iter_id)
        return


    def test(self, sess, loss, batch_loader, batch_pred_feats, batch_pred_coords, batch_pred_confs, final_state):
        loss_dataset_total = 0
        #TODO: put outputs somewhere
        # batch_pred_feats, batch_pred_coords, batch_pred_confs, self.final_state = self.LSTM('lstm', self.x, self.istate)
        batch_states = np.zeros((self.batchsize, 2*self.len_vec))
        iou_predict_truth = tf.reduce_mean(self.iou(batch_pred_coords, self.y[:,0:4]))

        output_path = os.path.join('rolo_loc_test/')
        iou_averages = []
        for batch_id in range(len(batch_loader.batches)):
            print("Validation batch %d" % batch_id)
            xs, ys, im_paths = batch_loader.load_batch(batch_id)
            loss_seq_total = 0

            init_state_zeros = np.zeros((len(xs), 2*xs[0].shape[-1]))

            pred_location, pred_confs = sess.run([batch_pred_coords, batch_pred_confs],feed_dict={self.x: xs, self.y: ys})
            # for i in range(len(pred_confs)):
            for i, loc in enumerate(pred_location):
                img = cv2.imread(im_paths[i])
                width, height = img.shape[1::-1]
                # print pred_location[i]
                # print ys[i]
                # print xs[i][2][self.len_feat+1:-1]
                img_result = debug_3_locations(img, locations_normal(width, height, ys[i]), locations_normal(width, height, xs[i][2][self.len_feat+1:-1]), locations_normal(width, height, pred_location[i]))
                cv2.imwrite('./results/%d_%d.jpg' %(batch_id, i), img_result)

                # print("predicted")
                # print(pred_location[i])
                # print("gold")
                # print(ys[i])
                # print("confidence")
                # print(pred_confs[i])
                # print("numpy iou")
                # print(iou(pred_location[i], ys[i]))

            # TODO: confidence interval to predict whether we have a box or not

            # TODO: output image with bounding box, see:

            # https://github.com/Guanghan/ROLO/blob/6612007e35edb73dac734e7a4dac2cd4c1dca6c1/update/utils/utils_draw_coord.py

            init_state = init_state_zeros

            batch_loss = sess.run(loss,
                                  feed_dict={self.x: xs,
                                             self.y: ys})
            iou_ground_truth = sess.run(iou_predict_truth,
                                  feed_dict={self.x: xs,
                                             self.y: ys})
            loss_seq_total += batch_loss

            # TODO: only do this if we have a prediction
            # iou_ground_truth_total += iou_ground_truth

            loss_seq_avg = loss_seq_total / xs.shape[0]
            iou_seq_avg = iou_ground_truth

            # print "Avg loss for " + sequence_name + ": " + str(loss_seq_avg)
            loss_dataset_total += loss_seq_avg
            iou_averages.append(iou_seq_avg)

        print('Total loss of Dataset: %f \n', loss_dataset_total)
        print('Average iou with ground truth: %f \n', np.mean(iou_averages))
        return loss_dataset_total

    def ROLO(self):
        print("Initializing ROLO")
        self.train()
        print("Training Completed")

'''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1, help="number of layers of LSTM to use, defaults to 1")
    parser.add_argument("-b", type=bool, default=False, help="Whether to use a bidirectional LSTM")
    args = parser.parse_args()

    ROLO_TF({'num_layers' : args.n, "bidirectional" : args.b})

if __name__ == "__main__":
    main(' ')
