import sys, os
# assumes running from main directory
sys.path.append(os.path.abspath("./"))

import time, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import LSTMCell

# from testing import test
from shared_utils.data import BatchLoader


class ROLO_TF:
    # Buttons
    validate = False
    validate_step = 1000
    display_validate = True
    save_step = 1000
    display_step = 1
    restore_weights = True
    display_coords = False
    display_regu = False

    # Magic numbers
    learning_rate = 0.0001
    lamda = 1.0

    # Path
    rolo_weights_file = 'weights/rolo_weights.ckpt'
    rolo_current_save = 'weights/rolo_weights_temp.ckpt'

    # Vector
    len_feat = 4096
    len_predict = 6
    len_coord = 4
    len_vec = 4102

    # Batch
    nsteps = 3
    batchsize = 16
    n_iters = 180000
    batch_offset = 0

    # Data
    x = tf.placeholder("float32", [None, nsteps, len_vec])
    y = tf.placeholder("float32", [None, len_coord])
    istate = tf.placeholder("float32", [None, 2*len_vec])
    list_batch_pairs = []

    # Initializing
    def __init__(self, argvs = []):
        print("ROLO Initializing")
        self.ROLO()

    # Routines: Network
    def LSTM(self, name,  _X, _istate):
        ''' shape: (batchsize, nsteps, len_vec) '''
        _X = tf.transpose(_X, [1, 0, 2])
        ''' shape: (nsteps, batchsize, len_vec) '''
        _X = tf.reshape(_X, [self.nsteps * self.batchsize, self.len_vec])
        # import pdb; pdb.set_trace()
        ''' shape: n_steps * (batchsize, len_vec) '''
        _X = tf.split(_X, num_or_size_splits=self.nsteps, axis=0)

        lstm_cell = tf.contrib.rnn.LSTMCell(self.len_vec, self.len_vec, state_is_tuple = False)
        state = _istate
        # for step in range(self.nsteps):
        pred, output_state = tf.contrib.rnn.static_rnn(lstm_cell, _X, state, dtype=tf.float32)
        # tf.get_variable_scope().reuse_variables()
            # if step == 0:   output_state = state

        batch_pred_feats = pred[0][:, 0:4096]
        batch_pred_coords = pred[0][:, 4097:4101]
        return batch_pred_feats, batch_pred_coords, output_state


    # Routines: Train & Test
    def train(self):
        ''' Network '''
        batch_pred_feats, batch_pred_coords, self.final_state = self.LSTM('lstm', self.x, self.istate)

        ''' Loss: L2 '''
        loss = tf.reduce_mean(tf.square(self.y - batch_pred_coords)) * 100

        ''' regularization term: L2 '''
        regularization_term = tf.reduce_mean(tf.square(self.x[:, self.nsteps-1, 0:4096] - batch_pred_feats)) * 100

        ''' Optimizer '''
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss  + self.lamda * regularization_term) # Adam Optimizer

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
        batch_loader = BatchLoader("./DATA/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["Human3","Human4", "Human8", "Human9"])
        # Validation set loader
        validation_set_loader = BatchLoader("./VALID/", seq_len=self.nsteps, batch_size=self.batchsize, step_size=1, folders_to_use=["Human6" ,"Human7"])

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

                batch_xs, batch_ys = batch_loader.load_batch(batch_id)
                # import pdb; pdb.set_trace()


                # ''' Reshape data '''
                # batch_xs = np.reshape(batch_xs, [self.batchsize, self.nsteps, self.len_vec])
                # batch_ys = np.reshape(batch_ys, [self.batchsize, 4])

                ''' Update weights by back-propagation '''
                # import pdb; pdb.set_trace()
                sess.run(optimizer, feed_dict={self.x: batch_xs,
                                               self.y: batch_ys,
                                               self.istate: batch_states})

                if self.iter_id % self.display_step == 0:
                    ''' Calculate batch loss '''
                    batch_loss = sess.run(loss,
                                          feed_dict={self.x: batch_xs,
                                                     self.y: batch_ys,
                                                     self.istate: batch_states})
                    print("Batch loss for iteration %d: %.3f" % (self.iter_id, batch_loss))

                if self.display_regu is True:
                    ''' Caculate regularization term'''
                    batch_regularization = sess.run(regularization_term,
                                                    feed_dict={self.x: batch_xs,
                                                               self.y: batch_ys,
                                                               self.istate: batch_states})
                    print("Batch regu for iteration %d: %.3f" % (self.iter_id, batch_regularization))

                if self.display_coords is True:
                    ''' Caculate predicted coordinates '''
                    coords_predict = sess.run(batch_pred_coords,
                                              feed_dict={self.x: batch_xs,
                                                         self.y: batch_ys,
                                                         self.istate: batch_states})
                    print("predicted coords:" + str(coords_predict[0]))
                    print("ground truth coords:" + str(batch_ys[0]))

                ''' Save model '''
                if self.iter_id % self.save_step == 1:
                    self.saver.save(sess, self.rolo_current_save)
                    print("\n Model saved in file: %s" % self.rolo_current_save)

                ''' Validation '''
                if self.validate == True and self.iter_id % self.validate_step == 0:
                    # Run validation set

                    dataset_loss = self.test(sess, loss, validation_set_loader)

                    ''' Early-stop regularization '''
                    if dataset_loss <= dataset_loss_best:
                        dataset_loss_best = dataset_loss
                        self.saver.save(sess, self.rolo_weights_file)
                        print("\n Better Model saved in file: %s" % self.rolo_weights_file)

                    ''' Write summary for tensorboard '''
                    summary = sess.run(summary_op, feed_dict={self.x: batch_xs,
                                                              self.y: batch_ys,
                                                              self.istate: batch_states})
                    test_writer.add_summary(summary, self.iter_id)
        return


    def test(self, sess, loss, batch_loader):
        loss_dataset_total = 0
        #TODO: put outputs somewhere
        batch_pred_feats, batch_pred_coords, self.final_state = self.LSTM('lstm', self.x, self.istate)

        output_path = os.path.join('rolo_loc_test/')
        for batch_id in range(len(batch_loader.batches)):
            xs, ys = batch_loader.load_batch(batch_id)
            loss_seq_total = 0

            init_state_zeros = np.zeros((len(xs), 2*xs[0].shape[-1]))

            pred_location = sess.run(batch_pred_coords,feed_dict={self.x: xs, self.y: ys, self.istate: batch_states})
            # TODO: output rolo prediction

            # TODO: should do a consecutive video? (it will already do this by default with the staggered steps)

            # TODO: output image with bounding box, see:
            # https://github.com/Guanghan/ROLO/blob/6612007e35edb73dac734e7a4dac2cd4c1dca6c1/update/utils/utils_draw_coord.py

            init_state = init_state_zeros

            batch_loss = sess.run(loss,
                                  feed_dict={self.x: xs,
                                             self.y: ys,
                                             self.istate: init_state})
            loss_seq_total += batch_loss

            loss_seq_avg = loss_seq_total / xs.shape[0]
            # print "Avg loss for " + sequence_name + ": " + str(loss_seq_avg)
            loss_dataset_total += loss_seq_avg

        print('Total loss of Dataset: %f \n', loss_dataset_total)
        return loss_dataset_total

    def ROLO(self):
        print("Initializing ROLO")
        self.train()
        print("Training Completed")

'''----------------------------------------main-----------------------------------------------------'''
def main(argvs):
    ROLO_TF(argvs)

if __name__ == "__main__":
    main(' ')
