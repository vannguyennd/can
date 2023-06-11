import tensorflow as tf
from tensorflow.contrib import rnn
import units
import os
import time
import units_start_end as u_se

opt_type = 'adam'
cell_type = 'lstm'
max_gradient_norm = 1.0
num_layers = 2

learning_rate = 0.001
training_steps = 20000

batch_size = 32
display_step = 150

num_input = 4  # input dimension
time_steps = 250  # sequence length of input, <-> (250 x 4 bytes), (125 x 8 bytes)
output_length = 250  # 250 for 4 bytes, 125 for 8 bytes
num_hidden = 256  # each hidden_layer's (hidden_state's) num of features
num_classes = 3  # E (end of function, 001), RI (right include, 000), RE (right exclude, 010)

# tf Graph input
X = tf.placeholder(tf.float32, shape=(None, time_steps, num_input))
Y = tf.placeholder(tf.int32, shape=(None, time_steps))

# Global step
global_step = tf.Variable(0, trainable=False)

# Define weights - Hidden layer weights => 2*n_hidden because of forward + backward cells
weights = {'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}


def bi_rnn(x, p_weights, p_biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, time_steps, n_input)
    # Required shape: 'time_steps' tensors list of shape (batch_size, num_input)
    # Unstack to get a list of 'time_steps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, time_steps, 1)

    # Define ls_tm cells with tensor_flow
    # Forward direction cell
    if cell_type == 'lstm':
        if num_layers > 1:
            fw_cell = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(num_hidden) for _ in range(num_layers)])
            # Backward direction cell
            bw_cell = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(num_hidden) for _ in range(num_layers)])
        else:
            fw_cell = rnn.LSTMCell(num_hidden)
            # Backward direction cell
            bw_cell = rnn.LSTMCell(num_hidden)
    elif cell_type == 'gru':
        if num_layers > 1:
            fw_cell = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(num_hidden) for _ in range(num_layers)])
            # Backward direction cell
            bw_cell = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(num_hidden) for _ in range(num_layers)])
        else:
            fw_cell = rnn.GRUCell(num_hidden)
            # Backward direction cell
            bw_cell = rnn.GRUCell(num_hidden)
    else:
        if num_layers > 1:
            fw_cell = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(num_hidden) for _ in range(num_layers)])
            # Backward direction cell
            bw_cell = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(num_hidden) for _ in range(num_layers)])
        else:
            fw_cell = rnn.BasicRNNCell(num_hidden)
            # Backward direction cell
            bw_cell = rnn.BasicRNNCell(num_hidden)

    # Get ls_tm cell output
    l_outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    l_outputs = tf.transpose(tf.stack(l_outputs, axis=0), perm=[1, 0, 2])
    l_outputs = tf.reshape(l_outputs, [-1, 2 * num_hidden])

    # Linear activation, using rnn inner loop last output
    return tf.matmul(l_outputs, p_weights['out']) + p_biases['out']


logits_ = tf.reshape(bi_rnn(X, weights, biases), [batch_size, time_steps, num_classes])

prediction = tf.nn.softmax(logits_)
correct_prediction = tf.argmax(prediction, 2)

loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=Y))

# Get all trainable variables
parameters = tf.trainable_variables()
# Calculate gradients
gradients = tf.gradients(loss_op, parameters)
# Clip gradients
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

if opt_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
elif opt_type == 'grad':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
else:
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

# Update operator
train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=global_step)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

dir_path = "./pe_x86/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

log_dir = dir_path + 'can_log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# get data
inputs, outputs = units.read_train_data()
inputs_valid_2, outputs_valid_2 = units.read_test_data()
inputs_t_valid, outputs_t_valid = units.read_valid_data()

log_dir_com = dir_path + 'compare_target_predict/'
if not os.path.exists(log_dir_com):
    os.makedirs(log_dir_com)

record_loss = open(log_dir_com + "/v_bi_rnn_loss.txt", "a+")

time_training = 0.0

# Start training
with tf.Session() as sess:

    # check whether we have the model trained or not
    check_point = tf.train.get_checkpoint_state(log_dir)
    if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
        print("load model parameters from %s" % check_point.model_checkpoint_path)
        saver.restore(sess, check_point.model_checkpoint_path)
    else:
        # if not, we start to initialize the model
        print("create the model with fresh parameters")
        sess.run(init)

    record_loss.write("opt: %s - cell: %s - lr: %f - h_size: %d - m_norm: %f \n" % (opt_type, cell_type, learning_rate,
                                                                                    num_hidden, max_gradient_norm))

    for step in range(training_steps):
        batch_x, batch_y = units.get_batch(inputs, outputs, batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, num_input))

        # Run optimization op (back_prop)
        time_start = time.time()
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        time_training += time.time() - time_start

        if step % display_step == 0 and step != 0 and step >= 19350:
            # we just need to consider/save the model and results for some last steps (epochs)

            # Calculate batch loss and accuracy
            loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})

            # for validation 2
            start = 0
            end = 32
            for i in range(int(inputs_valid_2.shape[0]) // batch_size):

                batch_x_valid_2, batch_y_valid_2 = inputs_valid_2[start:end], outputs_valid_2[start:end]
                batch_correct_prediction_2 = sess.run(correct_prediction,
                                                      feed_dict={X: batch_x_valid_2, Y: batch_y_valid_2})

                for idx_batch in range(batch_size):

                    compare_target_predict_2 = open(log_dir_com +
                                                    '/vi_compare_target_predict_valid_2_'
                                                    + str(step) + '.txt', 'a+')

                    result_valid_2 = 't: '
                    for j_value in batch_y_valid_2[idx_batch]:
                        result_valid_2 += str(j_value)
                    compare_target_predict_2.write(result_valid_2 + '\n')

                    result_predict_2 = 'p: '
                    for i_value in batch_correct_prediction_2[idx_batch]:
                        result_predict_2 += str(i_value)
                    compare_target_predict_2.write(result_predict_2 + '\n')

                start += 32
                end += 32

            end_recall_2 = u_se.result_target_predict_end_recall_2(dir_path, step)
            end_precision_2 = u_se.result_target_predict_end_precision_2(dir_path, step)

            if end_recall_2 == 0 and end_precision_2 == 0:
                end_predict_score_2 = 0
            else:
                end_predict_score_2 = (2 * end_recall_2 * end_precision_2) / (end_precision_2 + end_recall_2)

            start_recall_2 = u_se.result_target_predict_start_recall_2(dir_path, step)
            start_precision_2 = u_se.result_target_predict_start_precision_2(dir_path, step)

            if start_recall_2 == 0 and start_precision_2 == 0:
                start_predict_score_2 = 0
            else:
                start_predict_score_2 = (2 * start_recall_2 * start_precision_2) / (start_precision_2 + start_recall_2)

            # for training
            start = 0
            end = 32
            for i in range(int(inputs_t_valid.shape[0]) // batch_size):

                batch_x_t_valid, batch_y_t_valid = inputs_t_valid[start:end], outputs_t_valid[start:end]
                batch_correct_t__prediction = sess.run(correct_prediction, feed_dict={X: batch_x_t_valid,
                                                                                      Y: batch_y_t_valid})

                for idx_batch in range(batch_size):

                    compare_target_predict = open(log_dir_com
                                                  + '/vi_compare_target_predict_train.txt', 'a+')

                    result_valid = 't: '
                    for j_value in batch_y_t_valid[idx_batch]:
                        result_valid += str(j_value)
                    compare_target_predict.write(result_valid + '\n')

                    result_predict = 'p: '
                    for i_value in batch_correct_t__prediction[idx_batch]:
                        result_predict += str(i_value)
                    compare_target_predict.write(result_predict + '\n')

                start += 32
                end += 32

            train_end_recall = units.result_target_predict_t_end_recall(dir_path)
            train_end_precision = units.result_target_predict_t_end_precision(dir_path)

            if train_end_precision + train_end_recall == 0:
                train_en_score = 0
            else:
                train_en_score = (2 * train_end_recall * train_end_precision) / (train_end_precision + train_end_recall)

            os.remove(log_dir_com + '/vi_compare_target_predict_train.txt')

            print("step " + str(global_step.eval()) + ", time: " + str(time_training) + ", mini_batch_loss= " +
                  "{:.4f}".format(loss[0]) + ", train_end_score=" + "{:.4f}".format(train_en_score)
                  + ", predict_end_score_2=" + "{:.4f}".format(end_predict_score_2)
                  + ", predict_start_score_2=" + "{:.4f}".format(start_predict_score_2))

            record_loss.write("step: %d - time: %s - loss_train: %.4f - end_score_train: %.4f - "
                              "end_score_predict_2: %.4f - "
                              "start_score_predict_2: %.4f\n" % (step, str(time_training), loss[0], train_en_score,
                                                                 end_predict_score_2, start_predict_score_2))

            print("recall_end_2: %.4f - precision_end_2: %.4f - recall_start_2: %.4f - precision_start_2: %.4f - "
                  "end_score_predict_2: %.4f - "
                  "start_score_predict_2: %.4f\n" % (end_recall_2, end_precision_2, start_recall_2, start_precision_2,
                                                     end_predict_score_2, start_predict_score_2))

            record_loss.write("recall_end_2: %.4f - precision_end_2: %.4f - recall_start_2: %.4f - "
                              "precision_start_2: %.4f - "
                              "end_score_predict_2: %.4f - "
                              "start_score_predict_2: %.4f\n" % (end_recall_2, end_precision_2, start_recall_2,
                                                                 start_precision_2, end_predict_score_2,
                                                                 start_predict_score_2))

        if step % 150 == 0 and step != 0:
            check_point_path = os.path.join(log_dir, "can_rnn")
            saver.save(sess, check_point_path, global_step=step)

    record_loss.close()
    print("optimization finished!")
