import numpy as np

train_data = "./data_real/train_pe_x86.txt"
test_data = "./data_real/test_pe_x86.txt"
valid_data = "./data_real/valid_pe_x86.txt"


def read_train_data():

    data_path = train_data

    with open(data_path, 'r') as file:
        recs = file.readlines()
        l_inputs = []
        l_outputs = []

        for idx, rec in enumerate(recs):

            inp, out = rec[:-2].split(' output ')
            inp = inp.split(' ')
            out = out.split(' ')

            in_put = []
            for t in inp:
                in_put.append(float(t))

            in_put = np.array(in_put).reshape([-1, 4])
            l_inputs.append(in_put)

            output = []
            for i, i_out in enumerate(out):
                output.append(int(i_out))
            output = np.array(output).reshape([-1, 4])

            n_output = ''
            for i in output:
                n_output += str(i[0]) + ' '

            n_output = n_output[:-1].split(' ')
            n_output = np.array(n_output)

            l_outputs.append(n_output)

        l_inputs = np.stack(l_inputs)
        l_outputs = np.stack(l_outputs)

        print("Load inputs: " + str(l_inputs.shape))
        print("Load outputs:    " + str(l_outputs.shape))

        return l_inputs, l_outputs


def read_test_data():

    data_path = test_data

    with open(data_path, 'r') as file:
        recs = file.readlines()
        l_inputs = []
        l_outputs = []

        for idx, rec in enumerate(recs):

            inp, out = rec[:-2].split(' output ')
            inp = inp.split(' ')
            out = out.split(' ')

            in_put = []
            for t in inp:
                in_put.append(float(t))

            in_put = np.array(in_put).reshape([-1, 4])
            l_inputs.append(in_put)

            output = []
            for i, i_out in enumerate(out):
                output.append(int(i_out))
            output = np.array(output).reshape([-1, 4])

            n_output = ''
            for i in output:
                n_output += str(i[0]) + ' '

            n_output = n_output[:-1].split(' ')
            n_output = np.array(n_output)

            l_outputs.append(n_output)

        l_inputs = np.stack(l_inputs)
        l_outputs = np.stack(l_outputs)

        print("Load inputs: " + str(l_inputs.shape))
        print("Load outputs:    " + str(l_outputs.shape))

        return l_inputs, l_outputs


def read_valid_data():

    data_path = valid_data

    with open(data_path, 'r') as file:
        recs = file.readlines()
        l_inputs = []
        l_outputs = []

        for idx, rec in enumerate(recs):

            print(idx)
            inp, out = rec[:-2].split(' output ')
            inp = inp.split(' ')
            out = out.split(' ')

            in_put = []
            for t in inp:
                in_put.append(float(t))

            in_put = np.array(in_put).reshape([-1, 4])
            l_inputs.append(in_put)

            output = []
            for i, i_out in enumerate(out):
                output.append(int(i_out))
            output = np.array(output).reshape([-1, 4])

            n_output = ''
            for i in output:
                n_output += str(i[0]) + ' '

            n_output = n_output[:-1].split(' ')
            n_output = np.array(n_output)

            l_outputs.append(n_output)

        l_inputs = np.stack(l_inputs)
        l_outputs = np.stack(l_outputs)

        print("Load inputs: " + str(l_inputs.shape))
        print("Load outputs:    " + str(l_outputs.shape))

        return l_inputs, l_outputs


def get_batch(p_inputs, p_outputs, batch_size):

    data_size = p_inputs.shape[0]
    sample_values = np.random.choice(data_size, batch_size, replace=True)

    return p_inputs[sample_values], p_outputs[sample_values]


# this function for end recall
def count_correct_end_recall(p_i_rec, j_rec):
    c_true = 0
    c_false = 0
    for idx, rec in enumerate(p_i_rec):
        if int(rec) == 2:
            if int(j_rec[idx]) == 2:
                c_true += 1
            else:
                c_false += 1
    return c_true, c_false


# this function for end precision
def count_correct_end_precision(p_i_rec, j_rec):
    c_true = 0
    c_false = 0
    for idx, rec in enumerate(j_rec):
        if int(rec) == 2:
            if int(p_i_rec[idx]) == 2:
                c_true += 1
            else:
                c_false += 1
    return c_true, c_false


# for computing end_recall
def result_target_predict_t_end_recall(dir_path):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_train.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    count_true_sum = 0
    count_false_sum = 0

    for i, i_rec in enumerate(recs):
        if i % 2 == 0:

            i_content = recs[i].split('t: ')[1].split('\n')[0]
            in_content = recs[i + 1].split('p: ')[1].split('\n')[0]

            count_true, count_false = count_correct_end_recall(i_content, in_content)
            count_true_sum += count_true
            count_false_sum += count_false

    if count_true_sum + count_false_sum == 0:
        return 0
    else:
        return count_true_sum / (count_true_sum + count_false_sum)


# for computing end_precision
def result_target_predict_t_end_precision(dir_path):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_train.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    count_true_sum = 0
    count_false_sum = 0

    for i, i_rec in enumerate(recs):
        if i % 2 == 0:

            i_content = recs[i].split('t: ')[1].split('\n')[0]
            in_content = recs[i + 1].split('p: ')[1].split('\n')[0]

            count_true, count_false = count_correct_end_precision(i_content, in_content)
            count_true_sum += count_true
            count_false_sum += count_false

    if count_true_sum + count_false_sum == 0:
        return 0
    else:
        return count_true_sum / (count_true_sum + count_false_sum)
