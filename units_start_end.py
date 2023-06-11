def start_end_predicts_2(p_predict_combine):
    flag_start = False
    c_start = []
    c_end = []
    for i in range(len(p_predict_combine)):
        if flag_start is False:
            if p_predict_combine[i] == '2':
                c_start.append(i)
                c_end.append(i)
            elif p_predict_combine[i] == '0':
                c_start.append(i)
                flag_start = True
        else:
            if p_predict_combine[i] == '2':
                c_end.append(i)
                flag_start = False

    if len(c_start) == len(c_end) + 1:
        return c_start[:-1], c_end
    else:
        return c_start, c_end


def start_end_targets_2(p_target_combine):
    flag_start = False
    c_start = []
    c_end = []
    for i in range(len(p_target_combine)):
        if flag_start is False:
            if p_target_combine[i] == '2':
                c_start.append(i)
                c_end.append(i)
            elif p_target_combine[i] == '0':
                c_start.append(i)
                flag_start = True
        else:
            if p_target_combine[i] == '2':
                c_end.append(i)
                flag_start = False

    if len(c_start) == len(c_end) + 1:
        return c_start[:-1], c_end
    else:
        return c_start, c_end


def compare_target_predict_start(start_predicts, start_targets):
    count_true = 0
    count_false = 0
    for i in start_predicts:
        if i in start_targets:
            count_true += 1
        else:
            count_false += 1

    return count_true, count_false


def result_target_predict_start_precision_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if i % 2 == 0:
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        else:
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    start_targets, _ = start_end_targets_2(target_combine)

    start_predicts, _ = start_end_predicts_2(predict_combine)

    count_true_start, count_false_start = compare_target_predict_start(start_predicts, start_targets)

    if count_true_start + count_false_start == 0:
        return 0
    else:
        return count_true_start / (count_true_start + count_false_start)


def result_target_predict_start_recall_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if i % 2 == 0:
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        else:
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    start_targets, _ = start_end_targets_2(target_combine)
    start_predicts, _ = start_end_predicts_2(predict_combine)

    count_true_start, count_false_start = compare_target_predict_start(start_targets, start_predicts)

    if count_true_start + count_false_start == 0:
        return 0
    else:
        return count_true_start / (count_true_start + count_false_start)


def result_target_predict_end_recall_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if i % 2 == 0:
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        else:
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    _, end_targets = start_end_targets_2(target_combine)
    _, end_predicts = start_end_predicts_2(predict_combine)

    count_true_start, count_false_start = compare_target_predict_start(end_targets, end_predicts)

    if count_true_start + count_false_start == 0:
        return 0
    else:
        return count_true_start / (count_true_start + count_false_start)


def result_target_predict_end_precision_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if i % 2 == 0:
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        else:
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    _, end_targets = start_end_targets_2(target_combine)
    _, end_predicts = start_end_predicts_2(predict_combine)

    count_true_start, count_false_start = compare_target_predict_start(end_predicts, end_targets)

    if count_true_start + count_false_start == 0:
        return 0
    else:
        return count_true_start / (count_true_start + count_false_start)
