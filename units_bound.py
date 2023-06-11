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
            if p_target_combine[i] == '1':
                print("Error")
            if p_target_combine[i] == '2':
                c_end.append(i)
                flag_start = False

    if len(c_start) == len(c_end) + 1:
        return c_start[:-1], c_end
    else:
        return c_start, c_end


def compare_target_predict_bound(start_predicts, end_predicts, start_targets, end_targets):

    target_set = set(zip(start_targets, end_targets))
    predict_set = set(zip(start_predicts, end_predicts))

    c_true = len(target_set.union(predict_set))
    c_false = len(target_set - predict_set)

    return c_true, c_false


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


def result_target_predict_bound_recall_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if rec.__contains__('t: '):
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        elif rec.__contains__('p: '):
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    start_targets, end_targets = start_end_targets_2(target_combine)
    start_predicts, end_predicts = start_end_predicts_2(predict_combine)

    count_true, count_false = compare_target_predict_bound(start_predicts, end_predicts, start_targets, end_targets)

    if count_true + count_false == 0:
        return 0
    else:
        return count_true / (count_true + count_false)


def result_target_predict_bound_precision_2(dir_path, step):

    dir_path = dir_path + 'compare_target_predict/vi_compare_target_predict_valid_2_' + str(step) + '.txt'

    with open(dir_path, 'r') as file:
        recs = file.readlines()

    target_combine = ''
    predict_combine = ''

    for i, rec in enumerate(recs):
        if rec.__contains__('t: '):
            target_combine += recs[i].split('t: ')[1].split('\n')[0]
        elif rec.__contains__('p: '):
            predict_combine += recs[i].split('p: ')[1].split('\n')[0]

    start_targets, end_targets = start_end_targets_2(target_combine)
    start_predicts, end_predicts = start_end_predicts_2(predict_combine)

    count_true, count_false = compare_target_predict_bound(start_targets, end_targets, start_predicts, end_predicts)

    if count_true + count_false == 0:
        return 0
    else:
        return count_true / (count_true + count_false)
