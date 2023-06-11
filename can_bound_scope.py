import units_bound as units
import units_scope as units_s

dir_path = "./pe_x86/"
step = 19500  # this is the number of output file from the corresponding trained model

bound_recall = units.result_target_predict_bound_recall_2(dir_path, step)
bound_precision = units.result_target_predict_bound_precision_2(dir_path, step)

scope_recall = units_s.result_target_predict_scope_recall_2(dir_path, step)
scope_precision = units_s.result_target_predict_scope_precision_2(dir_path, step)

f_score_scope = (2 * scope_recall * scope_precision) / (scope_precision + scope_recall)
f_score_bound = (2 * bound_recall * bound_precision) / (bound_precision + bound_recall)

print("recall_scope=" + "{:.4f}".format(scope_recall) + ", precision_scope=" + "{:.4f}".format(scope_precision) +
      ", f_score_scope=" + "{:.4f}".format(f_score_scope))

print("recall_bound=" + "{:.4f}".format(bound_recall) + ", precision_bound=" + "{:.4f}".format(bound_precision) +
      ", f_score_bound=" + "{:.4f}".format(f_score_bound))

print("testing finished!")
