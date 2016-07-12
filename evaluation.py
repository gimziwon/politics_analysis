#coding: utf-8
import numpy as np

def get_statistics(pred_prob, pred_label, truth_label):
    acc = get_accuracy(pred_label, truth_label)
    logloss = get_logloss(pred_prob, truth_label)
    micro_precision, micro_recall, micro_f1 \
        = get_micro(pred_label, truth_label)
    macro_precision, macro_recall, macro_f1 \
        = get_macro(pred_label, truth_label)

    stats = dict()
    stats['accuracy'] = acc
    stats['logloss'] = logloss
    stats['micro_precision'] = micro_precision
    stats['micro_recall'] = micro_recall
    stats['micro_f1'] = micro_f1
    stats['macro_precision'] = macro_precision
    stats['macro_recall'] = macro_recall
    stats['macro_f1'] = macro_f1

    return stats

def get_accuracy(pred_label, truth_label):
    return np.sum(pred_label==truth_label)/len(truth_label)

def get_logloss(pred_prob, truth_label):
    logloss_sum = 0
    for idx, class_n in enumerate(np.unique(truth_label)):
        logloss = np.sum(
            np.log2(pred_prob[:, idx])*(truth_label==class_n)
            + np.log2(1-pred_prob[:, idx])*(truth_label!=class_n)
        )

        logloss_sum += -1*logloss/np.sum(truth_label==class_n)

    return logloss_sum

def get_micro(pred_label, truth_label):
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for class_n in np.unique(truth_label):
        tp = get_true_positive(pred_label, truth_label, class_n)
        fp = get_false_positive(pred_label, truth_label, class_n)
        fn = get_false_negative(pred_label, truth_label, class_n)

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    micro_precision = tp_sum/(tp_sum+fp_sum)
    micro_recall = tp_sum/(tp_sum+fn_sum)
    micro_f1 \
        = 2*micro_precision*micro_recall/(micro_precision+micro_recall)

    return micro_precision, micro_recall, micro_f1

def get_macro(pred_label, truth_label):
    precision_sum = 0
    recall_sum = 0
    for class_n in np.unique(truth_label):
        tp = get_true_positive(pred_label, truth_label, class_n)
        fp = get_false_positive(pred_label, truth_label, class_n)
        fn = get_false_negative(pred_label, truth_label, class_n)

        precision_sum += tp/(tp+fp) if (tp+fp) != 0 else 0
        recall_sum += tp/(tp+fn) if (tp+fn) != 0 else 0

    macro_precision = precision_sum/len(np.unique(truth_label))
    macro_recall = recall_sum/len(np.unique(truth_label))
    macro_f1 \
        = 2*macro_precision*macro_recall/(macro_precision+macro_recall)

    return macro_precision, macro_recall, macro_f1

def get_true_positive(pred_label, truth_label, class_n):
    return np.sum((pred_label==truth_label)&(truth_label==class_n))

def get_false_positive(pred_label, truth_label, class_n):
    return np.sum((pred_label!=truth_label)&(pred_label==class_n))

def get_false_negative(pred_label, truth_label, class_n):
    return np.sum((pred_label!=truth_label)&(truth_label==class_n))
