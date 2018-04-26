import numpy as np


def classification_error(tree_classification, real_classification):
    count = np.count_nonzero(tree_classification != real_classification)
    error = float(count)/float(len(tree_classification))
    return error


def accuracy(tree_classification, real_classification):
    count = np.count_nonzero(tree_classification == real_classification)
    return float(count)/float(len(tree_classification))

def precision(tree_classification, real_classification, positive_value):
    correct = tree_classification[tree_classification == real_classification]
    tp_count = np.count_nonzero(correct[correct == positive_value])
    pred_positive_count = np.count_nonzero(tree_classification[tree_classification == positive_value])        # Numero de valores positivos segun el arbol
    if (pred_positive_count != 0):
        return float(tp_count) / pred_positive_count
    else:
        return 0


def recall(tree_classification, real_classification, positive_value):
    real_positive_count = np.count_nonzero(real_classification[real_classification == positive_value])        # Numero de valores posiivos
    correct = tree_classification[tree_classification == real_classification]
    tp_count = np.count_nonzero(correct[correct == positive_value])
    if (real_positive_count != 0):
        return float(tp_count) / real_positive_count
    else:
        return 0


def calculate_all_metrics(tree_classification, real_classification, positive_value):
    classif_error = classification_error(tree_classification, real_classification)
    acc = 1 - classif_error # No hace falta usar la funcion accuracy(tree_classification, real_classification)
    prec = precision(tree_classification, real_classification, positive_value)
    rec = recall(tree_classification, real_classification, positive_value)

    return classif_error, acc, prec, rec