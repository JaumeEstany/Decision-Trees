import tree as tr
import numpy as np
import data_clean as dc
import validation_metrics as metr


def holdout_data(attributes, instance_class, train_ratio=0.8):
    """
    :param attributes: matrix of attributes of the data base (col = attribute)
    :param instance_class: vector with the classification of every sample
    :param train_ratio: % of data for the training set
    :return: the training samples with it classification
    and the validation samples with it classification
    """

    indices = np.arange(attributes.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(attributes.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    attr_train = attributes[indices_train, :]
    classif_train = instance_class[indices_train]
    attr_val = attributes[indices_val, :]
    classif_val = instance_class[indices_val]

    return attr_train, classif_train, attr_val, classif_val


def holdout(attributes, instance_class, repetitions, attributes_possible_values, classification_possible_values, positive_value, train_ratio=0.8):
    possible_attr_values = dc.attributePossibleValues(attributes)

    classif_errors = 0
    accuracys = 0
    precitions = 0
    recalls = 0

    for i in range(repetitions):
        attr_train, classif_train, attr_val, classif_val = holdout_data(attributes, instance_class, train_ratio)

        tree = tr.Tree(attr_train, classif_train, attributes_possible_values, classification_possible_values, 0)
        tree.generateTree()
        tree.representTree()
        classification = tree.classificate(attr_val)

        classif_error, accuracy, precision, recall = metr.calculate_all_metrics(classification, classif_val, positive_value)
        classif_errors += classif_error
        accuracys += accuracy
        precitions += precision
        recalls += recall

    rep = float(repetitions)
    return classif_errors/rep, accuracys/rep, precitions/rep, recalls/rep


def k_fold_data(attributes, instance_class, k):
    """
    :param attributes: matrix of attributes of the data base (col = attribute)
    :param instance_class: vector with the classification of every sample
    :param k: number of divisions of the dataset
    :return: list with length k containing the k diferents lists with divisions of data:
            [validation_attributes, validate_classification, training_attributes, training_classification]
    """

    indices = np.arange(attributes.shape[0])
    np.random.shuffle(indices)

    attr_split = np.asarray(np.array_split(attributes[indices], k))
    classif_split = np.asarray(np.array_split(instance_class[indices], k))

    data_subsets = []
    for i in range(k):
        separation = []

        indices = np.arange(k)
        indices = np.delete(indices, i)

        separation.append(attr_split[i])  # Validation attributes subset
        separation.append(classif_split[i])  # Validation classifications subset
        separation.append(np.concatenate(attr_split[indices]))  # Training attributes subset
        separation.append(np.concatenate(classif_split[indices]))  # Training classifications subset

        data_subsets.append(separation)

    return data_subsets


def k_fold(attributes, instance_class, attributes_possible_values, classification_possible_values, positive_value, k):
    data_subsets = k_fold_data(attributes, instance_class, k)

    classif_errors = 0
    accuracys = 0
    precitions = 0
    recalls = 0

    for i in range(len(data_subsets)):
        tree = tr.Tree(data_subsets[i][2], data_subsets[i][3], attributes_possible_values, classification_possible_values)
        tree.generateTree()
        classification = tree.classificate(data_subsets[i][0])

        classif_error, accuracy, precision, recall = metr.calculate_all_metrics(classification, data_subsets[i][1], positive_value)
        classif_errors += classif_error
        accuracys += accuracy
        precitions += precision
        recalls += recall

    k = float(k)
    return classif_errors/k, accuracys/k, precitions/k, recalls/k   #Tornem la mitja d'errors


def leave_one_out_data(attributes, instance_class, i):
    """
    :param attributes: matrix of attributes of the data base (col = attribute)
    :param instance_class: vector with the classification of every sample
    :param i: index of the element of validation
    :return: a list containing the data division with format:
            [validation_attributes, validate_classification, training_attributes, training_classification]
    """

    return attributes[i, :], instance_class[i], np.delete(attributes, i, axis=0), np.delete(instance_class, i, axis=0)


def leave_one_out(attributes, instance_class, attributes_possible_values, classification_possible_values, positive_value):

    classif_errors = 0
    accuracys = 0
    precitions = 0
    recalls = 0
    count_pos = 0

    n = attributes.shape[0]

    for i in range(n):
        print i
        val_attr, val_classif, train_attr, train_classif = leave_one_out_data(attributes, instance_class, i)
        val_attr = val_attr.reshape(1, len(val_attr))

        tree = tr.Tree(train_attr, train_classif, attributes_possible_values, classification_possible_values)
        tree.generateTree()
        classification = tree.classificate(val_attr)


        classif_error, accuracy, precision, recall = metr.calculate_all_metrics(classification, [val_classif], positive_value)
        classif_errors += classif_error
        accuracys += accuracy
        precitions += precision
        recalls += recall
        count_pos += (classification[0] == positive_value)

    n = float(n)#float(attributes.shape[0])
    return classif_errors/n, accuracys/n, precitions/count_pos, recalls/count_pos   #Tornem la mitja d'errors


def bootstrapping_data(attributes, instance_class, n_samples=None):
    """
    :param attributes: matrix of attributes of the data base (col = attribute)
    :param instance_class: vector with the classification of every sample
    :param n_samples: if specified the size of the obtained data set, if not n_sample will be the same as the input
                      attributes matrix size
    :return: a random and with repetitions data set of size n_samples
    """

    if n_samples is None:
        n_samples = attributes.shape[0]
    n_train = int(round(n_samples * 0.663))

    rand_indexs = np.random.randint(0, attributes.shape[0], n_samples)
    indexs_train = rand_indexs[:n_train]
    indexs_val = rand_indexs[n_train:]

    val_attr = attributes[indexs_val]
    val_classif = instance_class[indexs_val]
    train_attr = attributes[indexs_train]
    train_classif = instance_class[indexs_train]

    return val_attr, val_classif, train_attr, train_classif


def bootstrapping(attributes, instance_class, repetitions, attributes_possible_values, classification_possible_values, positive_value):
    """
    :param attributes: matrix of attributes of the data base (col = attribute)
    :param instance_class: vector with the classification of every sample
    :param repetitions: number of boostrap data sets we want to use simultaneously
    :return: different errors of the bootstrap trees
    """

    instance_class, attributes = dc.delete_unknown(instance_class, attributes)

    classif_errors = 0
    accuracys = 0
    precitions = 0
    recalls = 0

    for i in range(repetitions):
        val_attr, val_classif, train_attr, train_classif = bootstrapping_data(attributes, instance_class)

        tree = tr.Tree(train_attr, train_classif, attributes_possible_values, classification_possible_values, 0)
        tree.generateTree()
        classification = tree.classificate(val_attr)

        classif_error, accuracy, precision, recall = metr.calculate_all_metrics(classification, val_classif, positive_value)
        classif_errors += classif_error
        accuracys += accuracy
        precitions += precision
        recalls += recall

    rep = float(repetitions)
    return classif_errors/rep, accuracys/rep, precitions/rep, recalls/rep   #Tornem la mitja d'errors


def holdout_with_missing_values(attrs, classif, attributes_possible_values, classification_possible_values, insertMissing=False):
    attr_train, classif_train, attr_val, classif_val = holdout_data(attrs, classif, 0.8)

    tree = tr.Tree(attr_train, classif_train, attributes_possible_values, classification_possible_values, 0)
    tree.generateTree()
    tree.representTree()

    if insertMissing:
        attr_val = attr_val.copy()
        for i in range(attr_val.shape[1]):
            attr_val[i, 4] = '?'

    acc_error = 0.0
    for i in range(attr_val.shape[1]):
        curr_classif = tree.classificateSampleWithMissingValues(attr_val[i])

        for index in range(len(curr_classif)):
            if classif_val[i]==curr_classif[index][0]:
                sum = abs(1-curr_classif[index][1])
                acc_error += sum

    print 'Desviacio real-resposta mitja: ' + str(acc_error/(attr_val.shape[1]))
