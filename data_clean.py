import numpy as np
from __builtin__ import unicode


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def delete_unknown(classe, attr):
    clean_attr = np.zeros((1, attr.shape[1]))
    clean_class = np.zeros((1))
    for instance in range(attr.shape[0]):
        if not "?" in attr[instance]:
            clean_attr = np.vstack((clean_attr, attr[instance]))
            clean_class = np.vstack((clean_class, classe[instance]))

    return np.delete(clean_class, 0), np.delete(clean_attr, 0, axis=0)

def attributePossibleValues(attributes):
    ret = []
    for i in range(attributes.shape[1]):
        unique_elements = np.unique(attributes[:, i])
        indexes = np.argwhere(unique_elements != '?').reshape(-1)
        unique_elements = unique_elements[indexes]
        ret.append(unique_elements)

    return ret

def classificationPossibleValues(classification):
    unique_elements = np.unique(classification)
    indexes = np.argwhere(unique_elements!='?').reshape(-1)
    unique_elements = unique_elements[indexes]
    return unique_elements

def processContinuousAttributesAsNormal(attributes):
    attrs = np.transpose(attributes)
    isContinuous = np.zeros(attrs.shape[0], dtype=bool)
    for attribute in range(attrs.shape[0]):
        for sample in range(attrs.shape[1]):   
            if attrs[attribute, sample] != '?':
                isContinuous[attribute] = is_number(attrs[attribute, sample])
                break
        
    for attribute in range(attrs.shape[0]):
        if isContinuous[attribute]:
            mean = np.mean(attrs[attribute].astype(float))
            std = np.std(attrs[attribute].astype(float))
            for value in range(attrs.shape[1]):
                if attrs[attribute, value] > mean:
                    if attrs[attribute, value] > mean + std:
                        attrs[attribute, value] = "> mean+std"
                    else:
                        attrs[attribute, value] = "[mean, mean+std]"
                else:
                    if attrs[attribute, value] < mean - std:
                        attrs[attribute, value] = "< mean-std"
                    else:
                        attrs[attribute, value] = "[mean-std, mean]"

    
    return np.transpose(attrs)