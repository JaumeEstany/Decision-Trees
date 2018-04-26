
import validation_methods as methods
import math
import tree as tr
import numpy as np
import data_clean as dc
import validation_metrics as metr

class RandomForest:

    def __init__(self, num_trees):

        self.num_trees = num_trees
        self.trees = None

    def trainAndValidate(self, total_data, instance_class, attributes_possible_values, classification_possible_values, positive_value):

        instance_class, total_data = dc.delete_unknown(instance_class, total_data)

        const_num_total_attributes = total_data.shape[1]
        const_num_selected_attributes = int(math.sqrt(total_data.shape[1]))

        classif_errors = 0
        accuracys = 0
        precisions = 0
        recalls = 0
        self.trees = [None] * self.num_trees
                     
        for tree_num in range(self.num_trees):
            val_attr, val_classif, train_attr, train_classif = methods.bootstrapping_data(total_data, instance_class)

            # do the selection
            attributes_index = np.random.choice(np.arange(const_num_total_attributes), const_num_selected_attributes,
                                                replace=False)

            # train_attr = train_attr[:, attributes_index]
            train_attr = np.take(train_attr, attributes_index, axis=1)
            # val_attr = val_attr[:, attributes_index]
            val_attr = np.take(val_attr, attributes_index, axis=1)
            reordered_possible_attr_values = [attributes_possible_values[i] for i in attributes_index]

            # tree generation
            tree = tr.Tree(train_attr, train_classif, reordered_possible_attr_values, classification_possible_values)
            tree.generateTree()
            classification = tree.classificate(val_attr)

            classif_error, accuracy, precision, recall = metr.calculate_all_metrics(classification, val_classif,
                                                                                    positive_value)
            if tree_num > float(self.num_trees)/2:
                classif_errors += (classif_error**tree_num)*((1-classif_error)**(self.num_trees-tree_num)) * math.factorial(self.num_trees)/(math.factorial(self.num_trees-tree_num)*math.factorial(tree_num))  
            accuracys += accuracy
            precisions += precision
            recalls += (recall**tree_num)*((1-recall)**(self.num_trees-tree_num)) * math.factorial(self.num_trees)/(math.factorial(self.num_trees-tree_num)*math.factorial(tree_num))
            self.trees[tree_num] = (tree, attributes_index)

        return classif_errors, 1-classif_errors, precisions / self.num_trees, recalls / self.num_trees









