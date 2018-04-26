import numpy as np

save_path = "results/treeRepresentation.txt"

class Tree:

    def __init__(self, attributes, instance_class, attr_possible_values, class_possible_values, method=0):     # Method es el algoritmo que usas para escoger mejor atributo (ID3, C4.5,...)

        self.attr_possible_values = attr_possible_values
        self.class_possible_values = class_possible_values
        self.method = method

        unique_elements, unique_counts = np.unique(instance_class, return_counts=True)

        self.head = Node(self, attributes, instance_class, range(attributes.shape[1]),
													'General', unique_elements[unique_counts.argmax()])

    def generateTree(self):
        nodeList = [self.head]
        while nodeList:
            firstNode = nodeList.pop()
            firstNode.expand(nodeList)

    def representTree(self):
        with open(save_path, 'w') as file:
            self.head.represent(0, file)

    def classificate(self, val_attributes):

        tree_classif = np.zeros(val_attributes.shape[0]).astype(str)

        for i in range(val_attributes.shape[0]): #Per totes les mostres
            current_node = self.head
            while not current_node.leaf:
                current_node = current_node.getNextNode(val_attributes[i])
            tree_classif[i] = current_node.solution

        return tree_classif

    def classificateSampleWithMissingValues(self, sample):
        result = self.head.classificateSampleWithMissingValues(sample)

        return [(self.class_possible_values[i], result[i]) for i in range(self.class_possible_values.shape[0])]


def generalEntropy(instance_class): #Entropia(S)

    unique_elements, unique_counts = np.unique(instance_class, return_counts=True)
    num_elements = len(instance_class)

    probabilities = np.true_divide(unique_counts, num_elements)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def elementEntropy(attr_col, instance_class): #Entropia(S,A)
    # attr_row = np.transpose(attr_col.copy())
    attr_row = attr_col.ravel()

    unique_elements, unique_counts = np.unique(attr_row, return_counts=True)
    num_elements = len(instance_class)

    probabilities = np.true_divide(unique_counts, num_elements)

    entropy = 0
    for i in range(len(unique_elements)):
        indexes = np.argwhere(attr_row == unique_elements[i])
        # subarray = instance_class[indexes]
        subarray = np.take(instance_class, indexes, axis=0)
        entropy += probabilities[i] * generalEntropy(subarray)

    return entropy


class Node:

    def __init__(self, tree, attributes, instance_class, possible_attr, chosen_value, defaultValue):

        self.tree = tree                        # reference to the tree this node is contained in
        self.attribute = None                   # the index of the attribute if this is a question node; None otherwise
        self.solution = None                    # the number of the attribute if there is; None otherwise
        self.children = {}						# dictionary of the children this node has; indexed by attribute value
        self.attributes = attributes			# attribute matrix containing only the samples that belong here
        self.instance_class = instance_class	# real classification array containing only the samples that belong here
        self.possible_attr = possible_attr		# list of indexes that tells which of the attributes can still be chosen
        self.chosen_value = chosen_value		# value of the parent decision this node represents
        self.defaultValue = defaultValue		# in case this node recieves an empty list, this will be
												# a solution node that returns this value

        unique_elements, unique_count = np.unique(instance_class, return_counts=True)

        if len(unique_elements) == 1: #Comprovem si tota la classificacio es igual (el node es fulla)
            self.leaf = True
            self.solution = instance_class[0]
        elif len(unique_elements) == 0:
            self.leaf = True
            self.solution = defaultValue
        elif not possible_attr: # Comprovem si hi ha possibles atributs per classificar
            self.leaf = True    # Si no n'hi ha, encara que no haguem acabat de classificar a la perfeccio,
                                # posem que el node es resposta amb la resposta que mes apareix
            self.solution = unique_elements[np.argmax(unique_count)]
        else: # Encara tenim atributs possibles per classificar i la nostra classificacio conte valors diferents
            self.leaf = False
            if self.tree.method == 0:
                self.ID3()
            elif self.tree.method == 1:
                self.C4dot5()

            self.value_rel_freq = {}
            curr_col = attributes[:, self.attribute].ravel()
            unique_elements, unique_count = np.unique(curr_col, return_counts=True)
            for i in range(len(unique_elements)):
                self.value_rel_freq[unique_elements[i]] = float(unique_count[i]) / curr_col.shape[0]


    def ID3(self): # Escull el millor atribut amb el Gain

        max_gain = -1
        max_index = None

        general_entropy_basic = generalEntropy(self.instance_class)
        for i in self.possible_attr:
            general_entropy = general_entropy_basic
            curr_column = np.take(self.attributes, i, axis=1)
            indexes = np.argwhere(curr_column != '?')
            indexed_instance_class = np.take(self.instance_class, indexes)
            if len(indexes) != self.instance_class.shape[0]:
                general_entropy = generalEntropy(indexed_instance_class)
            gain = general_entropy - elementEntropy(np.take(curr_column, indexes), indexed_instance_class)

            gain *= (float(indexes.shape[0])/self.instance_class.shape[0])
            if gain > max_gain:
                max_gain = gain
                max_index = i

        self.attribute = max_index #Es guarda l'index de l'atribut pel que classificara


    def C4dot5(self): #Escull el millor atribut amb el Gain ratio

        max_ratio = -1
        max_index = None
        general_entropy_basic = generalEntropy(self.instance_class)
        for i in self.possible_attr:
            general_entropy = general_entropy_basic
            indexes = np.argwhere(self.attributes[:, i] != '?')
            if len(indexes) != self.instance_class.shape[0]:
                general_entropy = generalEntropy(self.instance_class[indexes])
            gain = general_entropy - elementEntropy(self.attributes[indexes, i], self.instance_class[indexes])
            gain *= (float(indexes.shape[0]) / self.instance_class.shape[0])

            splitInfo = generalEntropy(self.attributes[:, i])
            if splitInfo == 0.0:
                gainRatio = 0
            else:
                gainRatio = gain / splitInfo
                
            if gainRatio > max_ratio:
                max_ratio = gainRatio
                max_index = i
          
        self.attribute = max_index

    def expand(self, nodeList): #Crea els Nodes fills classificant per l'atribut que ha escollit

        if not self.leaf:
            # attr = np.take(self.attributes, self.attribute, axis=1).reshape(-1)
            attr = np.take(self.attributes, self.attribute, axis=1).ravel()
            self.possible_attr.remove(self.attribute)
            unique_elements = self.tree.attr_possible_values[self.attribute]

            unique_found_elements, unique_count = np.unique(self.instance_class, return_counts=True)

            for element in unique_elements:
                indexes = np.argwhere(attr == element)
                #indexes = indexes.reshape(-1)
                indexes = indexes.ravel()

                #paremeters start
                # below_attributes = self.attributes[indexes, :]
                below_attributes = np.take(self.attributes, indexes, axis=0)
                # below_instance_class = self.instance_class[indexes]
                below_instance_class = np.take(self.instance_class, indexes, axis=0)
                # below_default_solution = unique_found_elements[np.argmax(unique_count)]
                below_default_solution = np.take(unique_found_elements, np.argmax(unique_count), axis=0)

                self.children[element] = Node(self.tree, below_attributes, below_instance_class, self.possible_attr[:], element, below_default_solution)

            nodeList.extend(self.children.values()[:]) #Llista per saber quins nodes queden per expandir


    def classificateSampleWithMissingValues(self, sample):
        if self.leaf:
            ret = np.zeros(self.tree.class_possible_values.shape[0]).astype(float)
            index = np.argwhere(self.tree.class_possible_values == self.solution)[0]
            ret[index] = 1

        else:
            sample_value = sample[self.attribute]
            if sample_value != '?':

                node = self.children.get(sample_value)

                return node.classificateSampleWithMissingValues(sample)
            else:
                ret = np.zeros(self.tree.class_possible_values.shape[0]).astype(float)
                for answer, child in self.children.iteritems():
                    ret += self.value_rel_freq[answer] * child.classificateSampleWithMissingValues(sample)


        return ret


    def represent(self, indent, file):

        acc = ((indent-1) * "    |")         #guiones verticales
        if indent>0:
            acc += "    |"                          #espacios + 'T' apuntando pa la derecha
        acc += 3 * "-"                          #guion lateral
        acc += "Value: " + self.chosen_value + " --> "

        if self.leaf:
            file.write(acc + "Solution:\"" + str(self.solution) + "\"" + "\n")
        else:
            file.write(acc + "Attribute: " + str(self.attribute) + "\n")
            for node in self.children.values():
                node.represent(indent+1, file)



    def getNextNode(self, sample_row):
        ret = self.children[sample_row[self.attribute]]

        return ret


