import numpy as np
import torch
import torch.utils.data as data

import random
import pickle


class DataGenerator(data.DataLoader):
    """Data loader for model training"""
    def __init__(self, root, keys=['CN','MCI', 'AD']):
        with open(root, 'rb') as load_data:
            data_dict = pickle.load(load_data)

        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]
        self.data = data_
        self.channel = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(self, batch_size=5, n_way=4, num_shots=10, unlabelled_node=None, cuda=False, variable=False):
        # initialise all features and labels with zeros until we feed in data
        valid_unlabelled_node = isinstance(unlabelled_node, np.ndarray)
        batch_x = np.zeros((batch_size, self.channel, self.feature_shape[0], self.feature_shape[1]), dtype='float32')  # features
        labels_x = np.zeros((batch_size, n_way), dtype='float32')  # labels
        labels_x_global = np.zeros(batch_size, dtype='int64')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(np.zeros((batch_size, self.channel, self.feature_shape[0], self.feature_shape[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype='float32')))

        # build data batches
        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if class_num == pre_class:
                    # take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)

                    # for each sample in batch, provide a set of test samples
                    if valid_unlabelled_node and batch_counter == batch_size - 1:
                        batch_x[batch_counter,0, :,:] = unlabelled_node
                    else:
                        batch_x[batch_counter,0, :,:] = samples[0]
                        # one hot encode label for labelled samples
                        labels_x[batch_counter, class_num] = 1

                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)

                for sample in samples:
                    try:
                        batches_xi[indexes_perm[counter]][batch_counter, :] = sample
                    except:
                        print(sample)

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    counter += 1

            numeric_labels.append(pre_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr
