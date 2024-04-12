import numpy as np

import data
import model as m


class Manager:
    def __init__(self):
        self.datasets = dict()
        self.models = []

    def get_datasets_index(self):
        """
        return a list of existing datasets.
        """
        return list(self.datasets.keys())

    def get_datasets_number_labels(self, name):
        return self.datasets[name].attrs['labels']

    def get_sample(self, dataset, sample):
        return self.datasets[dataset][sample]

    def get_dataset_samples(self, name):
        sample_info = []
        for sample in list(self.datasets[name].keys()):
            sample_info.append((sample, list(self.datasets[name][sample].keys())))
        return sample_info

    def load_dataset(self, filename):
        """
        load a dataset from an existing file.
        """
        dataset = data.read_dataset(filename)
        if dataset.attrs['name'] in self.datasets.keys():
            raise ValueError(f'A dataset with same name already exist.\n{filename}')
        self.datasets[dataset.attrs['name']] = dataset
        return dataset.attrs['name']

    def new_dataset(self, name, labels=0, filename=None):
        """
        create a new dataset.
        """
        if filename is None:
            filename = name + '_DeepSCEM.hdf5'
        self.datasets[name] = data.create_dataset(name, labels, filename)

    def add_sample(self, name, sample_name, sample_image_filename, sample_labels_filenames):
        """
        add a sample composed of multiple files to an existing dataset.
        """
        dataset = self.datasets[name]

        # read the image
        try:
            sample_image = data.read_image(sample_image_filename, dtype=np.float32, normalize=True)
        except Exception as e:
            raise Exception(f'Sample image load error.\n{e}')
        # read the labels
        sample_labels = []
        for choice_label in sample_labels_filenames:
            # if label is empty or None, label is considered as valid but blank.
            if choice_label == '' or choice_label is None:
                sample_label = np.zeros(sample_image.shape, dtype=np.uint8)
            else:
                try:
                    sample_label = data.read_image(choice_label, dtype=np.uint8)
                    if not sample_image.shape == sample_label.shape:
                        raise Exception(f'Label shape does not match image shape.')
                except Exception as e:
                    raise Exception(f'Sample load error.\n{e}')
            sample_labels.append(sample_label)

        data.add_sample_to_dataset(dataset, sample_name, sample_image, sample_labels)

    def remove_dataset(self, name):
        self.datasets[name].close()
        del self.datasets[name]
        self.datasets.pop(name, None)

    def remove_sample(self, dataset, sample):
        # del self.datasets[dataset][sample]
        data.remove_sample_from_dataset(self.datasets[dataset], sample)

    def saveas_dataset(self, name, filename):
        data.copy_dataset(self.datasets[name], filename)

    def load_model(self, filename):
        import tensorflow as tf
        model = tf.keras.models.load_model(filename)
        self.models.append(model)

    def new_model(self,
                  dimension=2,
                  architecture='u-net',
                  backbone='residual',
                  kernel_size=3,
                  block_filters=32,
                  block_per_level=2,
                  normalization='batchnorm',
                  depth=5,
                  outputs=2,
                  activation='sigmoid'):
        model = m.create(dimension, architecture.lower(), backbone.lower(), kernel_size, block_filters, block_per_level,
                         normalization.lower(), depth, outputs, activation.lower())
        self.models.append(model)
