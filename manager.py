import numpy as np

import data


class Manager:
    def __init__(self):
        self.datasets = dict()

    def get_dataset_index(self):
        """
        return a list of existing datasets.
        """
        return list(self.datasets.keys())

    def load_dataset(self, filename):
        """
        load a dataset from an existing file.
        """
        dataset = data.read_dataset(filename)
        self.datasets[dataset.attrs['name']] = dataset

    def new_dataset(self, name, filename=None):
        """
        create a new dataset.
        """
        self.datasets[name] = data.create_dataset(name, filename)

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
