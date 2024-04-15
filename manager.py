import numpy as np

import data
import model as m
import patch


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

    def get_dataset_data_for_train(self, name):
        images, labels = [], []
        for sample in list(self.datasets[name].keys()):
            image = None
            label = []
            for data in list(self.datasets[name][sample].keys()):
                if 'image' in data:
                    image = np.expand_dims(np.array(self.datasets[name][sample][data]), axis=-1)
                elif 'label' in data:
                    label.append(np.array(self.datasets[name][sample][data]))
                else:
                    raise NotImplementedError
            if image is None or len(label) <= 0:
                raise ValueError
            label = np.moveaxis(np.array(label), 0, -1)
            images.append(image)
            labels.append(label)
        return images, labels

    def get_models_list(self):
        models = []
        for model in self.models:
            models.append(f'{model.name} {model.input_shape} > {model.output_shape}')
        return models

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

    def save_model(self, index, filename):
        import tensorflow as tf
        model = self.models[index]
        model.save(filename)

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
                  activation='sigmoid',
                  name=None):
        if normalization == 'None':
            normalization = False
        else:
            normalization = normalization.lower()
        model = m.create(dimension, architecture.lower(), backbone.lower(), kernel_size, block_filters, block_per_level,
                         normalization, depth, outputs, activation.lower(), name)
        self.models.append(model)
        print(model.name)

    def train_model(self,
                    model_index,
                    dataset_name_train,
                    dataset_name_valid,
                    dataset_name_test,
                    loss,
                    batch_size,
                    patch_size_z,
                    patch_size_y,
                    patch_size_x,
                    steps_per_epoch,
                    epochs,
                    validation_steps,
                    keep_best=True,
                    early_stop=True):
        import tensorflow as tf
        model = self.models[model_index]

        # Get model information
        is2d = len(model.input_shape) == 4  # (batch_size, y, x, chan=1)
        patch_size = (patch_size_y, patch_size_x) if is2d else (patch_size_z, patch_size_y, patch_size_x)

        # Load all data in RAM (and avoid copy of same data)
        train_img, train_lbl = self.get_dataset_data_for_train(dataset_name_train)
        if dataset_name_train == dataset_name_valid:
            valid_img, valid_lbl = train_img, train_lbl
        else:
            valid_img, valid_lbl = self.get_dataset_data_for_train(dataset_name_valid)
        if dataset_name_train == dataset_name_test:
            test_img, test_lbl = train_img, train_lbl
        elif dataset_name_valid == dataset_name_test:
            test_img, test_lbl = valid_img, valid_lbl
        else:
            test_img,  test_lbl  = self.get_dataset_data_for_train(dataset_name_test)

        # Create patch generators
        gen_train = patch.gen_patch_batch(patch_size, train_img, train_lbl, batch_size=batch_size, augmentation=True)

        # Create callbacks
        # todo

        # Train model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        # todo: use loss
        model.compile(optimizer=optimizer, loss='MSE')
        fit_history = model.fit(gen_train,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs)

        self.models[model_index] = model
