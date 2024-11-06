import time
import numpy as np

import data
import model as m
import metrics
import pred
import transform


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

    def get_datasets_labels_aliases(self, name, asdictionary=True):
        if 'labels_aliases' in self.datasets[name].attrs.keys():
            if asdictionary:
                return {f'label_{i:04}': item for i, item in enumerate(self.datasets[name].attrs['labels_aliases'])} | {
                        f'prediction_{i:04}': item for i, item in enumerate(self.datasets[name].attrs['labels_aliases'])}
            else:
                return self.datasets[name].attrs['labels_aliases']
        return {} if asdictionary else []

    def get_sample(self, dataset, sample):
        return self.datasets[dataset][sample]

    def get_dataset_samples(self, name, info=True):
        sample_info = []
        for sample in list(self.datasets[name].keys()):
            if info:
                sample_info.append((sample, list(self.datasets[name][sample].keys())))
            else:
                sample_info.append(sample)
        return sample_info

    def get_dataset_data_for_train(self, name):
        images, labels = [], []
        for sample in list(self.datasets[name].keys()):
            image = None
            label = []
            for data in list(self.datasets[name][sample].keys()):
                if 'image' in data:
                    image = np.expand_dims(np.array(self.datasets[name][sample][data], dtype=np.float32), axis=-1)
                elif 'label' in data:
                    label.append(np.array(self.datasets[name][sample][data], dtype=np.float32))
                else:
                    raise NotImplementedError
            if image is None or len(label) <= 0:
                raise ValueError
            label = np.moveaxis(np.array(label), 0, -1)
            images.append(image)
            labels.append(label)
        return images, labels

    def get_models_list(self, string=True):
        models = []
        for model in self.models:
            if string:
                models.append(f'{model.name} {model.input_shape} > {model.output_shape}')
            else:
                models.append((model.name, model.input_shape, model.output_shape))
        return models

    def load_dataset(self, filename, labels=None):
        """
        load a dataset from an existing file.
        """
        dataset = data.read_dataset(filename)

        if labels is not None and int(dataset.attrs['labels']) != labels:
            raise AssertionError(f'Labels attribute in dataset do not match expected labels.\n{filename}')

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

    def rename_dataset(self, name, new_name):
        dataset = self.datasets.pop(name)
        dataset.attrs['name'] = new_name
        self.datasets[new_name] = dataset

    def rename_sample(self, dataset_name, sample_name, new_name):
        dataset = self.datasets[dataset_name]
        # https://docs.h5py.org/en/stable/high/group.html#h5py.Group.move
        dataset.move(sample_name, new_name)
        dataset.flush()

    def rename_labels_aliases(self, dataset_name, aliases_table):
        dataset = self.datasets[dataset_name]
        dataset.attrs['labels_aliases'] = aliases_table

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
                    sample_label = data.read_image(choice_label, dtype=np.uint8, binarize=True)
                    if not sample_image.shape == sample_label.shape:
                        raise Exception(f'Label shape does not match image shape.')
                except Exception as e:
                    raise Exception(f'Sample load error.\n{e}')
            sample_labels.append(sample_label)

        data.add_sample_to_dataset(dataset, sample_name, sample_image, sample_labels)

    def export_sample(self, dataset, sample, folder):
        sample_ = self.datasets[dataset][sample]
        label_aliases = self.get_datasets_labels_aliases(dataset)
        for imgk in sample_.keys():
            data.write_image(f'{folder}/{label_aliases.get(imgk, imgk)}', sample_[imgk], extension='tiff')


    def crop_sample(self, name, sample_name, z_min, z_max, y_min, y_max, x_min, x_max, new_name=None):
        """
        crop an existing sample.
        """
        dataset = self.datasets[name]
        if new_name is None or new_name == '':
            new_name = f'{sample_name}_crop_{z_min}_{z_max}_{y_min}_{y_max}_{x_min}_{x_max}'
        data.crop_sample(dataset, sample_name, new_name, z_min, z_max, y_min, y_max, x_min, x_max)

    def remove_dataset(self, name):
        self.datasets[name].close()
        del self.datasets[name]
        self.datasets.pop(name, None)

    def remove_sample(self, dataset, sample):
        # del self.datasets[dataset][sample]
        data.remove_sample_from_dataset(self.datasets[dataset], sample)

    def saveas_dataset(self, name, filename):
        data.copy_dataset(self.datasets[name], filename)
    
    def eval_dataset(self, reference_name, segmentation_name, f1=True, iou=False):
        results = []
        ref_ds = self.datasets[reference_name]
        seg_ds = self.datasets[segmentation_name]

        if not ref_ds.attrs['labels'] == seg_ds.attrs['labels']:
            raise ValueError('Datasets number of labels does not match')

        for i in range(ref_ds.attrs['labels']):
            score_samples = {}
            for sample in ref_ds.keys():
                if sample in seg_ds.keys():
                    ref_samples_data = '\t'.join(list(ref_ds[sample].keys()))
                    if 'label_' in ref_samples_data:
                        ref = np.array(ref_ds[sample][f'label_{i:04}'])
                    elif 'prediction_' in ref_samples_data:
                        ref = np.array(ref_ds[sample][f'prediction_{i:04}'])
                    else:
                        raise ValueError('Reference do not have label or prediction')

                    seg_samples_data = '\t'.join(list(seg_ds[sample].keys()))
                    if 'prediction_' in seg_samples_data:
                        seg = np.array(seg_ds[sample][f'prediction_{i:04}'])
                    elif 'label_' in seg_samples_data:
                        seg = np.array(seg_ds[sample][f'label_{i:04}'])
                    else:
                        raise ValueError('Segmentation do not have label or prediction')

                    scores = {'f1': None, 'iou': None}
                    if f1:
                        scores['f1'] = metrics.f1(ref, seg)
                    if iou:
                        scores['iou'] = metrics.iou(ref, seg)

                    score_samples[sample] = scores
            results.append(score_samples)

        return results

    def distance_transform(self, name):
        dataset = self.datasets[name]

        for sample in list(self.datasets[name].keys()):
            for data in list(self.datasets[name][sample].keys()):
                if 'label' in data:
                    self.datasets[name][sample][data+'_dt'] = transform.label_dt(np.array(self.datasets[name][sample][data]))

    def load_model(self, filename, labels=None):
        import tensorflow as tf

        model = tf.keras.models.load_model(filename, compile=False)

        if len(model.input_shape) != len(model.output_shape):
            raise AssertionError(f'Model shapes error. Input and output shapes does not match.\n{filename}')
        if not (4 <= len(model.input_shape) <= 5):
            raise AssertionError(f'Model shapes error. Input shape is not 2D or 3D.\n{filename}')
        if not model.input_shape[-1] == 1:
            raise AssertionError(f'Model shapes error. Input shape chan is not equal to 1.\n{filename}')
        if labels is not None and model.output_shape[-1] != labels:
            raise AssertionError(f'Model output label do not match expected labels.\n{filename}')

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
                  outputs=1,
                  activation='sigmoid',
                  name=None):
        if normalization == 'None':
            normalization = False
        else:
            normalization = normalization.lower()
        model = m.create(dimension, architecture.lower(), backbone.lower(), kernel_size, block_filters, block_per_level,
                         normalization, depth, outputs, activation.lower(), name)
        self.models.append(model)

    def train_model(self,
                    model_index,
                    dataset_name_train,
                    dataset_name_valid,
                    loss,
                    batch_size,
                    patch_size_z,
                    patch_size_y,
                    patch_size_x,
                    steps_per_epoch,
                    epochs,
                    validation_steps,
                    keep_best=True,
                    early_stop=False,
                    augmentations=(False, False),
                    label_focus=0):
        import tensorflow as tf
        model = self.models[model_index]

        # Get model information and set options
        is2d = len(model.input_shape) == 4  # (batch_size, y, x, chan=1)
        if not is2d and len(model.input_shape) != 5:  # (batch_size, z, y, x, chan=1)
            raise NotImplementedError("Input model is not 2D or 3D")
        n_classes = self.datasets[dataset_name_train].attrs['labels']
        patch_size = (patch_size_y, patch_size_x) if is2d else (patch_size_z, patch_size_y, patch_size_x)

        # Load all data in RAM
        train_img, train_lbl = self.get_dataset_data_for_train(dataset_name_train)
        # Don't load validation data if no validation steps
        if validation_steps is None or validation_steps <= 0:
            valid_img, valid_lbl = None, None
        else:
            # Avoid copy of same data
            if dataset_name_train == dataset_name_valid:
                valid_img, valid_lbl = train_img, train_lbl
            else:
                valid_img, valid_lbl = self.get_dataset_data_for_train(dataset_name_valid)

        # Create callbacks
        train_id = int(time.time())
        callbacks = [tf.keras.callbacks.CSVLogger(f'{model.name}_{train_id}_logs.csv')]
        if keep_best:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(f'{model.name}_{train_id}_best.h5',
                                                   save_best_only=True,
                                                   save_weights_only=False))
        if early_stop:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=max(5, epochs//20)))

        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1))

        # Train model
        import train
        loss = train.get_loss(loss, n_classes)
        t0 = time.time()
        model = train.train_model(model, (train_img, train_lbl), (valid_img, valid_lbl),
                                  loss, batch_size, patch_size, steps_per_epoch, epochs,
                                  validation_steps, callbacks, augmentations, label_focus)
        t1 = time.time()
        print(f'training duration: {int(t1 - t0)} s')

        self.models[model_index] = model

    def pred_model(self,
                   model_index,
                   dataset_name,
                   patch_size_z,
                   patch_size_y,
                   patch_size_x,
                   overlapping,
                   threshold=None):
        import tensorflow as tf
        model = self.models[model_index]
        dataset = self.datasets[dataset_name]

        # Get model information and set options
        is2d = len(model.input_shape) == 4  # (batch_size, y, x, chan=1)
        if not is2d and len(model.input_shape) != 5:  # (batch_size, z, y, x, chan=1)
            raise NotImplementedError("Input model is not 2D or 3D")
        patch_size = (1, patch_size_y, patch_size_x) if is2d else (patch_size_z, patch_size_y, patch_size_x)
        overlap = 1
        if overlapping:
            overlap = (1, 2, 2) if is2d else (2, 2, 2)

        # Prediction
        if is2d:
            def predict(x):
                tf.keras.backend.clear_session()
                x = x[0]
                return np.expand_dims(model.predict(x, verbose=0), 0)
        else:
            def predict(x):
                tf.keras.backend.clear_session()
                return model.predict(x, verbose=0)

        # Create output dataset
        pred_dataset_name = f'{dataset_name} pred_' + f'{time.time():.4f}'.replace('.', '_')
        self.new_dataset(pred_dataset_name, labels=dataset.attrs['labels'])

        # Load data, predict and store result
        for sample in list(dataset.keys()):
            image = np.array(dataset[sample]['image'])
            prediction = pred.infer_pad(image,
                                        patch_size,
                                        predict,
                                        overlap=overlap,
                                        verbose=1)
            if threshold is not None:
                prediction = (prediction > threshold).astype(np.uint8)

            data.add_prediction_to_dataset(self.datasets[pred_dataset_name], sample, dataset[sample]['image'], prediction)
