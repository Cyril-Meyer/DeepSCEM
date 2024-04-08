import numpy as np
import tifffile
import h5py


def read_image(filename, dtype=np.float32, normalize=False):
    """
    read an image file into a numpy array with specific dtype and normalize option.
    """
    # read data with specific library for each file format.
    data = None
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        data = tifffile.imread(filename).astype(dtype)
    elif filename.endswith('.npy'):
        raise NotImplementedError
    else:
        raise NotImplementedError
    # normalize value in range [0, 1].
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
    # convert 2D array into 3D array.
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    # check that output is 3D array.
    if len(data.shape) != 3:
        raise NotImplementedError

    return data


def create_sample(filename, image, labels=None):
    data_image = read_image(image, normalize=True)
    data_shape = data_image.shape

    with h5py.File(filename + '.hdf5', 'w') as f:
        image_group = f.create_group('image')
        image_group.create_dataset('data', data=data_image)
        image_group.attrs['shape'] = data_shape
        labels_group = f.create_group('labels')

        for i, label in enumerate(labels):
            data_label = read_image(label, np.uint8)
            if data_label.shape != data_shape:
                raise ValueError('shapes does not match')
            labels_group.create_dataset(f'label_{i}', data=data_label)


def create_dataset(filename, samples=None):
    with h5py.File(filename + '.hdf5', 'w') as f:
        images_group = f.create_group('images')
        labels_group = f.create_group('labels')

        for i, sample in enumerate(samples):
            with h5py.File(sample + '.hdf5', 'r') as sample_file:
                image_data = sample_file['image/data'][:]
                image_shape = sample_file['image'].attrs['shape']
                images_group.create_dataset(f'image_{i}', data=image_data)
                images_group.attrs[f'image_{i}_shape'] = image_shape

                for j, label_name in enumerate(sample_file['labels']):
                    label_data = sample_file[f'labels/{label_name}'][:]
                    labels_group.create_dataset(f'label_{i}_{j}', data=label_data)
