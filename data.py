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
        raise NotImplementedError('NotImplementedError')
    else:
        raise NotImplementedError('NotImplementedError')
    # normalize value in range [0, 1].
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
    # convert 2D array into 3D array.
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    # check that output is 3D array.
    if len(data.shape) != 3:
        raise NotImplementedError('NotImplementedError')

    return data


def read_dataset(filename, readonly=False):
    """
    read a dataset file into a h5py object.
    """
    if readonly:
        file = h5py.File(filename, 'r')
    else:
        file = h5py.File(filename, 'a')

    if 'name' not in file.attrs:
        raise ValueError('missing name attribute')
    if 'labels' not in file.attrs:
        raise ValueError('missing labels attribute')

    return file


def create_dataset(name, labels=0, filename=None):
    if filename is None:
        filename = name + '.hdf5'
    file = h5py.File(filename, 'w')

    file.attrs['name'] = name
    file.attrs['labels'] = labels
    # file.create_dataset('name', data=name)
    file.flush()

    return file


def add_sample_to_dataset(dataset, name, image, labels=None):
    sample_group = dataset.create_group(name)
    sample_group.attrs['shape'] = image.shape
    sample_group.create_dataset('image', data=image)

    for i, label in enumerate(labels):
        sample_group.create_dataset(f'label_{i:04}', data=label)

    dataset.flush()


def remove_sample_from_dataset(dataset, name):
    del dataset[name]
    dataset.flush()


def copy_dataset(dataset, filename):
    # inspiration: https://stackoverflow.com/a/53010788
    fd = h5py.File(filename, 'w')
    fs = dataset

    for a in fs.attrs:
        fd.attrs[a] = fs.attrs[a]
    for d in fs:
        # SFS_TRANSITION ? For now, we juste reuse working code.
        if not 'SFS_TRANSITION' in d: fs.copy(d, fd)

    fd.close()
