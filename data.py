import numpy as np
import tifffile
import h5py


def read_image(filename, dtype=np.float32, normalize=False, binarize=False):
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
    # binarize value in {0, 1}.
    if binarize:
        data = (data > 0).astype(dtype)
    # convert 2D array into 3D array.
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    # check that output is 3D array.
    if len(data.shape) != 3:
        raise NotImplementedError('NotImplementedError')

    return data


def write_image(filename, data, extension='tiff'):
    if extension in ['tif', 'tiff', 'TIF', 'TIFF']:
        tifffile.imwrite(filename + f'.{extension}', data)
    else:
        raise NotImplementedError('NotImplementedError')


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


def crop_sample(dataset, sample_name, new_name, z_min, z_max, y_min, y_max, x_min, x_max):
    sample_group = dataset.create_group(new_name)
    sample_group.attrs['shape'] = [z_max - z_min, y_max - y_min, x_max - x_min]

    for s in dataset[sample_name].keys():
        data = np.array(dataset[sample_name][s])
        sample_group.create_dataset(f'{s}', data=data[z_min:z_max, y_min:y_max, x_min:x_max])

    dataset.flush()


def add_prediction_to_dataset(dataset, name, image, prediction):
    sample_group = dataset.create_group(name)
    sample_group.attrs['shape'] = image.shape
    sample_group.create_dataset('image', data=image)

    for i in range(prediction.shape[-1]):
        sample_group.create_dataset(f'prediction_{i:04}', data=prediction[:, :, :, i])

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
