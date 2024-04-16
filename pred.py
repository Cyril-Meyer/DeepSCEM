# this code is based on the project "Large Image Inference" : https://github.com/Cyril-Meyer/lii
import numpy as np
from tqdm import tqdm


def infer_out_smaller(image, window_in_shape, window_out_shape, f, verbose=0):
    """
    infer on large image with shrinking inference function.
    :param image: a 3D or 4D numpy array, channel last, shape = (Z, Y, X, C) or (Z, Y, X)
                  if 3D array, a channel will be added.
    :param window_in_shape: a 3D tuple, a 3D array or an int (if same value for all axis)
    :param window_out_shape: a 3D tuple, a 3D array or an int (if same value for all axis)
    :param f: a function or class to apply,
    taking a single argument, a 5D numpy array as input, channel last, shape = window_in_shape
    returning an array of window_out_shape.
    :param verbose: verbosity, int, 0 = silent, 1 = show progress, 2 = show debug
    :return: a 3D or 4D numpy array
    :precondition window_in_shape > window_out_shape.
    """
    # parameters processing
    window_in_shape = np.array(window_in_shape)
    window_out_shape = np.array(window_out_shape)
    size_difference = window_in_shape - window_out_shape
    strides = np.array(window_out_shape)

    # assertion
    assert len(image.shape) in [3, 4], \
        f'image shape {image.shape} invalid'
    assert len(window_in_shape) == 3, \
        f'window in shape {window_in_shape} invalid'
    assert len(window_out_shape) == 3, \
        f'window out shape {window_out_shape} invalid'
    assert (np.array(size_difference) > 0).all(), \
        f"window_in_shape {window_in_shape} <= window_out_shape {window_out_shape}"

    # add a color dim if necessary
    n_dim_in = len(image.shape)
    if n_dim_in == 3:
        image = np.expand_dims(image, -1)
    image_shape = np.array(image.shape)[0:3]
    image_out_shape = image_shape - size_difference

    assert (image_out_shape % np.array(window_out_shape) == [0, 0, 0]).all(), \
        f"image shape {image_shape} - size_difference {size_difference} :" \
        f"{image_out_shape} cannot be divided by window_out_shape {window_out_shape}"

    # loop and infer
    result = None

    z_max = image_out_shape[0]
    y_max = image_out_shape[1]
    x_max = image_out_shape[2]

    for z in tqdm(range(0, z_max, strides[0]), disable=(not verbose > 0)):
        for y in tqdm(range(0, y_max, strides[1]), disable=(not verbose > 0), leave=False):
            for x in tqdm(range(0, x_max, strides[2]), disable=(not verbose > 0), leave=False):
                # patch prediction
                p = image[z:z + window_in_shape[0], y:y + window_in_shape[1], x:x + window_in_shape[2]]

                if not (np.array(p.shape)[0:3] == window_in_shape).all():
                    raise LookupError

                p = f(np.expand_dims(p, 0))[0]

                if result is None:
                    result = np.zeros((image_out_shape[0], image_out_shape[1], image_out_shape[2], p.shape[-1]), dtype=p.dtype)
                result[z:z+p.shape[0], y:y+p.shape[1], x:x+p.shape[2], :] = p

    return result


def infer(image, window_shape, f, overlap=1, verbose=0, dtype=None):
    """
    infer on large image.
    :param image: a 3D or 4D numpy array, channel last, shape = (Z, Y, X, C) or (Z, Y, X)
                  if 3D array, a channel will be added.
    :param window_shape: a 3D tuple, a 3D array or an int (if same value for all axis)
    :param f: a function or class to apply,
    taking a single argument, a 5D numpy array as input, channel last, shape = (B, Z, Y, X, C)
    returning an array of same shape, with fixed number of channel.
    :param overlap: 3D tuple or int, values can be 1 or 2 (1 = no overlap, 2 = overlap)
    :param verbose: verbosity, int, 0 = silent, 1 = show progress, 2 = show debug
    :return: a 3D or 4D numpy array
    :precondition image.shape must be multiple of window_shape.
    infer is similar to infer_pad but with precondition.
    """
    # parameters processing
    if isinstance(overlap, int):
        overlap = (overlap, overlap, overlap)

    window_shape = np.array(window_shape)
    strides = np.array(window_shape) // overlap

    # assertion
    assert len(image.shape) in [3, 4], \
        f'image shape {image.shape} invalid'
    assert len(window_shape) == 3, \
        f'window shape {window_shape} invalid'
    assert (np.array(overlap) > 0).all() and (np.array(overlap) <= 2).all(), \
        f"overlap {overlap} values invalid"
    assert (strides == np.array(window_shape) / overlap).all(), \
        f"window_shape {window_shape} cannot be divided by overlap {overlap}"
    assert ((strides % overlap) == [0, 0, 0]).all(), \
        "strides cannot be divided by overlap"

    # add a color dim if necessary
    n_dim_in = len(image.shape)
    if n_dim_in == 3:
        image = np.expand_dims(image, -1)
    image_shape = np.array(image.shape)[0:3]

    assert (image_shape % np.array(window_shape) == [0, 0, 0]).all(), \
        f"image shape {image.shape} cannot be divided by window_shape {window_shape}"

    # loop and infer
    result = None
    # center border
    cb = strides - (strides // overlap)

    z_max = image_shape[0] - (strides[0] if overlap[0] == 2 else 0)
    y_max = image_shape[1] - (strides[1] if overlap[1] == 2 else 0)
    x_max = image_shape[2] - (strides[2] if overlap[2] == 2 else 0)

    for z in tqdm(range(0, z_max, strides[0]), disable=(not verbose > 0)):
        for y in tqdm(range(0, y_max, strides[1]), disable=(not verbose > 0), leave=False):
            for x in tqdm(range(0, x_max, strides[2]), disable=(not verbose > 0), leave=False):
                # patch prediction
                p = image[z:z + window_shape[0], y:y + window_shape[1], x:x + window_shape[2]]

                if not (np.array(p.shape)[0:3] == window_shape).all():
                    raise LookupError

                p = f(np.expand_dims(p, 0))[0]

                if dtype is None:
                    dtype = p.dtype

                if result is None:
                    result = np.zeros((image_shape[0], image_shape[1], image_shape[2], p.shape[-1]), dtype=dtype)

                # keep center
                z_, y_, x_ = z, y, x

                if image_shape[0] == strides[0]:
                    p = p[:, :, :, :]
                elif z == 0:
                    p = p[0:cb[0] + strides[0], :, :, :]
                elif z + strides[0] == z_max:
                    z_ = z + cb[0]
                    p = p[cb[0]:, :, :, :]
                else:
                    z_ = z + cb[0]
                    p = p[cb[0]:cb[0] + strides[0], :, :, :]

                if image_shape[1] == strides[1]:
                    p = p[:, :, :, :]
                elif y == 0:
                    p = p[:, 0:cb[1] + strides[1], :, :]
                elif y + strides[1] == y_max:
                    y_ = y + cb[1]
                    p = p[:, cb[1]:, :, :]
                else:
                    y_ = y + cb[1]
                    p = p[:, cb[1]:cb[1] + strides[1], :, :]

                if image_shape[2] == strides[2]:
                    p = p[:, :, :, :]
                elif x == 0:
                    p = p[:, :, 0:cb[2] + strides[2], :]
                elif x + strides[2] == x_max:
                    x_ = x + cb[2]
                    p = p[:, :, cb[2]:, :]
                else:
                    x_ = x + cb[2]
                    p = p[:, :, cb[2]:cb[2] + strides[2], :]

                result[z_:z_ + p.shape[0], y_:y_ + p.shape[1], x_:x_ + p.shape[2], :] = p

    return result


def infer_pad(image, window_shape, f, overlap=1, verbose=0):
    """
    infer on large image with arbitrary image and window shapes.
    :param image: a 3D or 4D numpy array, channel last, shape = (Z, Y, X, C) or (Z, Y, X)
                  if 3D array, a channel will be added.
    :param window_shape: a 3D tuple, a 3D array or an int (if same value for all axis)
    :param f: a function or class to apply,
    taking a single argument, a 5D numpy array as input, channel last, shape = (B, Z, Y, X, C)
    returning an array of same shape, with fixed number of channel.
    :param overlap: 3D tuple or int, values can be 1 or 2 (1 = no overlap, 2 = overlap)
    :param verbose: verbosity, int, 0 = silent, 1 = show progress, 2 = show debug
    :return: a 3D or 4D numpy array
    infer_pad is similar to infer but without precondition.
    padding is added to the input image.
    """
    # parameters processing
    if isinstance(overlap, int):
        overlap = (overlap, overlap, overlap)

    window_shape = np.array(window_shape)
    strides = np.array(window_shape) // overlap

    # assertion
    assert len(image.shape) in [3, 4], \
        f'image shape {image.shape} invalid'
    assert len(window_shape) == 3, \
        f'window shape {window_shape} invalid'
    assert (np.array(overlap) > 0).all() and (np.array(overlap) <= 2).all(), \
        f"overlap {overlap} values invalid"
    assert (strides == np.array(window_shape) / overlap).all(), \
        f"window_shape {window_shape} cannot be divided by overlap {overlap}"
    assert ((strides % overlap) == [0, 0, 0]).all(), \
        "strides cannot be divided by overlap"

    # add a color dim if necessary
    n_dim_in = len(image.shape)
    if n_dim_in == 3:
        image = np.expand_dims(image, -1)

    image_shape = np.array(image.shape)[0:3]
    # pad image (image_p = image with padding)
    image_p_shape = np.ceil(image_shape / strides).astype(np.int32) * strides
    pad = image_p_shape - image_shape
    # image_p = np.pad(image, ((0, pad[0]), (0, pad[1]), (0, pad[2]), (0, 0)), mode='reflect')
    image_p = np.pad(image,
                     ((strides[0], strides[0] + pad[0]),
                      (strides[1], strides[1] + pad[1]),
                      (strides[2], strides[2] + pad[2]),
                      (0, 0)),
                     mode='reflect')
    image_p_shape = image_p.shape

    # loop and infer
    result = None
    # center border
    cb = strides - (strides // overlap)

    if verbose > 1:
        print("input shape:", image.shape, "padded shape:", image_p.shape)
        print(len(range(0, image_p_shape[0] - strides[0], strides[0])) *
              len(range(0, image_p_shape[1] - strides[1], strides[1])) *
              len(range(0, image_p_shape[2] - strides[2], strides[2])))
        print(len(range(0, image_p_shape[0] - strides[0], strides[0])),
              len(range(0, image_p_shape[1] - strides[1], strides[1])),
              len(range(0, image_p_shape[2] - strides[2], strides[2])))

    for z in tqdm(range(0, image_p_shape[0] - strides[0], strides[0]), disable=(not verbose > 0)):
        for y in tqdm(range(0, image_p_shape[1] - strides[1], strides[1]), disable=(not verbose > 0), leave=False):
            for x in tqdm(range(0, image_p_shape[2] - strides[2], strides[2]), disable=(not verbose > 0), leave=False):
                # patch prediction
                p = image_p[z:z + window_shape[0], y:y + window_shape[1], x:x + window_shape[2]]
                if not (np.array(p.shape)[0:3] == window_shape).all():
                    raise LookupError

                p = f(np.expand_dims(p, 0))[0]

                if result is None:
                    result = np.zeros((image_p_shape[0], image_p_shape[1], image_p_shape[2], p.shape[-1]),
                                      dtype=p.dtype)

                # keep center
                p = p[cb[0]:cb[0] + strides[0], cb[1]:cb[1] + strides[1], cb[2]:cb[2] + strides[2], :]
                result[z + cb[0]:z + cb[0] + strides[0], y + cb[1]:y + cb[1] + strides[1],
                x + cb[2]:x + cb[2] + strides[2], :] = p

    # remove padding
    result = result[strides[0]:strides[0] + image_shape[0],
             strides[1]:strides[1] + image_shape[1],
             strides[2]:strides[2] + image_shape[2], :]
    return result


def infer_2d(image, window_shape, f, overlap=1, verbose=0):
    """
    :param image: a 2D or 3D numpy array, channel last, shape = (Y, X, C) or (Y, X)
    :param window_shape: a 2D numpy array
    :param f: see infer_pad
    :param overlap: see infer_pad
    :param verbose: see infer_pad
    :return:
    """
    assert len(image.shape) in [2, 3], \
        f"image shape {image.shape} is incorrect (length is {len(image.shape)} but should be 2 or 3."
    assert len(window_shape) == 2, \
        f"window_shape {window_shape} is incorrect (length is {len(window_shape)} but should be 2."

    image = np.expand_dims(image, 0)
    window_shape = (1,) + window_shape

    if isinstance(overlap, int):
        overlap = (1, overlap, overlap)
    if len(overlap) == 2:
        overlap = (1,) + overlap

    return infer(image, window_shape, f, overlap, verbose)[0]
