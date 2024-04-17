# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy.

DeepSCEM is an automatic segmentation tool integrating deep learning methods.
DeepSCEM is developed to respond to the problem of cellular electron microscopy image segmentation.
The toolkit is developed to be both easy to use and deploy initially, but also very efficient and configurable for the most advanced users.


## Installation and requirements

### Windows binaries

### Linux and windows

#### Requirements
* Python 3.8 - 3.11
* TensorFlow tensorflow-2.10.1 or tensorflow-2.10.1
  * `pip install tensorflow==2.10.1` : deep learning framework
  * TensorFlow 2.10 is the last TensorFlow release that supported GPU on native-Windows.
* `pip install h5py PyQt5 pyqt5-tools tifffile tqdm matplotlib numpy imagecodecs`
  * `numpy` : linear algebra = array
  * `tifffile` : read and write `.tif` and `.tiff` files
  * `imagecodecs` : codecs for images
  * `h5py` : read and write HDF5 files
  * `PyQt5` : graphical user interface
  * `pyqt5-tools` : PyQt5 tools (designer)
    * required to run `pyqt5-tools designer`
  * `matplotlib` : view outside user interface
  * `tqdm` : command line progress view

### Use a GPU
Windows user :

# User manual

## Keywords
Definitions for keywords used in this documentation.

* Image : A 3D or 2D grayscale image (.tif, .tiff or .npy)
* Label : A 3D or 2D binary image (.tif, .tiff or .npy)
* Labels : One or more **label**
* Sample : An image / labels combination
* Samples : One or more **sample**
* Dataset : A collection (= a list) of samples
* Architecture : A type of network (e.g. U-Net, FCN)
* Model : A deep neural network (it's layers and weights)


# Developers

## Work in progress
#### Essential modules

* [x] data
  * [x] data I/O
  * [x] dataset creation
* [ ] train
  * [x] model creation
  * [ ] model training
    * [ ] loss functions
    * [ ] metrics
    * [ ] valid / test output
  * [x] patch generation
* [x] inference (predict)
  * [x] model load
  * [x] predict
  * [x] image prediction with overlap

#### User Interface

* CLI
  * [x] data
  * [ ] train
  * [ ] predict
  * replace current second stage parser with subparser
* UI
  * [x] data
  * [x] view
  * [x] train
  * [x] predict

#### Refactoring

* data
  * need to set number of labels in the dataset file attribute
    * added, but not checked yet
* train
  * number of labels in data and model can be different and cause problems
  * validation steps set to 0 is not tested yet  
* pred
  * use infer without pad when it's possible
* UI
  * Check if `GenericWorker` create major performance issue
* Current code is prone to user input error


## CLI Examples
* Create an empty dataset `I3_EXAMPLE.hdf5` with name **I3** and requiring **2** labels per sample
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2`
* Add sample **i31** to dataset `I3_EXAMPLE.hdf5`
  * `--add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif `
* Create dataset with samples
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2 --add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif --add-sample I3_EXAMPLE.hdf5 i32 i3_2.tif i3_label_21.tif i3_label_22.tif`
