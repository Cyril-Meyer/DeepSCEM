# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy.

DeepSCEM is an automatic segmentation tool integrating deep learning methods.
DeepSCEM is developed for organelles segmentation in cellular electron microscopy
image segmentation.
The toolkit is developed to be both easy to use and deploy initially,
but also very efficient and configurable for the most advanced users.


## User manual
Welcome to the User Manual of DeepSCEM.
This manual will guide you through the process of using DeepSCEM,
making your experience with the software as seamless as possible.

Before we begin, here is a summary of the section
* [Installation](#installation)
* [Usage](#usage)
  * [Graphical user interface](#graphical-user-interface)
  * [Command line interface](#command-line-interface)
* [Keywords](#keywords)

### [Installation](INSTALL.md)


### Usage

Our goal is to provide you with a toolkit that is not only user-friendly
but also highly efficient and customizable for advanced users.

If you are not familiar with deep learning, you may want to check the
[keywords](#keywords)



#### Graphical user interface



#### Command line interface
##### CLI Examples
* Create an empty dataset `I3_EXAMPLE.hdf5` with name **I3** and requiring **2** labels per sample
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2`
* Add sample **i31** to dataset `I3_EXAMPLE.hdf5`
  * `--add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif `
* Create dataset with samples
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2 --add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif --add-sample I3_EXAMPLE.hdf5 i32 i3_2.tif i3_label_21.tif i3_label_22.tif`


#### Keywords
Definitions for keywords used in this documentation.

* Image : A 3D or 2D grayscale image (.tif, .tiff or .npy)
* Label : A 3D or 2D binary image (.tif, .tiff or .npy)
* Labels : One or more **label**
* Sample : An image / labels combination
* Samples : One or more **sample**
* Dataset : A collection (= a list) of samples
* Architecture : A type of network (e.g. U-Net, FCN)
* Model : A deep neural network (it's layers and weights)


## Bug report & Feature request


## Developers

### Requirements explained

* Python 3.8 - 3.11
  * Python version compatible with TensorFlow 
* TensorFlow 2.10.1
  * `pip install tensorflow==2.10.1` : deep learning framework
  * TensorFlow 2.10 is the last TensorFlow release that supported GPU on native-Windows.
* Python packages
  * General
    * `numpy` : linear algebra and arrays
    * `matplotlib` : view outside user interface, may be used for debug
    * `tqdm` : command line progress view
  * Data
    * `h5py` : read and write HDF5 files (our dataset format)
    * `tifffile` : read and write `.tif` and `.tiff` files
    * `imagecodecs` : codecs for compressed images
  * Graphical user interface
    * `PyQt5` : Qt5 Python bindings for GUI
    * `pyqt5-tools` : PyQt5 tools (designer)
      * required to run `pyqt5-tools designer`
  * Install everything `pip install h5py imagecodecs PyQt5 pyqt5-tools tifffile tqdm matplotlib numpy`


### Work in progress
#### Essential modules

* [x] data
  * [x] data I/O
  * [x] dataset creation
* [x] train
  * [x] model creation
  * [x] model training
    * [x] patch generation
* [x] inference (predict)
  * [x] model load
  * [x] predict
  * [x] image prediction with overlap
* [ ] test
  * [ ] metrics

#### Tested

* [x] 2D binary
* [x] 2D multiclass
* [ ] 3D binary
* [ ] 3D multiclass
* Loss (tested in binary)
  * [x] Dice
* Activation (tested in binary)
  * [x] sigmoid

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
  * better early stopping parameters
* pred
  * use infer without pad when it's possible
* UI
  * Check if `GenericWorker` create major performance issue
* Current code is prone to user input error
