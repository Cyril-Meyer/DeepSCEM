# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy


## Work in progress
#### Essential modules

* [x] data
  * [x] data I/O
  * [x] dataset creation
* [ ] train
  * [x] model creation
  * [ ] model training
  * [ ] patch generation
  * [ ] loss functions
* [ ] inference (predict)
  * [ ] model load
  * [ ] predict
  * [ ] image prediction with overlap

#### User Interface

* CLI
  * replace current second stage parser with subparser
  * [x] data
  * [ ] train
  * [ ] predict
* UI
  * [x] data
  * [x] view
  * [ ] train
  * [ ] predict

#### Refactoring

* data
  * need to set number of labels in the dataset file attribute
    * added, but not checked yet.
* UI
  * Better handling of long operation (using thread)
* Current code is prone to user input error


## Keywords
* Image : A 3D or 2D grayscale image (.tif, .tiff or .npy)
* Label : A 3D or 2D binary image (.tif, .tiff or .npy)
* Labels : One or more **label**
* Sample : An image / labels combination
* Samples : One or more **sample**
* Dataset : A collection (= a list) of samples
* Architecture : A type of network (e.g. U-Net, FCN)
* Model : A deep neural network (it's layers and weights)


## Examples
* Create an empty dataset `I3_EXAMPLE.hdf5` with name **I3** and requiring **2** labels per sample
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2`
* Add sample **i31** to dataset `I3_EXAMPLE.hdf5`
  * `--add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif `
* Create dataset with samples
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2 --add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif --add-sample I3_EXAMPLE.hdf5 i32 i3_2.tif i3_label_21.tif i3_label_22.tif`


### Requirements
* Python 3.8 - 3.11
* TensorFlow tensorflow-2.10.1 or tensorflow-2.10.1
  * `pip install tensorflow==2.10.1` : deep learning framework
  * TensorFlow 2.10 is the last TensorFlow release that supported GPU on native-Windows.
* `pip install h5py PyQt5 pyqt5-tools tifffile matplotlib numpy imagecodecs`
  * `numpy` : linear algebra = array
  * `tifffile` : read and write `.tif` and `.tiff` files
  * `imagecodecs` : codecs for images
  * `h5py` : read and write HDF5 files
  * `PyQt5` : graphical user interface
  * `pyqt5-tools` : PyQt5 tools (designer)
    * required to run `pyqt5-tools designer`
  * `matplotlib` : view outside user interface
