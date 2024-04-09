# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy


## Work in progress
#### Essential modules

* data
  * [x] data load
  * [ ] dataset creation
* train
  * model creation
  * model training
  * patch generation
  * loss functions
* inference (predict)
  * model load
  * predict
  * image prediction with overlap

#### User Interface

* CLI
  * Repair or remove the broken dataset and sample creation
  * argparse to run with arguments
    * replace current second stage parser with subparser
* UI

#### Refactoring

* data
  * code (sample and dataset) need comments and cleaning


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
* Create samples
  * `--create-sample name1 image1 label11 label12 --create-sample name2 image2 label21 label22 --create-sample name3 image3`
  * `--create-sample i31 i3_1.tif i3_label_11.tif i3_label_12.tif --create-sample i32 i3_2.tif i3_label_21.tif i3_label_22.tif`
* Create datasets
  * `--create-dataset i3 i31 i32`


### Requirements
* Python 3.8 - 3.11
* TensorFlow tensorflow-2.10.1 or tensorflow-2.10.1
  * `pip install tensorflow==2.10.1` : deep learning framework
  * TensorFlow 2.10 is the last TensorFlow release that supported GPU on native-Windows.
* `pip install h5py PyQt5 pyqt5-tools tifffile numpy`
  * `numpy` : linear algebra = array
  * `tifffile` : read and write `.tif` and `.tiff` files
  * `h5py` : read and write HDF5 files
  * `PyQt5` : graphical user interface
  * `pyqt5-tools` : PyQt5 tools (designer)
    * required to run `pyqt5-tools designer`
