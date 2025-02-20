# Changelog

**v2025.02.20-17.29**
* Fixed a pyinstaller build bug for linux causing crash

**v2025.02.19-14.17**
* Change GitHub build NumPy requirements to specific version to fix a crash when creating a new model.


## Developers notes 💻

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
    * `edt` : euclidean distance transform
  * Data
    * `h5py` : read and write HDF5 files (our dataset format)
    * `tifffile` : read and write `.tif` and `.tiff` files
    * `imagecodecs` : codecs for compressed images
  * Graphical user interface
    * `PyQt5` : Qt5 Python bindings for GUI
    * `pyqt5-tools` : PyQt5 tools (designer)
      * required to run `pyqt5-tools designer`
  * Install everything `pip install edt h5py imagecodecs PyQt5 pyqt5-tools tifffile tqdm matplotlib numpy`
* PyInstaller
  * Packaged DeepSCEM app without installing a Python interpreter or any modules
  * `pip install pyinstaller` : only for binary release


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
* [x] test
  * [x] metrics

#### Tested

* Models
  * [x] 2D binary (256 x 256 patches)
  * [x] 2D multiclass (256 x 256 patches)
  * [x] 3D binary (32 x 128 x 128 patches, 16-4 model)
  * [x] 3D multiclass (32 x 128 x 128 patches, 16-4 model)
* Loss
  * [x] Dice (binary and multiclass)
  * [x] CrossEntropy (binary)
* Activation
  * [x] sigmoid
  * [x] softmax

#### User Interface

* CLI
  * [x] data
  * [ ] create model
  * [x] train
  * [ ] predict
  * [ ] test
* UI
  * [x] data
    * [x] rename
  * [x] view
  * [x] create model
  * [x] train
  * [x] predict
  * [x] test
  * [x] safe mode

#### Refactoring

* data
  * choice to load data as float16 or float32
  * alert when label is not in {0, 1}
  * need to set number of labels in the dataset file attribute
    * added, but not checked yet
* train
  * number of labels in data and model can be different and cause problems
  * validation steps set to 0 is not tested yet
  * better early stopping parameters
* prediction
  * use infer without pad when it's possible
* test
  * user can do a lot of bad choice and this will result in crash
  * related to previous, user cannot compare dataset if it is not label and pred
* Current code is prone to user input error

#### Documentation

* [x] user manual for gui (list of functionality)
* [x] user manual for cli (list of functionality)
* [x] good practice and how to train models effectively
* [x] tutorials
  * [x] Create a dataset
  * [x] Train your first model
  * [x] Use your model to produce automatic segmentation
  * [x] Evaluate your segmentation on a test set
* [ ] video tutorials

#### Know bugs

* prediction padding when matching shapes
  * usage of `pred.infer_pad` should be replaced with a `infer` when precondition
    are met.
* unexpected nan loss with large 3D patches (dice and bce experimented)
  3D models are most time very bad or does not converge, this may not be a bug.
  * create 3D model (default but 16 filter at start)
  * `python ../run.py --train-model i3-unet-3d-bin-mito.h5 i3-unet-3d-bin-mito-train.h5 I3-MITO-BIN.hdf5 I3-MITO-BIN.hdf5 Dice 1 96 192 192 128 32 64`
  * after a few steps, you get a nan loss
    * `4/128 [..............................] - ETA: 1:13 - loss: nan`
