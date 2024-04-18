# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy 🦠🔬⚛️.

DeepSCEM is an automatic segmentation tool integrating deep learning methods.
DeepSCEM is developed for organelles segmentation in cellular electron microscopy
image segmentation.
The toolkit is developed to be both easy to use and deploy initially,
but also very efficient and configurable for the most advanced users.


## Quick links

* Download latest release (todo)


## Bug report 🐛 & Feature request ✨

To report bugs or request a new feature, you are invited to create a new issue.


## User manual 📝
Welcome to the User Manual of DeepSCEM.
This manual will guide you through the process of using DeepSCEM,
making your experience with the software as seamless as possible.

Before we begin, here is a summary of the section
* [Installation](#installation) - A step-by-step guide to help you install DeepSCEM on your system.
* [Usage](#usage) - Learn how to use DeepSCEM effectively
  * [Graphical user interface](#graphical-user-interface) - An overview of the GUI features and how to use them
  * [Command line interface](#command-line-interface) - Use DeepSCEM via command-line commands for advanced users.
* [Keywords](#keywords) - A glossary of important terms and keywords

### [Installation](INSTALL.md)
To use DeepSCEM, the first step is to install it on your system.
We've made this process simple and user-friendly for everyone,
regardless of your technical expertise.

For users who prefer a straightforward installation, we provide a binary release
for Windows.
Simply download the zip file in [quick links](#quick-links), extract it, and you're ready to go.

If you want to use your GPU and do not know how to install dependecy, check the
[INSTALL](INSTALL.md) guide.

For more experienced users who want to run DeepSCEM from their own Python setup:
Detailed instructions on how to install and set up DeepSCEM are available in the
[INSTALL](INSTALL.md) guide.


### Usage

Our goal is to provide you with a toolkit that is not only user-friendly
but also highly efficient and customizable for advanced users.
For a user-friendly experience, you may go with the graphical user interface.
For higher efficiency, you will probably be interested in the functionality available
with command line interface.

First, if you are not familiar with deep learning, you may want to check the
[keywords](#keywords) used in this documentation.

### Graphical user interface

First, here is an view of the main window and and explanation of the button features.

![image](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/5e51e835-df4b-4c63-9871-190958285d8f)

1. Load dataset
2. Unload dataset
3. Add sample to dataset
4. Remove sample from dataset
5. Save dataset as
6. List of loaded dataset and their samples
7. Load model
8. Create a new model
9. Train model
10. Predict model
11. Evaluate model
12. Save model as
13. Lost of loaded models
14. 3D images Z selector
15. MainView correspond to the area for data visualization


#### Good practice
⚠ Never store your dataset and models in the working dir.  
When working with dataset, DeepSCEM create dataset file in realtime.
This is very convenient to avoid in memory storage, but it may also rewrite
existing files.  
The correct workflow is to export (save) your dataset once you're happy with it.

💡 Save dataset again if something changed.
The HDF5 format file used with h5py may not release space when samples are removed.

💡 Use 2D models at first.  
This is more advice than a good practice, but in our experience, 2D models
was always better than 3D ones for organelles segmentation.

💡 Don't forget to save you models.  
Contrary to datasets, models are directly used in memory.
When you close DeepSCEM the unsaved models disappear.

💡 Overring a widget will show information about it. Example with cursor over "Load model".

![image](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/e8b6d01d-5f59-47d5-ba5a-b025f23f1f2e)


### Command line interface

The command line interface (CLI) allows advanced users to process data
with single-call commands, streamlining your workflow and saving time.

List of arguments :
* `--create-dataset <filename> <name> <n_labels>`  
  Create a new empty dataset.
* `--add-sample <filename> <name> <image> [<label1> <label2> ...]`  
  Add a sample from image and labels to an existing dataset.

#### Examples
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
* Working dir : The directory from where you have started DeepSCEM


## Developers 💻

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
  * [ ] test
  * replace current second stage parser with subparser
* UI
  * [x] data
  * [x] view
  * [x] train
  * [x] predict
  * [ ] test

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
* Current code is prone to user input error
