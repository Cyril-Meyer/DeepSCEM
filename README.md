# DeepSCEM
Deep Segmentation for Cellular Electron Microscopy 🦠🔬⚛️.

DeepSCEM is an automatic segmentation tool integrating deep learning methods.
DeepSCEM is developed for organelles segmentation in cellular electron microscopy
image segmentation.
The toolkit is developed to be both easy to use and deploy initially,
but also very efficient and configurable for the most advanced users.


## Quick links

### [Download latest release](https://github.com/Cyril-Meyer/DeepSCEM/releases/latest)


## Bug report 🐛 & Feature request ✨

To report bugs or request a new feature, you are invited to create a new issue.


## User manual 📝
Welcome to the User Manual of DeepSCEM.
This manual will guide you through the process of using DeepSCEM,
making your experience with the software as seamless as possible.

Before we begin, here is a summary of the section
* [Installation](#installation) - A step-by-step guide to help you install DeepSCEM on your system.
* [Tutorials](#tutorials)
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

If you want to use your GPU and do not know how to install dependency, check the
[INSTALL](INSTALL.md) guide.

For more experienced users who want to run DeepSCEM from their own Python setup:
Detailed instructions on how to install and set up DeepSCEM are available in the
[INSTALL](INSTALL.md) guide.

### Tutorials

The tutorials list below will allow you to familiarize yourself
with the use of DeepSCEM by focusing on the features that interest you.
For all tutorials, sample data is also provided if you don't have one.

* [Create a dataset](TUTORIALS/TUTO_01.md)
* [Train your first model](TUTORIALS/TUTO_02.md)
* [Use your model to produce automatic segmentation](TUTORIALS/TUTO_03.md)
* [Evaluate your segmentation on a test set](TUTORIALS/TUTO_04.md)

### Usage

Our goal is to provide you with a toolkit that is not only user-friendly
but also highly efficient and customizable for advanced users.
For a user-friendly experience, you may go with the graphical user interface.
For higher efficiency, you will probably be interested in the functionality available
with command line interface.

First, if you are not familiar with deep learning, you may want to check the
[keywords](#keywords) used in this documentation.

### Graphical user interface

First, here is a view of the main window and explanation of the button features.

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
The HDF5 format file used with `h5py` do not release space when samples are removed.

💡 Use 2D models at first.  
This is more advice than a good practice, but in our experience, 2D models
was always better than 3D ones for organelles segmentation.

💡 Don't forget to save you models.  
Contrary to datasets, models are directly used in memory.
When you close DeepSCEM the unsaved models disappear.

💡 Moving the cursor on a button will show information about its purpose.
Example with cursor over "Load model".

![image](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/e8b6d01d-5f59-47d5-ba5a-b025f23f1f2e)

#### How to train models effectively
⚠ Advice from someone who has practiced a lot, potentially empirical reasoning
and biases.

1. 2D models are very effective, especially if physical data are
  [registered](https://en.wikipedia.org/wiki/Image_registration).
   Don't go with 3D model unless you have particular reason.
2. Size of model is important but not as much important as patch and batch size.
3. Prefer larger patch size than larger batch size, especially for 2D models.
4. Always use as large patch as possible for prediction.
   1. Patch size in prediction can be a lot larger than in training as we do
      not compute gradients.

### Command line interface

The command line interface (CLI) allows advanced users to process data
with single-call commands, streamlining your workflow and saving time.

List of arguments :
* `--create-dataset <filename> <name> <n_labels>`  
  Create a new empty dataset.
* `--add-sample <filename> <name> <image> [<label1> <label2> ...]`  
  Add a sample from image and labels to an existing dataset.

You can use the same arguments multiple times (excepting train),
for example, you can create 3 dataset calling 3 times `--create-dataset`
in the same row.
Whatever the number of arguments and their placement in the command line call,
they will always be evaluated in the order defined by the list of arguments above.

#### Examples
* Create an empty dataset `I3_EXAMPLE.hdf5` with name **I3** and requiring **2** labels per sample
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2`
* Add sample **i31** to dataset `I3_EXAMPLE.hdf5`
  * `--add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif `
* Create dataset with samples
  * `--create-dataset I3_EXAMPLE.hdf5 I3 2 --add-sample I3_EXAMPLE.hdf5 i31 i3_1.tif i3_label_11.tif i3_label_12.tif --add-sample I3_EXAMPLE.hdf5 i32 i3_2.tif i3_label_21.tif i3_label_22.tif`
* Train a model
  * `--train-model i3-unet-3d-bin-mito.h5 i3-unet-3d-bin-mito-train.h5 I3-MITO-BIN.hdf5 I3-MITO-BIN.hdf5 Dice 1 128 128 128 192 32 64`

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


