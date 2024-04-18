# Installation

## Binaries (Windows only)
You never use Python and you are not an expert in computer science ?  
No problem, you can use windows binaries to run DeepSCEM as a standalone application.
The link to download the current release of DeepSCEM is in the
[README.md](README.md) file.

To run the application, execute `DeepSCEM.exe`.

💡 If you want to speed up the training and prediction steps, you may want to setup
a [GPU](#gpu) if you got one.

💡 If you have somme missing .dll errors, you may need to install
[Microsoft Visual C++ redistributable](https://learn.microsoft.com/fr-FR/cpp/windows/latest-supported-vc-redist?view=msvc-170)
as said in the [Install TensorFlow 2](https://www.tensorflow.org/install)
official documentation.


## Run from source with Python

* Install Python (3.8 - 3.11)
* Create a virtual environment
  * `python -m venv venv`
* Activate the virtual environment
  * windows: `call venv/Scripts/activate.bat`
  * linux: `source venv/bin/activate`
* Check that your Python version is valid (3.8 - 3.11)
  * `python --version`
* Update package installer
  * `pip install -U pip setuptools wheel`
* Install requirements
  * `pip install tensorflow==2.10.1 h5py PyQt5 pyqt5-tools tifffile tqdm matplotlib numpy imagecodecs`
* Run DeepSCEM
  * `python run.py`


## GPU
If you have a GPU, you may want to use it to make faster training and prediction.

* TensorFlow official documentation
  * [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
  * [🇫🇷 Compatibilité avec les GPU](https://www.tensorflow.org/install/gpu?hl=fr)

### Windows

[CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

To check that gpu is used, use the task manager performance tab and check
GPU CUDA and memory usage :

![task-manager-gpu](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/239c8b0e-d884-4f21-9887-377a4fb9d8b1)


## Docker

There is no docker for DeepSCEM yet, but you may use an image with
Python and TensorFlow installed to run DeepSCEM from source code.

* TensorFlow official documentation
  * [Docker ](https://www.tensorflow.org/install/docker)
