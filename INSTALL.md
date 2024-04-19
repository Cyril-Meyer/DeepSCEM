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
as stated in [Install TensorFlow 2 Documentation](https://www.tensorflow.org/install).


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
This guide only work for NVIDIA GPU.

### Windows
First, you need to have working NVIDIA GPU and its driver installed.
This is normally the case on your computer.
We need to install cuda (>=11.2) and cuDNN (>=8.1) for TensorFlow 2.10.

We will install cuda using
[conda](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#using-conda-to-install-the-cuda-software).
You can install miniconda which is the lightest distribution for conda.

1. Install [miniconda](https://docs.anaconda.com/free/miniconda/)
2. Start conda prompt  
![miniconda prompt](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/6908f7ab-c5ba-404c-9ec6-4ff594afdb50)
3. Install CUDA and cuDNN `conda install cudatoolkit==11.3.1 cudnn==8.2.1`
4. Run DeepSCEM with miniconda prompt `DeepSCEM.exe` or `python run.py`

To check that gpu is used durint training,
use the task manager performance tab and check
GPU CUDA and memory usage :

![task-manager-gpu](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/239c8b0e-d884-4f21-9887-377a4fb9d8b1)

To use your GPU, the only important thing,
is to activate a conda environment with cudatoolkit and cudnn installed.
You can install requirements in the conda environment or
activate a previously created venv after starting the conda prompt.

Example of my "Windows test" setup :

![image](https://github.com/Cyril-Meyer/DeepSCEM/assets/69190238/60524ee3-0422-4edc-b48b-61aaaf5eb334)


## Docker

There is no docker for DeepSCEM yet, but you may use an image with
Python and TensorFlow installed to run DeepSCEM from source code.

* TensorFlow official documentation
  * [Docker ](https://www.tensorflow.org/install/docker)


## References
* [tensorflow.org/install/source/gpu](https://www.tensorflow.org/install/source#gpu)
* [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
* [🇫🇷 Compatibilité avec les GPU](https://www.tensorflow.org/install/gpu?hl=fr)
* [NVIDIA CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
