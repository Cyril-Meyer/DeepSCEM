# Installation and requirements

## Binaries (Windows only)
You never use Python and you are not an expert windows computers ?  
No problem, you can use windows binaries to run DeepSCEM as a simple application !

If you have somme missing .dll errors, you may need to install
[Microsoft Visual C++ redistributable](https://learn.microsoft.com/fr-FR/cpp/windows/latest-supported-vc-redist?view=msvc-170).


## Run from source with Python

* Install Python (3.8 - 3.11)
* Create a virtual environment
  * `python -m venv venv`
* Activate the virtual environment
  * windows: `call venv/Scripts/activate.bat`
  * linux: `source venv/bin/activate`
* Check that your Python version is valid (3.8 - 3.11)
  * `python --version`
* Install requirements
  * `pip install tensorflow==2.10.1 h5py PyQt5 pyqt5-tools tifffile tqdm matplotlib numpy imagecodecs`
* Run DeepSCEM
  * `python run.py`


## GPU
If you have a GPU, you may want to use it to make faster training and prediction.

### Windows

### Linux


## Docker
