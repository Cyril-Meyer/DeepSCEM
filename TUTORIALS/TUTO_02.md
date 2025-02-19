# Tutorial 02 : Train your first model

The results of the previous tutorial are required.
In the previous tutorial, we created two dataset, each composed of a single sample with two classes.

## Create a model

In this section we will create a model and explain how to configure it.
First, click on "New model".

![image](https://github.com/user-attachments/assets/cd10bc30-ac1f-43df-a1b6-a5c8c1e21de5)

This will open the model configuration window.

![image](https://github.com/user-attachments/assets/0160496d-b3dd-42b5-9233-f833024e7ec5)

Here is an explanation for each parameter and how to "choose" it.
* Model dimension : 2D or 3D
  * Do you want your model input and output to be 2D (square patches of slices) or 3D (cube patches) ?
  * Most of time 2D works good (sometimes better than 3D) and are much more lightweight.
  * 💡 In our experience, 2D models was always better than 3D ones for organelles segmentation.
* Architecture : U-Net
  * This option is only present for further work (e.g. if someone want to implement its own architecture).
* Construction block : Residual or VGG
  * The residual block add residual connection from the input of each block to the output.
  * 💡 Always use residual.
* Kernel size
  * The size of the convolution layer kernels.
  * 💡 Do not touch this.
* Initial block filters
  * Number of filters in the first block of filters.
  * Number of filters in the next blocks depends of this values.
    * `initial_block_depth * 2^level`.
  * 💡 Use 32 most of time and increase to 64 or 128 for larger models.
* Block per level
  * Number of "processing" (conv + batchnorm) block per level.
  * 💡 Do not touch this.
* Normalization : BatchNorm or None
  * Normalization layer to use in the processing blocks.
  * 💡 Always use BatchNorm.
* Model depths (levels)
  * The depths of the U-Net architecture (number of time we do max pooling and process with blocks)
  * 💡 Reduce to 4 if your model do not fit in memory (e.g. for large 3D models)
* Outputs : locked in safe mode if dataset already loaded
  * Number of ouput (classes)
* Output activation : sigmoid, tanh, linear, softmax
  * 💡 Use sigmoid or softmax
    * Using sigmoid allow you to have overlapping segmentations
   


## Train a model



This is the end for this tutorial.
For the following tutorial, go here: [TUTO_03.md](TUTO_03.md)
