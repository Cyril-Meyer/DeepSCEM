# Tutorial 02 : Train your first model

The results of the previous tutorial are required.
In the previous tutorial, we created two dataset, each composed of a single sample with two classes.

## Create a model

In this section we will create a model and explain how to configure it.
First, click on "New model".

![image](https://github.com/user-attachments/assets/cd10bc30-ac1f-43df-a1b6-a5c8c1e21de5)

This will open the model configuration window.

![image](https://github.com/user-attachments/assets/0160496d-b3dd-42b5-9233-f833024e7ec5)

🚀 If you do not want to know more (yes, you are not required to), here is a shortlist 
of the most important steps (from most important to less) to do to change parameters for non expert.

1. Default will work fine in most cases, do not change it if you do not have an issue
2. Not enough memory ? Training is very long ? You may want a lighter model !
   * Reduce initial block filters (e.g. from 32 to 16).
3. Results are not very good ?
   * Increase initial block filters (e.g. from 32 to 64).
4. Results still not good ?
   * Try 3D model (if memory problem, see point 2).
5. My labels are supposed to overlap but does not / I never want any overlap even if model do not know how to choose
   * Activation : sigmoid allow overlapping and softmax does not.

If you want more information, here is an explanation for each parameter and how to "choose" it.
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
  * The depths of the U-Net architecture (number of time we do max pooling and process with blocks).
  * 💡 Reduce to 4 if your model do not fit in memory (e.g. for large 3D models).
* Outputs : locked in safe mode if dataset already loaded
  * Number of ouput (classes).
* Output activation : sigmoid, tanh, linear, softmax
  * 💡 Use sigmoid or softmax
    * Using sigmoid allow you to have overlapping segmentations.

When you click ok, the model will be created in memory. For this tutorial, we do not change anything in the configuration.

![image](https://github.com/user-attachments/assets/4b01d5db-7a7e-4ec2-8e25-55fd0ddc3296)

You may be interested by the output coming from the command line interface.
I you get warnings, do not panic, everything is fine, tensorflow will print a lot of things.

For example, in this case, everything is fine.

```
yyyy-mm-dd hh:mm:06.413643: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
yyyy-mm-dd hh:mm:07.258768: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5941 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080 SUPER, pci bus id: 0000:05:00.0, compute capability: 7.5
```

In the following case, you are just not using a GPU.

```
yyyy-mm-dd hh:mm:10.441044: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
yyyy-mm-dd hh:mm:10.441660: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
yyyy-mm-dd hh:mm:15.640560: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
yyyy-mm-dd hh:mm:15.641154: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
yyyy-mm-dd hh:mm:15.641748: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
yyyy-mm-dd hh:mm:15.642322: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
yyyy-mm-dd hh:mm:15.642883: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
yyyy-mm-dd hh:mm:15.643449: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
yyyy-mm-dd hh:mm:15.644064: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
yyyy-mm-dd hh:mm:15.644624: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
yyyy-mm-dd hh:mm:15.644710: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
yyyy-mm-dd hh:mm:15.646240: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

Always read the messages. To know if things are serious or not, your favorite search engine will redirect you to more informations.

Now, thats it, you have a model.

![image](https://github.com/user-attachments/assets/abe6e40c-4e70-4bc6-906e-1438fee009ca)

Note that this model is not trained yet.
That means that the neural network has not yet tuned its weights and cannot produce a segmentation.


## Train a model

To train a model, click the "Train" button.

![image](https://github.com/user-attachments/assets/41aac493-fce5-40ae-bb45-19db9ca06a75)

The train window will open to configure the training.

![image](https://github.com/user-attachments/assets/2806db76-42af-4139-a671-e84284494031)

🚀 If you do not want to know more (yes, you are still not required to), here is a shortlist 
of the most important steps to do to.

1. Default settings are ok to start with but need tuning each time (always do point 2 and 3)
2. Always check that selected dataset (Train / Valid) are the good ones.
   * If you do not want to use valid, use Train as valid.
   * **NEVER USE TEST AS VALID WITH KEEP BEST**
3. Use label focus, most of time, 75% is a good value.
4. Reduce epochs / steps if training is too long for you.

If you want more information, here is an explanation for each parameter and how to "choose" it.

* Model
  * The model we created in previous step
  * 💡 Check only if you have more than one model loaded.
* Train
  * The train dataset will be used as examples used to change the model weights
  * 💡 Always double check this.
* Valid
  * The valid dataset will be used to evaluate the model during the training phase
    (and to select the best model during this phase) but the network will not have seen this data directly.
  * 💡 Always double check this.
* Loss function : Dice / CrossEntropy / MeanSquarredError
  * The function used to compute the error between the model prediction and the reference segmentation.
  * 💡 Dice is good most of time
* Batch size
  * The number of patches given to the network each step.
  * 💡 Prefer larger patch size than larger batch size, especially for 2D models.
* Patch size
  * The size of patches given to the network each step.
  * 💡 Avoid non square shapes in 2D.
* Steps per epoch
  * The number of batches in one epoch.
* Epochs
  * The number of epochs.
* Validation per epoch
  * The number of batches to use to compute validation score at the end of each epoch.
  * Keep best (checkbox)
    * Save a copy of the best model weights (best = best score on valid).
    * 💡 Use it in addition to manually saving the last model to have two model at the end.
  * Early stopping (checkbox)
    * Stop after some times if the model does not improve anymore.
* Augmentation
  * Apply transformation to train and valid to add more variability and have a more generalized model. 
  * Rotation (checkbox).
  * Flip (checkbox).
* Label focus
  * Patches are taken randomly from the dataset samples.
  * Label focus is the percentage of patches that are not taken completely randomly but which
    are garanted to contains label in the center of the patches.
  * 💡 Use label focus, most of time, 75% is a good value.

For this tutorial, I made the following changes :
* Select **_train** dataset as Train and Valid
* Reduce batch size from 8 to 4 (no really reason to this but to give example of changes)
* Reduce epochs from 64 to 32 (this will make training a lot faster)
* Use label focus 75%

If you do not have a GPU (or do not have configured it yet) reduce patch size from 256x256 to 128x128 and epochs to 16.

| Before | After |
|:-:|:-:|
| ![image](https://github.com/user-attachments/assets/2806db76-42af-4139-a671-e84284494031)| ![image](https://github.com/user-attachments/assets/9785f80d-face-460e-b2eb-94422910d684) |

When you click Ok, the training phase will start.

![image](https://github.com/user-attachments/assets/4a3a732d-8b06-4fb1-88f6-1ce982f0f5a1)

This phase can be long, check the command line interface for more informations.

![image](https://github.com/user-attachments/assets/6d1a8559-dc20-4f52-b6a3-ceef016151b0)

![image](https://github.com/user-attachments/assets/fe3fa86f-a87c-484f-bbc7-c7aa268d2dfe)

When the training is done, you can see in your directory that the "best on valid" model has been saved, but not your current model.

![image](https://github.com/user-attachments/assets/594d5e99-9daa-4fcd-9941-e45190f87150)

Always remember : DeepSCEM do not save your models automatically.

So now, lets save the model by selecting it and clicking the "save model" button.

![image](https://github.com/user-attachments/assets/52eba7bf-90f8-4d48-9711-03a7994c4300)

And save the model with another name than the "best on valid" model.

![image](https://github.com/user-attachments/assets/575b8308-4e5c-45f6-896f-cdae52bd5fd3)


This is the end for this tutorial.
For the following tutorial, go here: [TUTO_03.md](TUTO_03.md)
