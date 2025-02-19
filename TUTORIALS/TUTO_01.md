# Tutorial 01 : Create a dataset

In this tutorial, you will learn to use DeepSCEM to create a dataset from images
and annotations.

First, you need to get images and annotations. For this tutorials, we will use
the 
[IGBMC ICube LW4-1 : Image and annotations](https://zenodo.org/records/8344292)
dataset.
You will need to download the following files :
* LW4-600.tif (the image)
* Labels_LW4-600_All-Step40_mito.tif (the mitochondria annotations)
* Labels_LW4-600_1-40_81-120_Reti.tif (the endoplasmic reticulum annotatios)

### Import data

First, use the "add sample" (or CTRL+B) to add a sample to a dataset.

![image](https://github.com/user-attachments/assets/d5eedb33-b53b-483e-a7fa-734c82dec7e5)

As there is no dataset yet, your only option is to create a new one.

![image](https://github.com/user-attachments/assets/9c49268e-9339-47aa-9d54-06d54600c532)

And to give the dataset the name you choose.

![image](https://github.com/user-attachments/assets/555031df-e092-4e6b-abc2-e3f7b3a92157)

We now need to choose the number of labels used.

DeepSCEM is intended to be used by non expert user.
For this purpose, it integrates a safe mode system.
With safe mode enabled, the user is less likely to create situation that will make the software crash.
(e.g. not matching number of labels in data and in model training)

In our case, we want to work with 2 labels (mitocondria and ER) so we whoose 2.

![image](https://github.com/user-attachments/assets/185f2e6c-7f4c-4a73-93d6-219b91249bc2)
increase 1 to 2
![image](https://github.com/user-attachments/assets/12426580-3df8-4c8c-8991-b0819a0a2780)

When we will use DeepSCEM with this dataset, it will always lock (when safe mode is enabled) the system to work with 2 classes.

Then, we need to give our sample a name.

![image](https://github.com/user-attachments/assets/a221b849-ac48-4d5b-b99c-4b7c1860abfe)

Now that the sample is configured, we need to import data.
The import process consist of selecting files for the sample.
The window caption will always specify what part of the sample (image or label) is asked.

We start by giving the input image.

![image](https://github.com/user-attachments/assets/1e76c10d-999c-4e20-a54d-be4100b1a822)

Then, we import the label 0 (mitochondria).

![image](https://github.com/user-attachments/assets/5b69f488-430c-4881-9d5f-08aec95f7725)

And we do the same for label 1 (ER).
Note that the label number is "0 based", if you have 5 labels, last one will be label 4.

When the last 

![image](https://github.com/user-attachments/assets/87317433-27c5-417c-9107-56e086707e93)


### Transform whole image into train/test
