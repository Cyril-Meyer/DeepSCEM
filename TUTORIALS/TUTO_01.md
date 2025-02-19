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

⚠️ If your computer has not a lot of RAM you may want to crop the images first.  
The tutorial has been made with a computer equiped with 32G of RAM.

<details> 
  <summary>For this purpose, you may use ImageJ / Fiji to keep only the 120 first slices.</summary>
  <img src="https://github.com/user-attachments/assets/8e7c957d-54e3-4d7b-a2d0-f922f3a232be">
</details>


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

When the last label is selected, DeepSCEM will load all the data at once.
This process may be long (depend on the size of the images and label) but also use a lot of the computer memory.

![image](https://github.com/user-attachments/assets/87317433-27c5-417c-9107-56e086707e93)

When the loading process is done, the window will be unlocked, and you will be able to view the loaded data.

![image](https://github.com/user-attachments/assets/da3d3770-59d5-47d7-a8fe-b66150ce6957)

In our case, we select the two labels to check that everything seems ok.

![image](https://github.com/user-attachments/assets/1a96a8ce-e5ae-4a57-abc9-a2e5843f9bca)

During this process, DeepSCEM automatically saved the current dataset.

![image](https://github.com/user-attachments/assets/d704e52f-a5cb-4070-a788-bdab0e45135c)

In reality, DeepSCEM always works live on the data files.
This is advantageous because most of the time, the data is not loaded into memory and in case of crash, the data are safe.
However, this also has disadvantages depending on the use cases


### Transform whole image into train/test

Now  that we have our image loaded, we need to transform it into usable datasets for the training process.
Our current dataset is composed of missing labels (only some slices are annotated).

In this section, we will transform the dataset into two small dataset.
We want the following result :
* Train : slice 0 to 40
* Test : slice 80 to 120

First, select the sample (select LW4_ALL in the list) and open the crop tool (menu Data > Crop sample or CTRL+C).

![image](https://github.com/user-attachments/assets/ce59deab-5094-4f34-9b97-2b5889722059)

The crop tool will ask where to start and stop along each axis.
The default value correspond to not cropping anything.

In our case we want to crop only along the Z axis, so we only need to changes thoses values.
We crop in Z from 0 to 40.

![image](https://github.com/user-attachments/assets/2633eb1c-356d-4c49-a6f3-c89b8b3a3c7e)

After the first crop, a new sample will be created in the dataset.

We do the same for the second crop (Z from 80 to 120)

80 ![image](https://github.com/user-attachments/assets/c52d30ba-bb09-4e1c-b5ad-7f13718942f9) to 120 ![image](https://github.com/user-attachments/assets/d9534021-56cf-4515-aca6-dd282999ac2e)

Now, we have a dataset with 3 samples.
The sample names give us information about the crops parameters.

![image](https://github.com/user-attachments/assets/6e8bed55-8d7d-4f7f-96b9-7a7ab6d28127)

We can rename any sample by double clicking, using menu Data > Rename or using CTRL+R.x

Now, we can delete the LW4_ALL sample.

![image](https://github.com/user-attachments/assets/dcc733e7-e69f-45f1-99c5-a4c6184e2304)

And save the current data in a new file. (in our case, create two copy, one for train and one for test).

![image](https://github.com/user-attachments/assets/2495a06a-a6c0-4c73-bed3-4d7ee08d640e)

Note that the newly created save is lighter than the previous file.
When deleting samples, the samples data will not be cleaned, but you can solve this by saving into a new file.

![image](https://github.com/user-attachments/assets/3cd28138-74a0-40cc-9df8-53b0e6df058d)


### Split to sample in two datasets

Most of time, we want to use datasets with multiple samples, that is why DeepSCEM implement this system of samples and crops.
But for this tutorial, we want two datasets composed of a single sample.

Splitting a dataset is not straightforward as DeepSCEM always work directly on the data.
The easiest way to do this is to save multiples copy of the dataset with both samples and to delete the not wanted dataset in each copy.

First unload the dataset.

![image](https://github.com/user-attachments/assets/a3c370d6-8d30-4955-98d9-9e9cdcd852e2)

And load the train copy.

![image](https://github.com/user-attachments/assets/31f5c986-6d37-46e8-95b4-79862e8ec298)


![image](https://github.com/user-attachments/assets/30e2207d-2ee4-45a1-b77c-d86760f36380)
