# Tutorial 03 : Use your model to produce automatic segmentation

The results of the previous tutorial are required.
In the previous tutorial, we created a model and trained it.

If you do not have a model loaded, load it from a saved one.
Then select the model and click the "predict" button.

![image](https://github.com/user-attachments/assets/bcffc53d-4608-4f04-8dd9-5d33953fd0fc)

The prediction window will open.

![image](https://github.com/user-attachments/assets/f26ca3fa-2443-4140-bf1a-fa5ae80bb433)

Select the model to use for prediction and the dataset to predict on (in our case the **_test** dataset).

You can increase patch size during prediction time as the model will
not required as much compute power and memory as in the training phase.
In our case, the patch size is increased from 256x256 to 512x512.

* 💡 If you use a 2D model, the Z patch size is ignored
* Overlapping force the model to predict overlaping patches to avoid problems on the border of the patches.
  * 💡 It takes more time but the resulting segmentation may be better.
* Binary threshold is used to produce binary segmentation.

![image](https://github.com/user-attachments/assets/1b04c047-30bf-43af-8992-0bf3da27770f)

When you click "ok", the prediction will start.

![image](https://github.com/user-attachments/assets/fed92364-140d-4016-a260-a6f4f1b91fdb)

This phase can be long, check the command line interface for more informations.

![image](https://github.com/user-attachments/assets/086d83d3-79c4-41e7-b7bd-8d243826db0e)

When the prediction is done, a new dataset will be created as a result.

![image](https://github.com/user-attachments/assets/9d2c3319-4504-4c81-bee3-0ee1c7d9204a)

![image](https://github.com/user-attachments/assets/8f9e638a-4c40-446e-bedb-9fd64871b8e8)

You can use DeepSCEM to visualize the prediction and to compare it with the reference manually.

![image](https://github.com/user-attachments/assets/21d424a1-fa41-4412-8cdc-6b5d2d23f2b6)

Here is the result of my running of the tutorial up to this point.

| Prediction | Reference |
|:-:|:-:|
| ![1739986823_3589](https://github.com/user-attachments/assets/de59836b-fcfc-4202-833c-64965289d244) | ![1739986833_7169](https://github.com/user-attachments/assets/983d8242-3217-4c20-a711-2835496e2a03) | 

Those images has been created using the DeepSCEM screenshot tool.

![image](https://github.com/user-attachments/assets/44ce1cf9-a5e6-4275-a393-574d9c89ff21)


This is the end for this tutorial.
For the following tutorial, go here: [TUTO_04.md](TUTO_04.md)
