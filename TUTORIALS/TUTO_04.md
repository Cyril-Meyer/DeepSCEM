# Tutorial 04 : Evaluate your segmentation on a test set

The results of the previous tutorial are required.
In the previous tutorial, we created a prediction with our model.

To evaluate a segmentation, you need to have a reference segmentaion (in our case, the test set) and a prediction on the same area (in our case, test_pred).

To start the evaluation, click the "evaluate prediction" button.

![image](https://github.com/user-attachments/assets/6023e6ed-2038-480e-ab53-a50523b1ffba)

If this is the first time you do this, you will get this information message.

![image](https://github.com/user-attachments/assets/b4e5b554-ae2f-446a-b597-1af2be55573d)

As said in [TUTO_01](TUTO_01.md), some features are locked in safe mode.
Evaluation is a locked feature as bad manipulation may cause a crash.

To disable safe mode, use the menu or CTRL+

![image](https://github.com/user-attachments/assets/43c1bd2c-a762-438d-b1f3-7118546cfe24)

![image](https://github.com/user-attachments/assets/a6215d64-e33a-41a0-b870-2843fc5ea9eb)

Now lets use the "evaluate prediction" button again.

![image](https://github.com/user-attachments/assets/6023e6ed-2038-480e-ab53-a50523b1ffba)

The evaluation window will open.

![image](https://github.com/user-attachments/assets/adab403c-589c-4122-86b2-5692caf34561)

Select the reference and segmentation dataset and click ok.

![image](https://github.com/user-attachments/assets/26ffeaab-eb6e-4692-b172-ea01e25e2ac8)

A result window will appear.

![image](https://github.com/user-attachments/assets/d03e6cee-f866-4d39-883e-2eafd522be23)

The upper part of the window show your results, in our case :
* F1 score for mitochondria is 94.28%
* F1 score for ER is 68.39%

The bottom part allow you to copy and paste the results into a csv file.

```
label,sample,f1,iou
0,LW4_ALL_crop_80_120_0_1394_0_1832,0.9428620042046973,0.8919005919330015
1,LW4_ALL_crop_80_120_0_1394_0_1832,0.6839979545746571,0.519754476790029
```

This is the end for this tutorial. 
