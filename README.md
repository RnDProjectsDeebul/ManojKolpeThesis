# ManojKolpeThesis - Semantic segmentation using temporal fusion 

Semantic segmentation is the process of labelling each pixel in a RGB image. In general setting, for a video sequence data semantic segmentation is performed on all the frames of the video or on the key frames without fusing the information from the previous frames. The overlapping information from the past frames can be utilized to make the prediction better. This work aims to understand the fusing of information from the previous frame to the current frame predition with the help of Gaussian process and Long Short Term Memory.

<p align="center">
  <img  src="literature/test_depth.gif">
</p>

Figure 1.0 Pretrained model  [[Link](https://aaltoml.github.io/GP-MVS/)] result on the test data

<p align="center">
  <img width="480" height="283" src="literature/Glossary/Pictures/depth_estimation.gif">
</p>

Courtesy: [Link](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F21%2F1%2F54&psig=AOvVaw1Z_IIELQkBeOsCJuiD2j8k&ust=1643441741590000&source=images&cd=vfe&ved=0CAwQjhxqFwoTCODUq_Wt0_UCFQAAAAAdAAAAABAO)

<p align="center">
  <img  src="literature/Glossary/Pictures/depth.png">
</p>


Courtesy: [Link](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Fself-supervised-depth-estimation-breaking-down-the-ideas-f212e4f05ffa&psig=AOvVaw3va8tQsBacFhanuNSUk6Dk&ust=1643439567895000&source=images&cd=vfe&ved=0CAwQjhxqFwoTCPC8ueql0_UCFQAAAAAdAAAAABA2)

Model output on the generated test data

Original 

<p align="center">
  <img  src="literature/seq-03_formated_manoj_original.gif">
</p>

Predicted
<p align="center">
  <img  src="literature/seq-03_formated_manoj_predicted.gif">
</p>

Plotting of translation and quaterions (rotation matrix) extracted from the android phone

<p align="center">
  <img  src="literature/images/image1.png">
</p>
<p align="center">
  <img  src="literature/images/image2.png">
</p>
<p align="center">
  <img  src="literature/images/image3.png">
</p>

Deployment of model in the android device (Oneplus 7,GM1901,Snapdragon 855,6GB RAM)  
Source code - [Code](src/GPMVS_deployment_on_android/V2.0)

<p align="center">
  <img  src="literature/android_deployment.gif">
</p>

References

Android

[1] https://github.com/PyojinKim/ARCore-Data-Logger

[2] https://ksimek.github.io/2012/08/22/extrinsic/

[3] https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

[4] https://www.youtube.com/watch?v=zjMuIxRvygQ

[5] https://www.youtube.com/watch?v=lVjFhNv2N8o&list=PLYm0s3jier_Se9_HKG1X6rQrKFxYo8x2-&index=22

[6] https://eater.net/quaternions

