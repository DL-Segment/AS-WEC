This code is mainly used for brain MRI tumor segmentation

In the data folder, you can follow the URL to download the data.
In the images folder, there are two kinds of images, the original image and the segmentation result.
There are two py files in the model folder, which represent the edge recognition and segmentation model algorithms respectively.
Results are the evaluation results of each index.
The utils folder contains the relevant configuration files.
weights has the trained weights file, which can be downloaded through the link.
You can train by train.py and test the segmentation effect by test.py and predict.py.
You can use Otsu_Canny.py to segment brain MRI images and observe the edge recognition effect.
