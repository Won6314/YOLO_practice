# YOLO_practice
This is YOLO V1 implementation in pytorch.
original version of YOLO V1 used PASCAL dataset, but I tried to use COCO dataset instead.
I tried to simplify YOLO V1, only using COCO dataset's 12 supercategories.



# RUN
before run, you should download COCO dataset. then change data_root & save_root in the code.
you may run 'run.py' to train network

# Training
Instead of using 2box prediction and 20classes, I used 1box prediction and 12class(COCO supercategories). so I got 7x7x17 output, then changed it into boxes.
I trained about 24 epoches over COCO's 118,287 training data.
COCO dataset has plenty of humans, animals, foods and sports but has so little other class like indoor, appliance so network is tend to detect any object with human category.

here's the result images

![alt_text](https://github.com/Won6314/YOLO_practice/blob/master/train_images/110_result.jpg)
![alt_text](https://github.com/Won6314/YOLO_practice/blob/master/train_images/210_result.jpg)
![alt_text](https://github.com/Won6314/YOLO_practice/blob/master/train_images/330_result.jpg)
![alt_text](https://github.com/Won6314/YOLO_practice/blob/master/train_images/1870_result.jpg)
