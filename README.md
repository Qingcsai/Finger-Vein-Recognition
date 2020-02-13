# Finger-Vein-Recognition

## Background

My personal project for the final assignment of Class *Machine Vision Application* in SCUT.

## Requirements

* opencv-python
* opencv-contrib-python <= 3.4.0.10 (for SIFT & SURF algorithem can be used.)
* numpy
* matplotlib

## Descriptions

Here is the structure of the whole system:

![系统结构图](https://github.com/Qingcsai/Finger-Vein-Recognition/blob/master/README_images/system.png)

We use the traditional ways to process the images, rather than the deep-learning methods.

After your running ```vein_main.py```, you should see the histogram of the scores between inter-class and in-class. 

![特征匹配得分直方图](https://github.com/Qingcsai/Finger-Vein-Recognition/blob/master/README_images/histogram.png)

So we can set the threshold value to 60 for classification.

## Usage

``` python
python vein_main.py
```

Besides, you should dive into the file ```vein_main.py```, and adjust the comments for many other usages.

## Data preparation

I didn' t upload all of my own vein data for individual privacy.  
You should place your own vein data in the ```./data/600/2``` folder and name it like the format below.

```
├──data  
│   ├── 600                        // A Person's vein image  
│   │   ├── 1                      // the first machine  
│   │   ├── 2                      // the second machine   
│   │   │   ├── 600-1-1-1.bmp  
│   │   │   ├── 600-1-2-1.bmp  
│   │   │   ├── 600-1-3-1.bmp  
│   │   │   ├── ...  
│   │   │   ├── 600-2-1-1.bmp  
│   │   │   ├── 600-2-2-1.bmp  
│   │   │   ├── 600-2-3-1.bmp  
│   │   │   ├── ...  
│   ├── roi_600_2_all_320240       //saved ROI  
│   │   ├── 600-1-1-1.bmp  
│   │   ├── 600-1-2-1.bmp  
│   │   ├── 600-1-3-1.bmp  
│   │   ├── ...   
```

## License
The MIT License ([MIT](https://mit-license.org/))   
Copyright © 2020 <https://github.com/Qingcsai>
