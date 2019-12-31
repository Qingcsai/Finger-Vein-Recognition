# Finger-Vein-Recognition

## Background

My personal project for the final assignment of Class 'Machine Vision'

## Prerequests
This is an example of how to list things you need to use the software and how to install them.

* opencv-python
* opencv-contrib-python <= 3.4.0.10 (for SIFT & SURF algorithem can be used.)
* numpy
* matplotlib

## Usage

``` python
python vein_main.py
```

Besides, you should dive into the file **vein_main.py**, and adjust the comments for many other usages.

## Data preparation

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

## License

[MIT](LICENSE) © Richard Littauer
