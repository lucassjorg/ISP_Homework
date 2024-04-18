# ISP_Homework README

## How to use this code:

## 1: Run main.py.  
This will print information about the image, the double-precision arrays at each step, and the compression ratios for converting the image to a jpeg.  It will also save png images, jpeg uncompressed images, and jpeg compressed images of the white world balancing, gray world balancing, and camera preset balancing algorithms in respective folders. 

## 2: Pick points for manual white balancing
A plot with the image will be shown after, and the user needs to click in two spots that appear white, which will cause the images to be manually white balanced.  If the average of a rectangle between the selected points is not white, the image will not look right.  These new images will then be saved to the manual_white_balancing folder. 
