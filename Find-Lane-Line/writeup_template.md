# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied the gaussian smoothing
, next dectect edge of image, create mask, and final line is hough_transform of target. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by :
first calcualte the slope: if slope < 0, left_x1[].append(x1), else right_x1.append(x1), similar with other x2, y1, y2.
then calcualate the average of rightx1, right_y1, then draw right line. Similar with left line. 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to decect the line in every change light condition, like in the challenge.

Another potential improvement could be to ...
