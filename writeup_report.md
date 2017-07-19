**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/model.png "Model Visualization"
[left]: ./images/left.jpg "Center Lane Driving"
[center]: ./images/center.jpg "Center Lane Driving"
[right]: ./images/right.jpg "Center Lane Driving"
[rec1]: ./images/recovery1.jpg "Recovery Image 1"
[rec2]: ./images/recovery1.jpg "Recovery Image 2"
[rec3]: ./images/recovery1.jpg "Recovery Image 3"
[rec4]: ./images/recovery1.jpg "Recovery Image 4"
[rec5]: ./images/recovery1.jpg "Recovery Image 5"
[rec6]: ./images/recovery1.jpg "Recovery Image 6"
[rec7]: ./images/recovery1.jpg "Recovery Image 7"
[rec8]: ./images/recovery1.jpg "Recovery Image 8"
[rec9]: ./images/recovery1.jpg "Recovery Image 9"
[rec10]: ./images/recovery1.jpg "Recovery Image 10"
[rec11]: ./images/recovery1.jpg "Recovery Image 11"
[rec12]: ./images/recovery1.jpg "Recovery Image 12"
[rec13]: ./images/recovery1.jpg "Recovery Image 13"
[rec14]: ./images/recovery1.jpg "Recovery Image 14"
[rec15]: ./images/recovery1.jpg "Recovery Image 15"
[rec16]: ./images/recovery1.jpg "Recovery Image 16"
[rec17]: ./images/recovery1.jpg "Recovery Image 17"
[rec18]: ./images/recovery1.jpg "Recovery Image 18"
[rec19]: ./images/recovery1.jpg "Recovery Image 19"
[rec20]: ./images/recovery1.jpg "Recovery Image 20"
[rec21]: ./images/recovery1.jpg "Recovery Image 21"
[rec22]: ./images/recovery1.jpg "Recovery Image 22"
[flip]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 70-93) 

The model includes ELU layers to introduce nonlinearity (code lines: 73, 78, 83, 88, 93, 100, 102, 104, 106), and the data is normalized in the model using a Keras lambda layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines: 72, 77, 82, 87, 92). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105-113). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving smoothly close to the inner edge of the curves.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the model by NVIDIA, with 5 convolutional layers and 5 dense layers after that added with some dropouts. 

My first step was to use a convolution neural network model similar to the on that was used in the video lessons of the project. I thought this model was good so I only alteres it very little.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my loss was increasing in both data sets and I noticed that this was caused by the normalization of my data. I first normalized the data between -1 and 1 but that didn't seem so good as normalizing between -0.5 and 0.5 so I changed to the latter. 

To combat the overfitting, I modified the model so that added dropouts after each convolutional layer before the ELU activations.
Then I also changed the RELU's to ELU's to achieve faster training process.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in the curves. To improve the driving behavior in these cases, I added data where the car is driven to the center of the road from the sides of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Although in some spots the car was driving quite close to the edge, I suspect that this was caused because of my training data and the fact that I trained the return from the sides of the road too close to the road.

####2. Final Model Architecture

The final model architecture (model.py lines 61-103) consisted of a convolution neural network, ELU activations and Fully connected layers boosted with dropout layers is visualized here:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded ten laps to clockwise and counterclockwise direction on track one using center lane driving. Here is an example image of center lane driving from all the three cameras:

![alt text][left] ![alt text][center] ![alt text][right]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return from the side of the track to the center. This was done for 1 round to both directions. These images show what a recovery looks like starting from the right and returning to the center:

![alt text][rec1] ![alt text][rec2] ![alt text][rec3] ![alt text][rec4] ![alt text][rec5] ![alt text][rec6] ![alt text][rec7]
![alt text][rec8] ![alt text][rec9] ![alt text][rec10] ![alt text][rec11] ![alt text][rec12] ![alt text][rec13]![alt text][rec14]
![alt text][rec15] ![alt text][rec16] ![alt text][rec17] ![alt text][rec18] ![alt text][rec19] ![alt text][rec20] 
![alt text][rec21] ![alt text][rec22] 

To augment the data set, I also flipped images and angles thinking that this would make the model more general so that it doesn't turn to the other direction all the time. For example, here is an image that has then been flipped:

![alt text][rec1] ![alt text][flip]

After the collection process, I had 39907 number of data points. And for the task I used the images from all the cameras so in total I had 119 721. Then I augmented all these files by flipping them so I had 239 442 images for the task. I then preprocessed this data by first normalizing the data between -0.5 and 0.5 and then I cropped 70 pixels from from the upper edge and 25 pixels from the lower edge of each image.

I finally randomly shuffled the data set and put 80 % of the data into a training set and 20 % into a validation set. So I was left with 191 554 images for the training and 47 888 images for validation.

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. With my ultimate model I first trained for 3 epochs and the model was behaving quite nicely so I decided to train 6 epochs and I was satisfied with the process. I used an adam optimizer so that manually training the learning rate wasn't necessary.

In addtion I used a generator to read the data from the memory, the generator is shown in the model.py code file.

#### 4. Simulation

The video of my model driving one round around the track is provided here https://github.com/Pedrous/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4. The model functions very well and stays on the track, no tires leave the drivable part of the road. Some improvement might be possible, if the training data would be recorded again and the model would be trained a little bit more. One more thing could be augment the image by doing horizontal transformation in the data.
