# **Behavioral Cloning**



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_cropping]: ./examples/cropping.PNG "Cropped Image"
[image_CNNStructure]: ./examples/CNNStructure.PNG "CNN Structure"
[image_Center]: ./examples/Center.PNG "Center Image"
[image_flipping]: ./examples/flipping.PNG "Flipped Image"
[image_histogram]: ./examples/histogram.PNG "Histogram of steering angle"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* run1.mp4 shows the car driving around track 1 autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the nVIDIA model. It consists of:

1. Lambda Layer to normalize the input data to the range [-1, 1].
2. Cropping Layer that crops 60 pixels from the top an 25 pixels from the bottom. The following image shows an original image (left) and a cropped image (right).
![alt text][image_cropping]
3. Convolution Layer: kernel: 5x5, depth: 24, activation: elu, max pooling 2x2
4. Convolution Layer:  kernel: 5x5, depth: 36, activation: elu, max pooling 2x2
5. Convolution Layer: kernel 5x5, depth: 48, activation: elu, max pooling 2x2
5. Convolution Layer: kernel 3x3, depth: 64, activation: elu, max pooling 2x2
5. Convolution Layer: kernel 3x3, depth: 64, activation: elu, max pooling 2x2
6. Flatten Layer
7. Dense Layer: 100, activation: elu
8. Dropout Layer: dropout rate = 0.1
9. Dense Layer: 50, activation: elu
10. Dense Layer: 10, activation: elu
11. Dense Layer: 1


#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (clockwise, counterclockwise driving, image flipping). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, yet the learning rate was manually tuned to 1e-4.

#### 4. Appropriate training data

The training data consists of one round going clockwise and one round going counterclockwise.
Furthermore, the training data contains recovery maneuvers.
It also contains extra training to go over the bridge an driving around the curves after the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first try involved one training lap and the simple network, described in the lesson "7. Training the Network":
The car left the road after a few seconds.

Next I tried the data preprocessing. I introduced a lambda layer that normalized the data to the range from -1.0 to 1.0: The steering was smoother and the car stayed a little bit longer on the road. Actually, the car got stuck at the bridge.

Then I tried the LeNet model: The steering was already very smooth and the car followed the road quite a bit. Unfortunately, the car drove only on the left edge of the road until it suddenly got stuck on a curb. Driving to much on the left edge comes probably from the training data set that drives the circuit counter-clockwise, e.g. steering to the left.

To augment the data, I flipped the images in a first step: The car now steered to the right, when it came close to the left edge. The car drove a longer time on the road before it did not took the left turn on the bridge and drove into the lake.

Next, I cropped the image to get rid of those parts of the image that don't contain usefull information. As we are only interested in the part that shows the road, I cropped off the upper and lower part of the image: With the cropped off images the car could drive up to the bridge.

Then I tried the nVIDIA network instead of the LeNet architecture: With the augmented net, the car was able to go over the bridge.

I collected a few more data until the car was able to accomplish driving the whole track by itself.



#### 2. Final Model Architecture

The final model architecture is already described above. Here is the summary of keras:
![alt text][image_CNNStructure]


#### 3. Creation of the Training Set & Training Process


To capture good driving behavior, I first recorded one lap counterclockwise on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_Center]

I then recorded one lap on track one driving clockwise.
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from driving too near to the road edge.

To augment the data i randomly took one image of the three cameras (center, left, right).
The steering wheel was corrected by a factor of 0.2.
For the left images, the correction factor was added, for the right image, the correction factor was subtracted.

To augment the data set further, I also flipped images and angles thinking that this would help the net learn to drive around left and right curves.
For example, here is an image that has then been flipped:
![alt text][image_flipped]


After the collection process, I had 26562 number of data points. I did not process the data any further.

The following histogram shows the distribution of the steering angle:
![alt text][image_histogram]

The data contains a huge amount of driving straight, shown by the peaks at steering angle 0 and -0.2 and 0.2. The peaks at -0.2 and 0.2 come from the left and right camera with the correction factor 0.2
The data also has enough samples of driving around curves. These are samples with steering angle unequal -0.2, 0.0 or 0.2


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer with a manual learning rate of 1e-4.
