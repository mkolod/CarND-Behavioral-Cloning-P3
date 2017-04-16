# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center1]: ./examples/center1.jpg
[flipped1]: ./examples/left1_flipped.jpg
[left1]: ./examples/left1.jpg
[right1]: ./examples/right1.jpg
[cropped1]: ./examples/cropped1.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/writeup_report.md) (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/drive.py) file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```
Note that when running Keras and TensorFlow inside a Docker container, we must expose port 4567 to the host, where the Udacity simulator is running. Also when running with GPU acceleration, we should first install the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin and start the container using `nvidia-docker` rather than `docker`. 

```sh
nvidia-docker run --rm -it -v `pwd`:/src -p 8888:8888 -p 4567:4567 udacity/carnd-term1-starter-kit /bin/bash
```

Once in the container, we can call the `drive.py` script as above. When running on bare metal, we can just call the script directly, as mentioned above.

We then need to start the Udacity simulator in autonomous mode. The simulator will then communicate with the Keras model, which will take image inputs and provide steering angles to the simulator.

#### 3. Submission code is usable and readable

The [model.py](https://github.com/mkolod/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with the following architecture (model.py code lines 118-155). This is a verbatim implementation of the NVIDIA model (see [paper](https://arxiv.org/pdf/1604.07316v1.pdf)), although the input dimensions are a bit different here, so the outputs of the particular layers are different than in the NVIDIA paper. However, the size and number of convolutional filters, the number of layers, the types of layers and the dense/fully connected/affine layer dimensions are the same. The only thing I took care to add was the dropout layer after the first fully connected layer, which contains most (2,385,036 out of 2,638,455, or ~90.4%) of the parameters of the model. See more on that below.

| Layer  | Output Shape  | Number of Parameters   | Value
|---|---|---| --- |
| Lambda  | 160 x 320 x 3   | 0  | lambda x: x/127.5 - 1. |
| Cropping 2D   | 68 x 320 x 3  | 0  | top=67, bottom=25, left=0, right=0 |
| Convolution 2D   | 64 x 316 x 24  | 1,824  | 24 5x5 filters, valid padding |
| ReLU Activation  | 64 x 316 x 24  | 0  |  |
| Max Pooling   | 32 x 158 x 24  | 0 | 2x2 |
| Convolution 2D  | 28 x 154 x 36  | 21,636  | 36 5x5 filters, valid padding |
| ReLU Activation  | 28 x 154 x 36  | 0  |  |
| Max Pooling  | 14 x 77 x 36  | 0  | 2x2 |
| Convolution 2D  | 10 x 73 x 48  | 43,248  | 48 5x5 filters, valid padding |
| ReLU Activation  | 10 x 73 x 48  | 0 |  |
| Max Pooling  | 5 x 36 x 48  | 0  | 2x2 |
| Convolution 2D  | 3 x 34 x 64   | 27,712  | 64 3x3 filters, valid padding |
| ReLU Activation  | 3 x 34 x 64  | 0  |  |
| Convolution 2D  | 1 x 32 x 64  | 36,928  | 64 3x3 filters, valid padding |
| ReLU Activation  | 1 x 32 x 64  | 0 |  |
| Flatten  | 2,048  | 0  |  |
| Dense  | 1,164  | 2,385,036  |  |
| ReLU Activation  | 1,164  | 0  |  |
| Dropout  | 1,164  | 0  | 0.5 |
| Dense  | 100  | 116,500  |  |
| ReLU Activation  | 100  | 0  |  |
| Dense  | 50  | 5,050  |  |
| ReLU Activation  | 50  | 0  |  |
|  Dense | 10  |  510 |  |
| ReLU Activation  | 10  | 0  |  |
| Dense  | 1 |  11 |  |
| Tanh Activation  | 1  | 0  |  |

The data is first normalized to prevent a wide input range, which would make the model harder to train. This is accomplished using a lambda layer (code line 120). I then apply cropping to take care of the "distractions" at the top of the image, such as sky and trees, and to remove a portion of the hood of the car at the bottom. I crop 67 pixels at the top and 25 at the bottom, while not cropping anything to the left and the right (code line 124). 

Next, I apply the convolutional and pooling layers as described in the table above (lines 125-137). Note that the first three convolutional layers have 5x5 filters and are followed by 2x2 max pooling (lines 125-133), while the last two convolutional layers have 3x3 filters and are not followed by pooling layers (lines 134-137). The reason to have two convolutional layers without pooling in between rather than one layer with more filters is to introduce more non-linearities in the model. Also, despite the lack of pooling after the second-to-last and last convolutional layers, the height and width of the filters shrink due to the choice of valid padding rather than same padding. 

Before applying fully-connected layers, we need to flatten the output (line 138). The flattened output has 2,048 units. The first fully-connected layer has 1,164 weights (plus biases), and this results in a huge matrix with 2,385,036 elements (line 139). This represents about 90.4% of the weights of this model. This is a typical problem in models that have both convolutional and fully-connected layers - the first fully-connected layer will likely have the majority of the weights, and so will contribute to overfitting. To address this, I apply 50% dropout (line 141). The subsequent fully-connected layers result in much fewer weights, and I didn't find that regularizing them helped with generalization (lines 142-149).

Note that all the activations except the last one were ReLU, but the last one was tanh (line 149). The reason for tanh at the end is that we need to predict both positive and negative angles, and they have to range between -1 and 1, so tanh is a natural choice here. The last layer in the network projects the output down to one element, which will predict the steering angle (lines 148-149). It's a real-valued prediction, since this will be a regression network.


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 141). As mentioned before, I experimentally determined that the only layer that needed dropout was the first fully-connected layer, which contained most (~90.4%) of the parameters.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 37, 107-108 and 161-163). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This is the reason why I only split the data into training and validation sets, and used visual inspection of driving as a test. While that isn't a proper numerical metric the way MSE was for validation data, the ultimate objective is for the vehicle to stay on track, and it's not something the model optimizes for directly while training (it indirectly optimizes by reducing MSE). It's kind of like with machine translation, where training optimizes for a reduction in perplexity, but the human-style evaluation of translation is the BLEU score. If we had some data as to how far from the center of the lane the car was at all time rather than the difference between the actual and predicted steering angle, we could have estimated the true objective of staying in the center. We would have had to accommodate for "hard losses," such as introducing an infinite loss if the car goes off the track boundary. Since no such data was available, I decided to first evaluate the hyperparameter tuning by comparing training and validation loss during training, and then test the car by visual inspection on the track.

In addition to dropout, I used data from all three cameras to make sure the dataset was balanced. Since the track has mostly straight road or very gentle curves, that would have resulted in the car learning to drive straight and take delicate turns, but it would have failed on the several sharp turns that exist. During straight driving, the steering angle should be zero (keyboard input) or close to zero (mouse input with the left key constantly depressed). This would result in a very unbalanced dataset, so per the [NVIDIA paper](https://arxiv.org/pdf/1604.07316.pdf), using the side camera images along with an imputed steering angle correction can produce better results. Since these aren't actual steering angles, it's possible that this doesn't let the car learn how to learn both gentle and sharp curves, but it's certainly better than using just the center camera. I also flipped the images horizontally (left-to-right) to ensure that the car wouldn't learn a left bias, since most turns on the track were to the left. This also generated more data in general to account for other problems, such as producing more examples of the driving on the bridge, where the road texture is different, or the fragments of the road that have sand or dirt instead of a concrete boundary.

I also briefly tried random cropping, e.g. by translating the image left or right by a certain number of pixels (say uniformly from -20 to 20 pixels, with negative being a left translation and a positive value being a right translation), and then compensating with a steering angle adjustment, just as in the case of the side cameras. Unfortunately, I did not find such translations to be very useful for a wide range of angle adjustments. I did make sure that after translation, I would take a central crop with the sides equal to the maximum translation in each direction being removed, but this mode of augmentation wasn't very helpful. It's a popular augmentation strategy though, and it often works, so it just wasn't helping with my particular data sample and model, and it's not that it's not useful in many cases. I could have also tried hue/saturation/brightness/contrast adjustments, but I ran out of time to experiment more.

.0115 - val_loss: 0.0110

The model certainly wasn't overfitting at the end, since after the last epoch, the training set loss was 0.0115, while the validation set loss was 0.0110. Considering the fact that the loss was close to 1.0 on the first iteration, that's a tiny difference between training and validation loss by end of training. It's hard to know if it wasn't underfitting since there is no guideline as to what the "typical" MSE should be for this model, but we can clearly tell from the behavior on the track that the car did quite well.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 154). Note that Adam is actually a tunable optimizer, for example in TensorFlow, we can set the learning rate (default=0.001), as well as the betat (default=0.9), beta2 (default=0.999) and epsilon (default=1e-08) hyperparameters. Still, in a lot of ways, Adam can optimize very well even without tuning the hyperparameters. A lot of big networks are actually trained using manually adjusted SGD with momentum (learning rate decay policy, weight decay policy, momentum), but it's good to start with an "automatic" optimizer first, particularly when assessing if the model architecture and data are good at all. Another good choice is Adagrad, which is very tolerant to a poor selection of the initial learning rate, since the history of gradients for each weight effectively auto-tunes the per-weight learning rate. Adagrad does have the drawback that the learning rate eventually tends to zero, but here we didn't have too many epochs, so it could have been a good choice. I tried Adagrad and found that Adam actually worked better, though there are other problems where the reverse is often true (e.g. in natural language processing). 


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, as well as recovering from the left and right sides of the road. I used both the Udacity-provided dataset, as well as my own data. 

In addition to the previously mentioned image flipping, which is a synthetic solution to the "left bias" problem, and a way to double the dataset size, I also drove several tracks in the reverse direction, because that also creates a different set of images and perspectives on various objects such as trees, etc. All in all, I amassed a dataset consisting of 73,120 images from all three cameras, and with left/right flipping, that brought the total to 146,240. This isn't a very small dataset, but it's not huge, either. For some perspective, MNIST has 60,000 images, while the ILSVRC 2012 version of ImageNet has about 1.2 million images in the training set. With color augmentations and random translations and center cropping, one could imagine the dataset growing dramatically, but of course, the invariance to color or brightness is not the same thing as say collecting more examples of off-center recovery.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a few convolutional and fully-connected layers, with typical filter sizes, such as 3x3 or 5x5. It was clear to me that the dataset wasn't big enough to train an Inception v3-style model training from randomly initialized weights, so it had to be something smaller. Also, VGG-style models are very hard to train, while AlexNet had way too many weights not to overfit, even with lots of dropout. Therefore, after experimenting with my own designs a bit, I finally decided to try the [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf), and it worked the best.

Despite the NVIDIA model's promise, I needed to add the dropout after the first fully-connected layer, in order to prevent overfitting (as mentioned before, that layer contained ~90.4% of all the weights in this model).

As mentioned earlier, I used data from all three cameras, along with the imputed steering angle correction for left and right cameras, flipped the images left-to-right, and drove the other way in order to create a realistic set of scenarios and to prevent overfitting and memorization of straight driving.

#### 2. Final Model Architecture

As mentioned before, my final model architecture was identical to the [NVIDIA model](https://arxiv.org/pdf/1604.07316v1.pdf), with the excption of the first fully-connected layer, after which I added 50% dropout.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Driving][center1]

I then added the images from the left and right cameras, along with an imputed steering angle correction added to or subtracted from the angle associated with the center camera image for the particular record. 

Here is the left camera image associated with the above center image:

![Left camera][left1]

And here is the right camera image associated with the above center image:

![Right camera][right1]

To prevent left bias (most curves are to the left on the track), and to generate more examples, I flipped all the images horizontally. Here is the flipped version of the image from the left camera above:

![Flipped image from left camera][flipped1]

Finally, I mentioned cropping earlier. It was used to remove the distractions at the top of the image, such as the sky and trees, as well as from the bottom, in order to remove the hood of the car. The center image shown earlier, when cropped 67 pixels from the top and 25 pixels from the bottom, would look like this:

![Cropped center camera image][cropped1]
