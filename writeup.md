# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./assets/barplot.png "barplot of ratio of each class"
[image2]: ./assets/original-images.png "original images"
[image3]: ./assets/grayscaling-images.png "grayscaling images"
[image4]: ./assets/test-images.png "test images"
[image5]: ./assets/conv1.png "conv1"
[image6]: ./assets/relu1.png "relu1"
[image7]: ./assets/maxpool1.png "maxpool1"
[image8]: ./assets/conv2.png "conv2"
[image9]: ./assets/relu2.png "relu2"
[image10]: ./assets/maxpool2.png "maxpool2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You are reading it! And here is a link to my [project code](https://github.com/GzuPark/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **(32, 32, 3)**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Exploratory visualizations of the dataset

Here is an exploratory visualization of the data set. First, it is a bar plot showing ratio of each class by data set, and it looks distributed properly by each label:

![Barplot of ratio of each class][image1]

Second, I draw examples of whole labels using by randomized picked each class:

![Original images][image2]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data

As a first step, I decided to convert the images to grayscale because this work help to process faster rather than RGB. What I mean is that RGB has 3 channels (Red, Green, Blue), but grayscaling has 1 channel. So, when a computer calculates matrix, RGB have to calculate vector and grayscale should calculate scalar. [link](https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale)

Here is an example of a traffic sign image after grayscaling.

![Grayscaling images][image3]

#### 2. Model architecture

I decide to use LeNet model basically, which I studied in the class. However, I want to test model whose architecture has best performance. Thus, I check **standard deviation** of `tf.truncated_normal()`, **shape of filter**, and **depth** (plus, **shape** of fully connected). I don't touch others, such as number of layers, what kind of activate functions, and etc. I select parameter which has the highest result step by step.

Starting default set up: `epochs = 10`, `batch_size = 128`, `learning_rate = 0.001`

##### 1. `stddev = [0.01, 0.03, 0.05, 0.1]`

Set up: `ksize = 5`, `depth = [8, 32]`

| stddev  | Test Accuracy   |
|:-------:|:---------------:|
| 0.01    | 0.923           |
| 0.03    | 0.927           |
| 0.05    | 0.908           |
| 0.1     | 0.854           |

I pick `stddev = 0.03`.

##### 2. `ksize = [3, 5]`

Set up: `stddev = 0.03`, `depth = [8, 32]`

| ksize   | Test Accuracy   |
|:-------:|:---------------:|
| 3       | 0.906           |
| 5       | 0.927           |

Shapes of fully connected depend on a result of convolutional network. I pick `ksize = 5`.

##### 3. `depth = [[6, 25], [8, 32]]`

Set up: `stddev = 0.03`, `ksize = 5`

| depth     | Test Accuracy   |
|:---------:|:---------------:|
| [6, 25]   | 0.924           |
| [8, 32]   | 0.927           |

Shapes of fully connected depend on a result of convolutional network. I pick `depth = [8, 32]`.

My final model consisted of the following layers:

| Layer         		|     Description	                     					|
|:-----------------:|:---------------------------------------------:|
| Parameters        | mu=0, stddev=0.03            			     				|
| Input         		| 32x32x1 grayscaling image   			     				|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x8   	|
| RELU    					|		                        										|
| Max pooling	      | 2x2 stride, outputs 14x14x8            				|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 10x10x32  	|
| RELU    					|		                        										|
| Max pooling	      | 2x2 stride, outputs 5x5x32            				|
| Flatten    				|	outputs 5x5x32 = 800      										|
| Fully connected		| inputs 800, outputs 512      									|
| Fully connected		| inputs 512, outputs 256      									|
| Fully connected		| inputs 256, outputs 43      									|

#### 3. How trained my model

##### 1. `epochs = [5, 10, 15]`

Set up: `batch_size = 128`, `learning_rate = 0.001`

| epochs  | Test Accuracy   |
|:-------:|:---------------:|
| 5       | 0.912           |
| 10      | 0.927           |
| 15      | 0.918           |

##### 2. `batch_size = [64, 128, 256]`

Set up: `epochs = 10`, `learning_rate = 0.001`

| batch_size  | Test Accuracy   |
|:-----------:|:---------------:|
| 64          | 0.923           |
| 128         | 0.927           |
| 256         | 0.917           |

##### 3. `learning_rate = [0.0005, 0.001, 0.003]`

Set up: `epochs = 10`, `batch_size = 128`

| learning_rate | Test Accuracy   |
|:-------------:|:---------------:|
| 0.0005        | 0.903           |
| 0.001         | 0.927           |
| 0.003         | 0.869           |

I choose **AdamOptimizer** which is one of [the efficient optimizer](http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms). I know that I should do **16 x 27 = 432** ways (4x2x2x3x3x3) if I want to check manually. Also, if training models are repeated, maybe test accuracy would be changed. But I skipped all.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

Grayscaled images are helpful [to reduce running time](https://www.quora.com/In-image-processing-applications-why-do-we-convert-from-RGB-to-Grayscale), and I designed
the architecture of LeNet model with 2 convolutional network layers and 3 fully connected layers. As you can see above, I select parameters step by step.

My final model results were:

* maximum validation set accuracy of **0.937**
* test set accuracy of **0.908**

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    * My first model is LeNet model with 2 CNN layers and 3 FC layers, and I just followed the model I learned.

* What were some problems with the initial architecture?
    * I have to change **depths** of CNN that is find good way.

* How was the architecture adjusted and why was it adjusted?
    * The LeNet model with MNIST had 10 classes, but this project didn't have equal classes, it had 43!

* Which parameters were tuned? How were they adjusted and why?
    * I tuned a standard deviation value on calculating convolution layers.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Fully connected shapes are important, because they were decided by convolutional layers and shrinking gap between each step effect to accuracy. I passed out dropout option due not to observe overfitting.

If a well known architecture was chosen:

* What architecture was chosen?
    * LeNet

* Why did you believe it would be relevant to the traffic sign application?
    * The size of the traffic sign images are 32x32, it is same with MNIST.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * It shows that test set accuracy has over 90%.


### Test a Model on New Images

#### 1. Choose five traffic signs found on the web and provide them in the report.

Here are five traffic signs that I found on the web:

![Test images][image4]

* **1** : 16, Vehicles over 3.5 metric tons prohibited
* **2** : 35, Ahead only
* **3** : 42, End of no passing by vehicles over 3.5 metric tons
* **4** : 14, Stop
* **5** : 14, Stop

The first, second, and third images included in German traffic signs test data set, so they are quite to easy to predict. [Fourth image](https://cdn-images-1.medium.com/max/1600/1*4GcPQlKRT1mGTwaI18axxQ.png) is an adversarial traffic signs and fifth image is a source from South Korea. However, they are hard to predict, because of some noise mark on the images: fourth image is hided parts of sign with black and white tape, and fifth image include Korean letters.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			                                          |     Prediction	        		                   			|
|:---------------------------------------------------:|:---------------------------------------------------:|
| Vehicles over 3.5 metric tons prohibited      		  | Vehicles over 3.5 metric tons prohibited   		      |
| Ahead only     		                                 	| Ahead only 						                      				|
| End of no passing by vehicles over 3.5 metric tons  | End of no passing by vehicles over 3.5 metric tons  |
| Stop	                                          		| Speed limit (20km/h)		               			 				|
| Stop			                                          | No passing      					                      		|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of **60%**. This compares favorably to the accuracy on the test set of **90.8%**. They are huge gap because of the fact that I selected two adversarial images. I want to check out whether a hard condition image can cognize or not. For this work, I decided a clear label, which is **14 STOP** in this case, and searched on the web what I want. If I train with noise images, I would predict fourth image correctly. But, fifth image has a different problem. I might apply a part of image, **STOP**, or should train Korean traffic signs too.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 14th cell of the [Ipython notebook](https://github.com/GzuPark/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

For the first image, the model is relatively sure that this is a **Vehicles over 3.5 metric tons prohibited** sign (probability of 1.0). The top five soft max probabilities were:

| Probability   |     Prediction	        					                 |
|:-------------:|:--------------------------------------------------:|
| 1.0      			| Vehicles over 3.5 metric tons prohibited					 |
| 0.0       		| No passing 										                     |
| 0.0					  | Speed limit (60km/h)											         |
| 0.0	      		| Speed limit (80km/h)					 				             |
| 0.0				    | No passing for vehicles over 3.5 metric tons       |

For the second image, the model is relatively sure that this is a **Ahead only** sign (probability of 0.993). The top five soft max probabilities were:

| Probability   |     Prediction	        					                 |
|:-------------:|:--------------------------------------------------:|
| 0.9930   			| Ahead only					                               |
| 0.0040     		| No passing 										                     |
| 0.0022			  | Speed limit (60km/h)											         |
| 0.0007    		| No vehicles					 				                       |
| 0.0001		    | No passing for vehicles over 3.5 metric tons       |

For the third image, the model is relatively sure that this is a **End of no passing by vehicles over 3.5 metric tons** sign (probability of 1.0). The top five soft max probabilities were:

| Probability   |     Prediction	        					                 |
|:-------------:|:--------------------------------------------------:|
| 1.0      			| End of no passing by vehicles over 3.5 metric tons |
| 0.0        		| End of no passing            	                     |
| 0.0   			  | No passing for vehicles over 3.5 metric tons		   |
| 0.0       		| Right-of-way at the next intersection              |
| 0.0   		    | Speed limit (100km/h)                              |

For the fourth image, the model is relatively sure that this is a **Stop** sign (probability of 0.0), but the image does contain a stop sign. The top five soft max probabilities were:

| Probability   |     Prediction	        					                 |
|:-------------:|:--------------------------------------------------:|
| 0.8232   			| Speed limit (20km/h)                               |
| 0.1753     		| Speed limit (30km/h)                               |
| 0.0014			  | Go straight of left	                               |
| 0.0001     		| General caution                                    |
| 0.0000   	    | No entry                                           |

For the fifth image, the model is relatively sure that this is a **Stop** sign (probability of 0.0), but the image does contain a stop sign. The top five soft max probabilities were:

| Probability   |     Prediction	        					                 |
|:-------------:|:--------------------------------------------------:|
| 0.9997   			| No passing                                         |
| 0.0003     		| Priority road                                      |
| 0.0000			  | Right-of-way at the next intersection              |
| 0.0000     		| No entry                                           |
| 0.0000   	    | Roundabout mandatory                               |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

First image shows a shape of sign and an object on the middle.

![conv1][image5]

Second image has 4 feature maps we can cognize. FeatureMap0 and FeatureMap6 can understand a shape of sign, FeatureMap3 and FeatureMap4 can recognize an object on the middle.

![relu1][image6]

Third image has 4 feature maps with a shape of sign, but it is getting hard what object is on the middle.

![maxpool1][image7]

From fourth image to sixth image, they are hard to understand what it is with human eyes. However, we can know that each FeatureMap has unique feature.

![conv2][image8]

![relu2][image9]

![maxpool2][image10]
