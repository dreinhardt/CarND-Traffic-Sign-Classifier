# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README




After Review, all changes are marked with a "-->"

For my defense, I had a coding bug within my model. That was the reason the results within validation accuracy were so worse.
in my normalization function pre(.), I subtracted with 128 instead of 128.0...
This failor took me hours!!




#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You can find my work within Traffic_Sign_Classifier.ipynb, as well, within the HTML document Traffic_Sign_Classifier.html
Within my code I do documention, as well. This should clarify my algorithm step by step.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:


--> is corrected now!

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
I display random images of my available traffic signs and I mark them each with its class.

![image1][./results/DataSet.png "Random Data Sets with its class"]



--> Additionally, after the review I added the wished bar picture.



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not converted my pictrues into greyscale. Because within the tutorial videos it was mentioned to work with RGB!
--> Therefore, I stay with the given 3 color channels.

I used several code out of the tutorial, but I suppose this is not forbidden.

I chose as parameters 100 Epochs with a batch size of 128.
My learning rate is still 0.001
I placed my variable (x,y,keep_probability a.s.o.)

Here, I placed a function pre(.) to normalize the pictures as described in the text.
--> I do that because of speed up the training and performance. It is necessery to have zero mean and equal variance. In case this is not given, it is difficult for the optimizer to find a proper solution. The image has a typical pixel value of 0 to 255 pixel value.

Afterwards I call that function with all my images sets.

I output no further pictures here.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I chose the identical model I used in the LeNet tutorial. It is managable for me and I adjusted the parameters, mentioned in the project instructions before (number of classes from 10->43 and 1->3 depth for the RBG images).

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		|   32x32x3 RGB image   						| 
| Convolution 3x3     	|   1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	|   2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    |   1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	|   2x2 stride,  outputs 5x5x16 				|
| Flatten				|	Input = 5x5x16. Output = 400.				|
| Fully connected		|   Input = 400. Output = 120.					|
| RELU					|												|
| Dropout				|	with 50%									|
| Fully connected		|   Input = 120. Output = 84.					|
| RELU					|												|
| Dropout				|	with 50%									|
| Fully connected		|   Input = 84. Output = 43.	image depth 3	|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I chose as parameters 100 Epochs, because I needed more to get a better result (needed more rounds for computation!).
With a batch size of 128 (not more, because than my result was much worse).
My learning rate is still 0.001, identical to the tutorial. It was recommended there!
I chose the AdamOptimizer, identical to the tutorial. It was recommended there!
Furthermore, I inlcuded the One_Hot method.

At the beginning, the validation accuracy was low. But fast, it was in a steady state around 0.818.
After that period, I saved my work for further processing.




#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

INFO:tensorflow:Restoring parameters from ./lenet
Train Accuracy = 0.988
Validation Accuracy = 0.831
Test Accuracy = 0.806

I chose the architecture out of the tutorial. I fitted best for me to get further experience by designing a neuronal network.
The final train accuracy seems to be quite good. Unfortunately, the test and validation accuracy was not similar.
Surely, there are other model architectures, which propably would fit better, but I chose exactly this architecture.
I did experiments with the pooling method and placed two pooling stages. Especially by using 2 stages, I got better results.


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

17,No entry
![alt text][../data/Germ_TS/1.png] 

12,Priority road
![alt text][../data/Germ_TS/2.png] 

9,No passing
![alt text][../data/Germ_TS/3.png] 

13,Yield
![alt text][../data/Germ_TS/4.png] 

14,Stop
![alt text][../data/Germ_TS/5.png]

I tried to keep them as sharp as possible.
I cutted them from random pictures I found on google.


--> Why I chose that pictures?
The first objective was to have clear pictures, which are really sharp.
I tried to have pictures with a good ancle of the traffic sign. It is easier without any additional corrections to have a "flat" perspective on the traffic sign.
Furthermore, I tried to have as less background noise as possible (trees, avoid many different colors).
An important requirement was, that only "one" traffic sign per image was clearly visible. Otherwise, the algorithm must be improved to focus on a specific sign.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

---> I overworked the whole section!


Here are the results of the prediction:
INFO:tensorflow:Restoring parameters from ./lenet
Results:      [17 12 40 13 14]
Dominiks GTS: [17 12  9 13 14]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Priority Road   		| Priority Road									|
| No passing			| Roundabout mandatory							|   <-----
| Yield	      			| Yield							 				|
| Stop					| Stop 											| 


Here I list again all my samles:
![alt text][./results/gts.png]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of the shown results in section 1.4

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

---> I overworked the whole section!

The code for making predictions on my final model is located below of the headline "Predict the Sign Type for Each Image" of the Ipython notebook.

Again my results: 
INFO:tensorflow:Restoring parameters from ./lenet
Results:      [17 12 40 13 14]
Dominiks GTS: [17 12  9 13 14]

For the values my model is relativly sure, I skip to note all following traffic sign possibilities!


For the first image, the model is relatively sure that this is a no entry sign. The top five soft max probabilities were:
[17, 14, 29,  0, 26]
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry   									| 
| .0     				|  												|
| .0					| 												|
| .0	      			| 								 				|
| .0				    |       										|


For the second image, the model is relatively sure that this is a priority road sign. The top five soft max probabilities were: 
[12, 42, 26, 11, 13]
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road  								| 
| .00     				|  												|
| .00					| 												|
| .00	      			| 					 							|
| .00				    |       										|



For the second image, the model is relatively sure that this is a roundabout mandatory sign. Unfortunately, all other propabilities are really low. So the model is sure it is that specific (wrong) sign. But for its defense, the signs are really very similar!
The top five soft max probabilities were:
[40, 38, 37, 39, 33]
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| roundabout mandatory   						| 
| 0.0     				|  												|
| 0.0					| 												|
| 0.0	      			| 								 				|
| 0.0				    |       										|



For the fourth image, the model is relatively sure that this is a yield sign. The top five soft max probabilities were:
[13, 12, 38, 25, 10]
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Yield   										| 
| .0     				|  												|
| .0					| 												|
| .0	      			| 					 							|
| .0				    |       										|



For the fifth image, the model is relatively sure that this is a Stop sign. The top five soft max probabilities were:
[14,  5, 13,  1, 29]
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Stop   										| 
| .0     				|  												|
| .0					| 												|
| .0	      			| 					 							|
| .0				    |       										|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


