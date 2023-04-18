# Malaria-prediction-transfer-learning
Detection in Thin Blood Smear Images using transfer learning and CNN.

MANUAL WORKFLOW

1. CNN MODEL- BASE MODEL

➔ Import required libraries

➔ Initializing the CNN

➔ Convolution:Thisisthefirststepintheprocessofextractingvaluablefeaturesfrom an image. A convolution layer has several filters that perform the convolution operation. Every image is considered as a matrix of pixel values.

➔ Pooling : Pooling is a down-sampling operation that reduces the dimensionality of the feature map. The rectified feature map now goes through a pooling layer to generate a pooled feature map.

➔ Adding a second convolution layer

➔ Flattening : After finishing thep revious two steps, we're supposed to have a pooled feature map by now. As the name of this step implies, we are literally going to flatten our pooled feature map into a column.

➔ Full Connection : The Full Connection Step in Convolutional Neural Networks, As its name implies, a fully connected layer's neurons are connected to all of the neurons in the next layer. The purpose of the fully connected layer in a convolutional neural network is to detect certain features in an image.

➔ Compiling the CNN

➔ Early Stopping : Early stopping is a method that allows you to specify an arbitrarily large number of training epochs and stop training once the model performance stops improving on the validation dataset.

➔ Evaluating the Model

2. VGG19 MODEL

➔ Resize image, convert RGB images to array-
Resizing an image involves changing the dimensions of an image, either by making it larger or smaller. This process can be useful for a variety of reasons, such as preparing images for web pages or social media, or creating images that meet specific size requirements.
   
When resizing an image, the aspect ratio (the relationship between the width and height of the image) should be maintained to avoid distorting the image. This can be achieved by either resizing the width or height while keeping the other dimension proportional, or by cropping the image to a new aspect ratio.
Converting an RGB image to an array involves transforming an image from its visual representation as a collection of pixels with red, green, and blue values to a numerical representation in the form of an array. This process is often necessary for computer vision and machine learning applications that require numerical data to perform calculations and analysis on images.
In an RGB image, each pixel is represented by three values: the red, green, and blue color values that range from 0 to 255. To convert an RGB image to an array, each pixel's RGB values are flattened into a single array value, resulting in a 2D or 3D array depending on the image's dimensions. This array can then be used for further processing or analysis using various programming languages and libraries such as Python and NumPy .

➔ Adding the data set into the model

➔ To append features and labels, convert the label to categorical, and reshape the image with 3 channels.
First, let me explain what each of these terms means in the context of machine learning:
Features: Features are the input variables or attributes that are used to train a machine learning model. These could be things like pixel values in an image, measurements of physical characteristics, or demographic data.
Labels: Labels are the output values that the model is trying to predict. These could be things like a classification label (e.g. "dog" or "cat") or a continuous value (e.g. the price of a house).
Categorical Label: Categorical Labels are the labels that represent discrete classes, such as the different breeds of dogs or types of flowers. These labels are typically represented as integers, with each integer corresponding to a particular class.
Reshaping the image with 3 channels: Images are typically represented as a matrix of pixel values, with each pixel representing a single color channel (e.g. red, green, or blue). To use an image as input to a machine learning model, we need to reshape the matrix so that it has 3 channels, one for each color channel.
To append features and labels, we first need to create two separate arrays or lists, one for the features and one for the labels. For example, if we were working with image data, the features might be an array of pixel values, and the labels might be an array of integers representing the class of each image.

➔ Splitting the dataset into train and test dataset

➔ Data augmentation
Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks.
➔ Training the VGG19 Model:
The next step is to train the VGG19 model on the training set of labeled images. This involves passing the images through the model and adjusting the weights of the model's layers to minimize the loss function. The loss function measures the difference between the predicted labels and the true labels. The training process typically involves several epochs (iterations) of passing the images through the model.

➔ Fine-tuning the Model:
After training the VGG19 model on the training set, the model can be fine-tuned on the validation set to improve its performance. This involves adjusting the hyperparameters of the model, such as the learning rate and regularization strength, and retraining the model on the combined training and validation sets.

➔ Testing the Model:

Once the model has been trained and fine-tuned, it can be tested on a separate test set of labeled images to evaluate its performance. The test set should be large enough to provide a reliable estimate of the model's accuracy.
➔ Using the Model for Malaria Prediction:
Finally, the trained VGG19 model can be used to classify new images of blood cells as either infected with malaria or uninfected. This involves passing the images through the model and interpreting the output probabilities as the likelihood of each class. The model can then be integrated into a larger malaria prediction algorithm, which may involve additional preprocessing steps, feature extraction, or post-processing of the predictions.
➔ Classification Report
  
3. Fast API
FastAPI is one of the fastest Python web frameworks. ➔ Install FastAPI and uvicorn
FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It is designed to be easy to use and to provide high-performance, asynchronous APIs. Uvicorn is a lightning-fast ASGI server that powers FastAPI by default, but it can also be used with other ASGI frameworks.

➔ Create a FastAPI app
Once you have FastAPI and uvicorn installed, you can create a new FastAPI app ➔ Define a route
In FastAPI, we define routes using decorators. A decorator is a special type of Python function that modifies the behavior of another function. In FastAPI, decorators are used to define routes and to specify the HTTP method that should be used for that route.
For example, we can define a simple route that responds to GET requests to the root URL ("/") and returns a JSON object.

➔ Run the app with uvicorn
Now that we defined a route, we can start the FastAPI app using the Uvicorn server. ➔ Create an HTML page
Next, we created an HTML page that can make requests to your FastAPI app.

➔ Create a route to handle the form submission
Finally, created a route in our FastAPI app to handle the form submission. This route expects to
receive the data submitted by the form, and it performs some operation based on that data.

 
   
