# Introduction to Image Classification

**Completed by Mangaliso Makhoba.**

**Overview:** Use ensamble methods, Random Forest Classifier, to classify images with hand drawn numbers, to the very numbers.

**Problem Statement:** Build a model that will classify an image into what number is drawn on the image. 

**Data:** [The MNIST Database](http://yann.lecun.com/exdb/mnist/)

**Deliverables:** A predictive Model

## Topics Covered

1. Machine Learning
3. Ensamble Methods
4. Raandom Forest Classification
5. Image Classification
6. Classification_report

## Tools Used
1. Python
1. Pandas
2. Scikit-learn
2. Jupyter Notebook

## Installation and Usage

Ensure that the following packages have been installed and imported.

```bash
pip install numpy
pip install pandas
pip install sklearn
```

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow instruction on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 

Alternatively:
VS Code can render Jupyter Notebooks

## Notebook Structure
The structure of this notebook is as follows:

 - First, we'll extract and load our data to get a view of the predictor and response variables we will be modeling. 
 - Secondly, Train the model. 
 - We then evaluate the model's accuracy.
 - Following this modeling, we evaluate the models performance in each label by printing the classification report.
 - Make a prediction and print the image to test the model



# Function 1: Extracting the data
Create a function get_data that uses the above functions to extract a certain number of images and their labels from the gzip files.

The function will take as input two integer values and return four variables in the form of (X_train, y_train), (X_test, y_test), where (X_train, y_train) are the extracted images / labels of the training set, and (X-test, y_test) are the extracted images / labels of the testing set.

Image pixel values range from 0-255. Normalise the image pixels so that they are in the range 0-1.

Function Specifications:

Should take two integers as input, one representing the number of training images, and the other the number of testing images.
Should return two tuples of the form (X_train, y_train), (X_test, y_test).
Note that the size of the MNIST images are 28x28

Usually when setting up your dataset, it is a good idea to randomly shuffle your data in case your data are ordered. Think of this as shuffling a pack of cards. Here, however, we aren't going to shuffle the data so that all our answers are the same.

** Expected Output **
```python
(X_train, y_train), (X_test, y_test) = get_data(5000,1000)
## Print off the shape of these arrays to see what we are dealing with
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)

(5000,)
(1000,)
(5000, 784)
(1000, 784)
```



# Function 2: Model Training

Now that we have formatted our data, we can fit a model using sklearn's `RandomForestClassifier` class with 20 estimators and its random_state is set to 42. We'll write a function that will take as input the image and label variables that we created previously, and return a trained model.

_**Function Specifications:**_
* Should take two numpy `arrays` as input in the form `(X_train, y_train)`.
* Should return an sklearn `RandomForestClassifier` model which has a random state of 42 and number of estimators 20.
* The returned model should be fitted to the data.



# Function 3: Testing the model Accuracy

Now that you have trained your model, lets see how well it does on the test set. Write a function which returns the accuracy of your trained model when tested with the test set.

_**Function Specifications:**_
* Should take the fitted model and two numpy `arrays` `X_test, y_test` as input.
* Should return a `float` of the accuracy of the model. This number should be between zero and one.


# Function 4: Printing the Classification Report

Classification reports gives us more information on where our model is going wrong - looking specifically at the performance caused by Type I & II errors. Write a function which returns the classification report of your test set.

_**Function Specifications:**_
* Should take the fitted model and two numpy `arrays` `X_test, y_test` as input.
* Should return a classification report

_**Hint**_ You don't need to do this manually, sklearn has a classification report function.

**Note:** Please find more details and specifics within the notebook

## Conclusion
Random Forest does a very excellent job in predicting the label, with an accuracy of 89.1% without it's hyperparameters tuned. The continuation of this project would realuate the performance of the model with the best hyperparameters.  

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## Contributing Authors
**Authors:** Mangaliso Makhoba, Explore Data Science Academy

**Contact:** makhoba808@gmail.com

## Project Continuity
This is project is on going


## License
[MIT](https://choosealicense.com/licenses/mit/)