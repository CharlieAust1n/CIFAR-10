# CIFAR-10

## What is CIFAR-10
CIFAR-10 is a dataset consiting of 60,000 32x32 colour images that belong to one of 10 different categories, where each category consists of 6,000 images.

The 60,000 images are split up into 50,000 training images and 10,000 testing images.

The 10 categories which make up the 60,000 images are:
* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Model Used
The model I used to train on this dataset was a **Convolutional Neural Network** (will be refered to as a CNN).
A CNN 

## TensorBoard
### What is TensorBoard
TensorBoard is a tool created by TensorFlow, that provides the measurements and visualizations needed during the machine learning workflow.
This allows for the tracking of data such as accuracy & loss, visulation of the model graph, etc.

### TensorBoard and My Model
For my model, TensorBoard was used to track the **in-sample and out-of-sample accurarcy and loss** over a stretch of an arbitrary number of epochs, ***n***.
By tracking this data, I was able:
1. Reduce undersirable behavior such as underfitting and overfitting.
2. Effectively determine how many convolutional layers, linear layers, and input nodes should be included inside my model.
3. Identify any other flaws inside my model.

**TODO:** Insert TensorBoard images here

# Results
**TODO:** Insert model accuracy and loss here!


