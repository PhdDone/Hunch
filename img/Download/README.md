##Image classification

Based on [tensorflow tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html#convolution-and-pooling), apply convnet to classify Jean and Shirt images.

##NN structure
<<<<<<< HEAD
Two onvolutional layers followed by one densely connected layer.

##Data
*pos -> Shirts & Blouses

*neg -> Jeans

*training data -> 60 pos + 60 neg

*dev data -> 30 pos + 30 neg

*test data -> 30 pos + 30 neg

##Usage
python convolutional_network.py

##Results
    | train | dev | test 
--- | --- | --- | ---
Accuracy | 0.96667 | 0.86667 | 0.8
