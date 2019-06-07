# Traffic-Signs-Classification

## Dataset
The dataset that was used was the pickled German Traffic Signs dataset and can be found at https://www.kaggle.com/imadmoussa/traffic-signs. It contains 43 different categories of traffic signs which correspond to the different classes of the
classification problem. The dataset contains 3 different files e.g. train.p, valid.p and test.p that were
used to train and validate the model and finally test whether the final classification model is capable
of predicting correctly the images of the test file. 

The training set contains 34.799 images, the validation
set 4.410 and the test set 12.630. Each image is in the form of a 32x32x3 matrix where 32x32
corresponds to the width and height of the image and the 3rd dimension to the 3 color bands. After
some exploration and visualization, it was observed that the classes did not contain the same number
of training examples and thus this would introduce bias in our models resulting to bad predictions.
Figure 1 shows the histogram of training examples per class, that visualizes the class imbalance in the
training data.
