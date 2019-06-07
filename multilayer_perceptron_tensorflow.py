#Import the necassary libraries and files
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from load_and_explore_data import X_train, y_train, X_valid, y_valid, X_test, y_test, signnames

#Reshape the data for the multi layer perceptron
X_train = X_train.reshape(X_train.shape[0],32*32*3).astype('float32')
X_valid = X_valid.reshape(X_valid.shape[0],32*32*3).astype('float32')
X_test = X_test.reshape(X_test.shape[0],32*32*3).astype('float32')

#Normalize the data
X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

#Multi Layer Perceptron with 3 hidden layers of 256 neurons each
class Multi_Layer_Perceptron:
    
    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        #Hyperparameters
        self.mu = mu
        self.sigma = sigma
        
        #Fully Connected Layer 1. Input = 3072. Output = 256.
        self.connected1_input = 3072
        self.connected1_output = 256
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(self.connected1_input, self.connected1_output), mean = self.mu, stddev = self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(self.connected1_output))
        self.fully_connected1 = tf.add((tf.matmul(x, self.connected1_weights)), self.connected1_bias)
        
        #Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)
    
        #Fully Connected Layer 2: Input = 256. Output = 256.
        self.connected2_input = 256
        self.connected2_output = 256
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(self.connected2_input, self.connected2_output), mean = self.mu, stddev = self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(self.connected2_output))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)
        
        #Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)
        #Dropout
        self.fully_connected2 = tf.nn.dropout(self.fully_connected2, keep_prob)
    
        #Fully Connected Layer 3: Input = 256. Output = 256.
        self.connected3_input = 256
        self.connected3_output = 256
        self.connected3_weights = tf.Variable(tf.truncated_normal(shape=(self.connected3_input, self.connected3_output), mean = self.mu, stddev = self.sigma))
        self.connected3_bias = tf.Variable(tf.zeros(self.connected3_output))
        self.fully_connected3 = tf.add((tf.matmul(self.fully_connected2, self.connected3_weights)), self.connected3_bias)
        
        #Activation.
        self.fully_connected3 = tf.nn.relu(self.fully_connected3)
        #Dropout
        self.fully_connected3 = tf.nn.dropout(self.fully_connected3, keep_prob)
        
        #Fully Connected Layer 4: Input = 256. Output = 43.
        self.connected4_input = 256
        self.connected4_output = 43
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(self.connected4_input, self.connected4_output), mean = self.mu, stddev = self.sigma))
        self.output_bias = tf.Variable(tf.zeros(self.connected4_output))
        self.logits = tf.add((tf.matmul(self.fully_connected3, self.output_weights)), self.output_bias)
                
        #Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        #Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        #Saving all variables
        self.saver = tf.train.Saver()
        
    #Define a function that predicts the label given the image     
    def y_predict(self, X_data, batch_size=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x = X_data[offset:offset+batch_size]
            y_pred[offset:offset+batch_size] = sess.run(tf.argmax(self.logits, 1), feed_dict={x:batch_x, keep_prob:1})
        return y_pred
    
    #Define a function that evaluates the model
    def evaluate(self, X_data, y_data, batch_size=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
        
x = tf.placeholder(tf.float32, (None, 32*32*3))
y = tf.placeholder(tf.int32, (None))

keep_prob = tf.placeholder(tf.float32)

epochs = 50
batch_size = 200
DIR = 'Saved_Models'

MLP = Multi_Layer_Perceptron(n_out=43)
model_name = "Multi_Layer_Perceptron_256_256_256"

accuracy_list =[]
loss_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Train on {} samples, validate on {} samples".format(X_train.shape[0], X_valid.shape[0]))
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, accuracy, loss = sess.run([MLP.training_operation, MLP.accuracy_operation, MLP.loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        #Accuracy on training set
        training_accuracy = MLP.evaluate(X_train, y_train)
        #Accuracy on validation set
        validation_accuracy = MLP.evaluate(X_valid, y_valid)
        print("Epoch {}/{}: Training Acc. = {:.3f}%, Validation Acc. = {:.3f}%".format(i+1, epochs, (training_accuracy*100), (validation_accuracy*100)))
    #Accuracy on test set
    test_accuracy = MLP.evaluate(X_test, y_test)
    print("Test Accuracy on {} Epochs = {:.3f}".format(epochs, test_accuracy*100))
    
    #Alternative way for calculating the accuracy on test set
    y_pred = MLP.y_predict(X_test)
    #acc = np.sum(y_pred==y_test)/np.size(y_pred)
    #print("Test Accuracy based on y_pred = {:.3f}%".format(acc*100))
    
    #Save the model for future use
    print("Saving the model...")
    MLP.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")

#Restore the saved model    
with tf.Session() as sess:
    MLP.saver.restore(sess, r'C:\Users\Bill\Desktop\traffic_signs_restricted_area\Saved_Models\Multi_Layer_Perceptron_256_256_256')
    print("MLP model restored")
    y_pred = MLP.y_predict(X_test)
    #Print the test accuracy
    test_accuracy = MLP.evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy*100))

#Plot loss and accuracy for every epoch
plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.title('Accuracy')
plt.plot(accuracy_list)
plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(loss_list)

#Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Log of normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#Write into a CSV file with columns ImageId & Label
submission = pd.DataFrame({"ImageId": list(range(0, len(y_pred))), "Predicted Label": y_pred, "True Label": y_test})
submission.to_csv("Multi_Layer_Perceptron_256_256_256_Results.csv", index=False)

#Testing the Model on Web Images
web_test_images = []
path = './traffic-signs-data/web/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    web_test_images.append(img)
web_labels = [14, 17, 18, 27, 38]
print("Number of new testing examples: ", len(web_test_images))

#Plotting the random Web Images
plt.figure(figsize=(15, 16))
for i in range(len(web_test_images)):
    plt.subplot(1, 5, i+1)
    plt.imshow(web_test_images[i])
    plt.xlabel(signnames[web_labels[i]])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()

#Reshape web_test_images so that it can be fed into MLP
web_test_images = np.asarray(web_test_images)
web_test_images = web_test_images.reshape(web_test_images.shape[0],32*32*3).astype('float32')

#Define a y_predict alike function which takes the input data and outputs the top k softmax probabilities 
def y_predict_model(Input_data, top_k=5):
    num_examples = len(Input_data)
    y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
    y_prob = np.zeros((num_examples, top_k))
    with tf.Session() as sess:
        MLP.saver.restore(sess, r'C:\Users\Bill\Desktop\traffic_signs_restricted_area\Saved_Models\Multi_Layer_Perceptron_256_256_256')
        y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(MLP.logits), k=top_k), 
                             feed_dict={x:Input_data, keep_prob:1.0})
    return y_prob, y_pred

y_prob, y_pred = y_predict_model(web_test_images)

#
test_accuracy = 0
for i in enumerate(web_test_images):
    accu = web_labels[i[0]] == np.asarray(y_pred[i[0]])[0]
    if accu == True:
        test_accuracy += 0.2
print("Web Images Test Accuracy = {:.1f}%".format(test_accuracy*100))

web_test_images = []
path = './traffic-signs-data/web/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    web_test_images.append(img)
web_test_images = np.asarray(web_test_images)
#web_test_images = web_test_images.reshape(web_test_images.shape[0], 32, 32, 3).astype('float32')

plt.figure(figsize=(15, 16))
for i in range(len(web_test_images)):
    plt.subplot(5, 2, 2*i+1)
    plt.imshow(web_test_images[i]) 
    plt.title(signnames[y_pred[i][0]])
    plt.axis('off')
    plt.subplot(5, 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), y_prob[i, :])
    labels = [signnames[j] for j in y_pred[i]]
    plt.yticks(np.arange(1, 6, 1), labels)
plt.show()
