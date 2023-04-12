
#importing dataset


from google.colab import drive
drive.mount('/content/drive')

#importing python libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#data preprocessing

#training image preprocessing

training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/fruits-360-original-size/fruits-360-original-size/Training',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#validation image preprocesing

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/fruits-360-original-size/fruits-360-original-size/Validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#building model

cnn = tf.keras.models.Sequential()

#building convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(64,64,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2, strides =2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2, strides =2))

cnn.add(tf.keras.layers.Dropout(0.5)) #to avoid overfitting

cnn.add(tf.keras.layers.Flatten())


#using dense function to create neurons

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#creating an output layer

#using 24 because we have 24 classes in the dataset and this is the output layer
cnn.add(tf.keras.layers.Dense(units=24, activation ='softmax'))

#compiling and training phase

#using categorical_crossentropy for loss because we have more than 2 class
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=30)

#saving model

cnn.save('trained_model.h5')

training_history.history #returns history dictionary

#recording history in a json file
import json
with open('training_hist.json', 'w') as f:
  json.dump(training_history.history, f)

print(training_history.history.keys())

#calculating accuracy of model achieved on model validation set

print("Validation set accuracy: {} %" .format(training_history.history['val_accuracy'][-1]*100))

#Accuracy visualization

#training visualization


epochs = [i for i in range(1,31)] 
plt.plot(epochs, training_history.history['accuracy'], color='green')
plt.xlabel('Epochs') #labelling for x-axis
plt.ylabel('Training Accuracy') #labelling for y a-xis
plt.title('Visualization of training accuracy results')#title label
plt.show()#to remove text above graph if not the title

#validation accuracy

plt.plot(epochs, training_history.history['val_accuracy'], color='red')
 plt.xlabel('Number of Epochs') #labelling for x-axis
 plt.ylabel('validation Accuracy') #labelling for y a-xis
 plt.title('Visualization of validation accuracy results')#title label
 plt.show()#to remove text above graph if not the title