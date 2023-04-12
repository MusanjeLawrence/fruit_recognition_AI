
#importing libraries


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#loading model

cnn = tf.keras.models.load_model("/content/trained_model.h5")

#visualization and performing prediction on single image

import cv2
image_path="/content/drive/MyDrive/fruits-360-original-size/fruits-360-original-size/Test/carrot_1/r0_107.jpg"
img = cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

#testing model

image = tf.keras.preprocessing.image.load_img(image_path, target_size = (64, 64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) #converting single image to batch
predictions  = cnn.predict(input_arr)

print(predictions[0])
print(max(predictions[0]))

test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/fruits-360-original-size/fruits-360-original-size/Test',
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

#test_set.class_names

result_index = np.where(predictions[0] == max(predictions[0]))
print(result_index[0][0])
#display image

plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

#performing a single prediction
print("It is a {} ".format(test_set.class_names[result_index[0][0]]))

