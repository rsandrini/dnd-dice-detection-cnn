from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('dices_model.h5')

# Load the image file and resize it to the expected size
img = image.load_img('manual_test/d20_18.jpeg', target_size=(480, 480))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add a third dimension (the color channels)
img_array = np.expand_dims(img_array, axis=0)

# Scale the image pixels by 255
img_array /= 255.

# Use the model to predict the image's class
preds = model.predict(img_array)

# preds will be a 20-element array with the probabilities for each class
# To get the class with the highest probability, you can use argmax
print("The predicted class is:", np.argmax(preds))