from time import sleep

import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('dices_model.h5')

# Open a handle to the default system camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the captured image to match the input size expected by your model
    img = cv2.resize(frame_rgb, (480, 480))

    # Convert the image to a numpy array and scale the pixel values
    img_array = image.img_to_array(img) / 255.

    # Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class of the image
    preds = model.predict(img_array)

    # Print the predicted class
    print("The predicted class is:", np.argmax(preds))

    # Display the resulting frame
    cv2.imshow('DnD Dice Classifier', frame)

    sleep(0.1)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()