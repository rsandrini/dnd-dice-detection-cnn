from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Assuming images are 128x128

img_height_camera, img_width_camera = 1440, 960
img_height, img_width = 480, 480
batch_size = 32

# Use ImageDataGenerator to load your data
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)



# You need two separate generators for the dice detection and dice number classification tasks
dice_detection_generator = datagen.flow_from_directory('dataset/dice_detection/train/',
                                                       target_size=(img_height_camera, img_width_camera),
                                                       batch_size=batch_size,
                                                       class_mode='binary')  # It's a binary task (dice or no dice)


test_dice_detection = datagen.flow_from_directory('dataset/dice_detection/test',
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

dice_number_classification_generator = datagen.flow_from_directory('dataset/number_detection/train/d20',
                                                                   target_size=(img_height, img_width),
                                                                   batch_size=batch_size,
                                                                   class_mode='categorical')  # 20 classes for the 20-sided dice


test_number_detection = datagen.flow_from_directory('dataset/number_detection/test/d20',
                                                      target_size=(img_height, img_width),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')


# Assuming your images are 480x480 pixels
input_shape = (img_height_camera, img_width_camera, 3)

dice_detection_model = Sequential()

# Add convolutional layers
dice_detection_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
dice_detection_model.add(MaxPooling2D(pool_size=(2, 2)))

dice_detection_model.add(Conv2D(64, (3, 3), activation='relu'))
dice_detection_model.add(MaxPooling2D(pool_size=(2, 2)))

dice_detection_model.add(Conv2D(128, (3, 3), activation='relu'))
dice_detection_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutional layers
dice_detection_model.add(Flatten())

# Add fully connected layers
dice_detection_model.add(Dense(128, activation='relu'))

# Add a binary output layer (for "dice" or "no dice")
dice_detection_model.add(Dense(1, activation='sigmoid'))

# Compile the model
dice_detection_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define your CNN model for dice number classification
dice_number_classification_model = Sequential()

dice_number_classification_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dice_number_classification_model.fit(dice_number_classification_generator, epochs=50, validation_data=test_dice_detection)

dice_number_classification_model = Sequential()
dice_number_classification_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
dice_number_classification_model.add(MaxPooling2D(pool_size=(2, 2)))

dice_number_classification_model.add(Conv2D(32, (3, 3), activation='relu'))
dice_number_classification_model.add(MaxPooling2D(pool_size=(2, 2)))

dice_number_classification_model.add(Conv2D(64, (3, 3), activation='relu'))
dice_number_classification_model.add(MaxPooling2D(pool_size=(2, 2)))

dice_number_classification_model.add(Flatten())
dice_number_classification_model.add(Dense(64, activation='relu'))
dice_number_classification_model.add(Dense(20, activation='softmax'))  # Assuming a 20-sided die plus an error class

# Compile and train the model
dice_number_classification_model.compile(loss='categorical_crossentropy',
                                         optimizer='adam',
                                         metrics=['accuracy'])

dice_number_classification_model.fit(dice_number_classification_generator,
                                     epochs=50,
                                     validation_data=test_number_detection)

dice_number_classification_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dice_number_classification_model.fit(dice_number_classification_generator, epochs=50)

# Save the models
dice_detection_model.save('dice_detection_model.h5')
dice_number_classification_model.save('dice_number_classification_model.h5')


# # Define your CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(20, activation='softmax'))  # Assuming a 20-sided die plus an error class
#
# # Compile and train the model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(train_generator,
#           epochs=50,
#           validation_data=test_generator)
#
# # Save the entire model as a SavedModel.
# model.save('dices_model.h5')
