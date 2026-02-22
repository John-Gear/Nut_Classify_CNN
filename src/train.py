import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adamimport os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), input_shape=INPUT_SHAPE, activation='relu'),  # First convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # First max pooling layer
    Conv2D(32, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Second max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Third max pooling layer
    Flatten(),  # Flatten layer to convert feature maps into a vector
    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# Train the model
model.fit(
    train_generator,
    steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VALIDATION_SAMPLES // BATCH_SIZE
)

# Evaluate the model on test data
scores = model.evaluate(test_generator, steps=TEST_SAMPLES // BATCH_SIZE)
print(f"Test accuracy: {scores[1] * 100:.2f}%")

# Save the trained model
model.save("nut_classifier.h5")  # Save model in .h5 format