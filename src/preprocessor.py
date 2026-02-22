
# Data generators for training, validation, and testing

datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize pixel values

train_generator = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

val_generator = datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

test_generator = datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')