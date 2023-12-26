import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model(shape, num_classes=10):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape),
        layers.MaxPooling2D((2, 2)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])


# Step 4: Define Augmentation Models
augmentation_methods = [
    lambda img: tf.image.rgb_to_grayscale(img),
    lambda img: img,
    lambda img: tf.image.flip_left_right(img),
    lambda img: tf.image.adjust_brightness(img, delta=0.2),
    lambda img: tf.image.central_crop(img, central_fraction=0.8),
    lambda img: tf.image.random_crop(img),
    lambda img: tf.image.stateless_random_flip_left_right(img, seed=(1, 2))

]

for i, aug_method in enumerate(augmentation_methods):
    tf.keras.backend.clear_session()

    # Get the shape of the augmented training data
    augmented_x_train = aug_method(x_train)
    augmented_x_test = aug_method(x_test)

    input_shape = augmented_x_train.shape[1:]

    # Create the model with the dynamic input shape
    model = create_model(input_shape)

    # Train the model with augmented data
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(augmented_x_train, y_train, epochs=10, validation_data=(augmented_x_test, y_test))

    # Evaluate and print accuracy
    accuracy = model.evaluate(augmented_x_train, y_train)[1]
    print(f"Model with Augmentation Method {i + 1}: Accuracy - {accuracy}")
