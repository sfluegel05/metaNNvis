import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def get_tf_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    # Set random labels and a squares in top left corner corresponding to the labels
    rng = np.random.default_rng(724)
    y_train = rng.integers(0, 10, size=y_train.size)
    y_test = rng.integers(0, 10, size=y_test.size)
    print(y_train[0:8])
    print(y_test[:8])
    for i in range(y_train.size):
        x_train[i, :5, :5] = y_train[i] / 10
    for i in range(y_test.size):
        x_test[i, :5, :5] = y_test[i] / 10

    return (x_train, y_train), (x_test, y_test)


def train_tf_net():
    (x_train, y_train), (x_test, y_test) = get_tf_data()
    tf_net = get_tf_net()
    tf_net.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics=['accuracy'])

    history = tf_net.fit(x_train, y_train, epochs=10,
                         validation_data=(x_test, y_test))

    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = tf_net.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')
    tf_net.save(os.path.join('..', 'models', 'tf_clever_hans'))


def get_tf_net():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(10, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(20, (5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    train_tf_net()
