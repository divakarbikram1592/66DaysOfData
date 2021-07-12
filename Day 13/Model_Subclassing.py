import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 32
epochs = 15

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10_000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

class MNIST(Model):
    def __init__(self):
        super().__init__()
        self.conv_1 = layers.Conv2D(32, kernel_size = (3, 3), activation = "relu")
        self.conv_2 = layers.Conv2D(64, kernel_size = (3, 3), activation = "relu")
        self.max_pool = layers.MaxPooling2D(pool_size = (2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.out =  layers.Dense(num_classes, activation = "softmax")

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.out(x)

model = MNIST()

loss_fn = CategoricalCrossentropy()
optimizer = Adam()

train_loss = Mean(name = 'train_loss')
train_accuracy = CategoricalAccuracy(name = 'train_accuracy')

test_loss = Mean(name = 'test_loss')
test_accuracy = CategoricalAccuracy(name = 'test_accuracy')

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(images, labels):
    predictions = model(images, training = False)
    t_loss = loss_fn(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(epochs):
    start_time = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    time_elapsed = time.time() - start_time

    print(
        f'\nEpoch {epoch + 1:2}: '
        f'Loss: {train_loss.result():5.3f}, '
        f'Accuracy: {train_accuracy.result():5.3f}, '
        f'Test Loss: {test_loss.result():5.3f}, '
        f'Test Accuracy: {test_accuracy.result():5.3f}, '
        f'Time Elapsed: {time_elapsed:4.1f}s'
    )