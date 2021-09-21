import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt


data_generator = ImageDataGenerator(rescale = 1.0/255)
training_iterator = data_generator.flow_from_directory('train', class_mode = 'categorical', batch_size = 16, color_mode = 'grayscale')
validation_iterator = data_generator.flow_from_directory('test', class_mode = 'categorical', batch_size = 16, color_mode = 'grayscale')

#sample_input, sample_label = training_iterator.next()
#print(sample_input.shape, sample_label.shape)

model = Sequential()
model.add(tf.keras.layers.Input(shape = (256,256,1)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))
model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005),
  loss = tf.keras.losses.CategoricalCrossentropy(),
  metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)
history = model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/16,
        epochs=8,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/16)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
 
fig.savefig('my_plot1.png')
plt.show()