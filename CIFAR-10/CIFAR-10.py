import keras
import time
from keras.callbacks import TensorBoard
from keras import layers


NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

dense_layers= [1]
layer_sizes = [256]
conv_layers = [3]

'''for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            ##NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            NAME =  f"CIFAR-10-CNN-Dropout{int( time.time())}"

            model = keras.Sequential()
            
            model.add(layers.Conv2D(layer_size, (3, 3), activation="relu", input_shape=(32, 32, 3)))
            model.add(layers.MaxPool2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.2))

            for i in range(conv_layer - 1):
                model.add(layers.Conv2D(layer_size, (3, 3), activation="relu"))
                model.add(layers.MaxPool2D(pool_size=(2, 2)))
                model.add(layers.Dropout(0.2))

            model.add(layers.Flatten())
            for i in range(dense_layer):
                model.add(layers.Dense(layer_size, activation="relu"))

            model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

            tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])'''

NAME =  f"CIFAR-10-CNN-DropoutIncremental-ValData{int(time.time())}"

model = keras.Sequential()
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(256, (3, 3), padding='same',activation='relu',))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=75, validation_data=(x_test, y_test), callbacks=[tensorboard])
