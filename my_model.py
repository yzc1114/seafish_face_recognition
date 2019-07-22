from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adadelta

optimizer = Adadelta()

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(240, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(50, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
