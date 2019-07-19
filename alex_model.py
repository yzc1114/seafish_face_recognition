import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D, BatchNormalization

# AlexNet
model = Sequential()
#第一段
model.add(Conv2D(filters=96, kernel_size=(11,11),
                 strides=(4,4), padding='valid',
                 input_shape=(150, 150, 3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3),
                       strides=(2,2),
                       padding='valid'))
#第二段
model.add(Conv2D(filters=256, kernel_size=(5,5),
                 strides=(1,1), padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3),
                       strides=(2,2),
                       padding='valid'))
#第三段
model.add(Conv2D(filters=384, kernel_size=(3,3),
                 strides=(1,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=384, kernel_size=(3,3),
                 strides=(1,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3),
                 strides=(1,1), padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),
                       strides=(2,2), padding='valid'))
#第四段
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(50))
model.add(Activation('softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
