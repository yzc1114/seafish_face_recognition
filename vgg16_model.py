from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D

# 定义输入
input_shape = (150, 150, 3)  # RGB影像224x224（height,width,channel)

# 使用序贯模型(sequential)来定义
model = Sequential(name='vgg16-sequential')

# 第1个卷积区块(block1)
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=input_shape,name='block1_conv1'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu',name='block1_conv2'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block1_pool'))

# 第2个卷积区块(block2)
model.add(Conv2D(128,(3,3),padding='same',activation='relu',name='block2_conv1'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu',name='block2_conv2'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block2_pool'))

# 第3个区块(block3)
model.add(Conv2D(256,(3,3),padding='same',activation='relu',name='block3_conv1'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',name='block3_conv2'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',name='block3_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block3_pool'))

# 第4个区块(block4)
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block4_conv1'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block4_conv2'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block4_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block4_pool'))

# 第5个区块(block5)
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block5_conv1'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block5_conv2'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',name='block5_conv3'))
model.add(MaxPool2D((2,2),strides=(2,2),name='block5_pool'))

# 前馈全连接区块
model.add(Flatten(name='flatten'))
model.add(Dense(4096,activation='relu',name='fc1'))
model.add(Dense(4096,activation='relu',name='fc2'))
model.add(Dense(50,activation='softmax',name='predictions'))

# 打印网络结构
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
