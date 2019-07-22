import os
from keras.preprocessing.image import ImageDataGenerator
from vgg16_model import model as vgg_16_model
from alex_model import model as alex_net_model
from lenet_model import model as lenet_model
from my_smaller_vgg_model import model as smaller_vgg_model
from my_model import model as my_model
from collections import Counter
from keras.callbacks import TensorBoard
import time

raw_dir = os.path.join(os.getcwd(), 'dataset', 'raw')
log_dir = os.path.join(os.getcwd(), 'log')


epochs = 100
train_batch_size = 32
steps_per_epoch = 5144 / train_batch_size

tb = TensorBoard(log_dir=log_dir,  # 日志文件保存位置
                histogram_freq=0,  # 按照何等频率（每多少个epoch计算一次）来计算直方图，0为不计算
                batch_size=32,     # 用多大量的数据计算直方图
                write_graph=True,     # 是否在tensorboard中可视化计算图
                write_grads=False,    # 是否在tensorboard中可视化梯度直方图
                write_images=False,   # 是否在tensorboard中以图像形式可视化模型权重
                update_freq='batch')   # 更新频率)

train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_gen.flow_from_directory(
    raw_dir,
    target_size=(150, 150),
    batch_size=train_batch_size,
    class_mode='categorical',
)

validation_generator = val_data_gen.flow_from_directory(
    raw_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

#model = vgg_16_model
# model = alex_net_model
# model = lenet_model
#model = my_model
model = smaller_vgg_model

model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=20,
        class_weight=class_weights,
        callbacks=[tb],
)

model.save('saved_weights_{}.h5'.format(int(time.time())))
