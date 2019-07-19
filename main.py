import os
from keras.preprocessing.image import ImageDataGenerator
from vgg16_model import model as vgg_16_model
from alex_model import model as alex_net_model
from lenet_model import model as lenet_model
from collections import Counter

raw_dir = os.path.join(os.getcwd(), 'dataset', 'raw')

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
    batch_size=32,
    class_mode='categorical',
)

validation_generator = val_data_gen.flow_from_directory(
    raw_dir,
    target_size=(150, 150),
    batch_size=4,
    class_mode='categorical'
)


counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

# model = vgg_16_model
# model = alex_net_model
model = lenet_model

model.fit_generator(
        train_generator,
        steps_per_epoch=172,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=20,
        class_weight=class_weights
)

model.save_weights(os.getcwd())
