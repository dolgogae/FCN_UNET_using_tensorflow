import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from config import model_name
from model import segmentation_model
# from segmentation_model import unet, fcn_8
from config import model_name, imshape, n_classes

def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))

def train():

    if 'unet' in model_name:
        train_model = segmentation_model(model_name, imshape).model
    elif 'fcn_8' in model_name:
        train_model = segmentation_model(model_name, imshape).model

    image_paths = [os.path.join('images', x) for x in sorted_fns('images')]
    annot_paths = [os.path.join('annotated', x) for x in sorted_fns('annotated')]

    tg = DataGenerator(image_paths=image_paths, annot_paths=annot_paths,
                        batch_size=5, augment=True)

    checkpoint = ModelCheckpoint(os.path.join('models', model_name+'.model'), monitor='dice', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=10)
    train_model.fit_generator(generator=tg,
                              steps_per_epoch=len(tg),
                              epochs=500,
                              verbose=1,
                              callbacks=[checkpoint])

def main():
    train()

if __name__ == '__main__':
    main()