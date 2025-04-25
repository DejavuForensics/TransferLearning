import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet:
    def __init__(self, epochs=50, batch_size=128, load_weights=True, num_classes=10, transfer_learning=False):
        self.name = 'lenet'
        self.model_filename = 'FeatureExtractor/networks/pretrained_weights/lenet.h5'
        self.num_classes = num_classes
        self.input_shape = (32, 32, 3)
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = 0.0001
        self.log_filepath = r'FeatureExtractor/networks/pretrained_weights/lenet/'
        self.transfer_learning = transfer_learning

        # Cria diretórios se não existirem
        os.makedirs('FeatureExtractor/networks/pretrained_weights', exist_ok=True)
        os.makedirs(self.log_filepath, exist_ok=True)

        # Inicializa o modelo
        self._model = self.build_model()
        
        if load_weights:
            try:
                # Tenta carregar os pesos pré-treinados
                self._model.load_weights(self.model_filename)
                print('Successfully loaded weights for', self.name)
                
                if transfer_learning:
                    # Congela as camadas convolucionais
                    for layer in self._model.layers[:-4]:  # Congela até a última camada convolucional
                        layer.trainable = False
                    print('Camadas convolucionais congeladas para transfer learning')
                    
                    # Recompila o modelo com as camadas congeladas
                    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
                    self._model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            except (ImportError, ValueError, OSError):
                print('Failed to load weights for', self.name)
                print('Training new model...')
                self.train()
    
    def count_params(self):
        return self._model.count_params()

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]

    def color_preprocessing(self, x_train,x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def train(self, x_train=None, y_train=None, x_test=None, y_test=None):
        if x_train is None or y_train is None:
            # Se não fornecer dados, usa CIFAR-10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)
        else:
            # Converte os labels para one-hot encoding
            y_train = keras.utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        # Pré-processamento das cores
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # Callbacks
        tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        ckpt = ModelCheckpoint(self.model_filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        cbks = [tb_cb, ckpt]

        # Data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.125,
            height_shift_range=0.125,
            fill_mode='constant',
            cval=0.
        )

        datagen.fit(x_train)

        # Treinamento
        self._model.fit(
            datagen.flow(x_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=x_train.shape[0] // self.batch_size,
            epochs=self.epochs,
            callbacks=cbks,
            validation_data=(x_test, y_test)
        )
        
        self._model.save(self.model_filename)
        self.param_count = self._model.count_params()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape, kernel_regularizer=regularizers.l2(self.weight_decay)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(self.weight_decay)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes),
            tf.keras.layers.Activation('softmax')
        ])

        # set optimizer
        sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        return model



