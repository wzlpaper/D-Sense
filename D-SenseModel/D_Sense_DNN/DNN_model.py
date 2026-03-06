# This code is provided by Zhelun Wang.
# Email: wzlpaper@126.com

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, GRU, LSTM, Bidirectional,
                                     Dense, Flatten, Dropout, Conv2D,
                                     MaxPooling2D, TimeDistributed,
                                     Conv3D, MaxPooling3D, Concatenate,
                                     Lambda, GlobalAveragePooling1D, Reshape,
                                     LayerNormalization, Add)
from keras.models import Model
import os
import pandas as pd
from tensorflow.keras.callbacks import Callback

''' 
    This code defines the 'myModel' class with various model architectures:
    CNN + RNN: Convolutional layers followed by recurrent layers (GRU/LSTM/BiGRU/BiLSTM).
    Transformer: Transformer-based architecture.
    CNN + Transformer: Combines CNN and Transformer layers.
    RNN + Transformer: Combines RNN layers with Transformer layers.
    CNN + RNN + Transformer: Hybrid architecture combining CNN, RNN, and Transformer.
'''

class myModel(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 n_class,
                 N_rnn,
                 learning_rate,
                 dropout_ratio,
                 RNN_type,
                 model_type,
                 Tasks,
                 use_DWM
                 ):
        super(myModel, self).__init__()
        self.input_shape = input_shape
        self.n_class = n_class
        self.N_rnn = N_rnn
        self.learning_rate = learning_rate
        self.dropout_ratio = dropout_ratio
        self.RNN_type = RNN_type
        self.model_type = model_type
        self.Tasks = Tasks
        self.use_DWM = use_DWM
        self.build_model()

    def build_model(self):
        model_input = Input(shape=self.input_shape, dtype='float32', name='model_input')
        model_map = {
            'CNN-RNN': self.CNN_RNN,
            'Transformer': self.Transformer,
            'CNN-Transformer': self.CNN_Transformer,
            'RNN-Transformer': self.RNN_Transformer,
            'CNN-RNN-Transformer': self.CNN_RNN_Transformer,
            'CNN': self.CNN,
            'RNN': self.RNN
        }
        if self.model_type in model_map:
            outputs = model_map[self.model_type](model_input)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.dwm_weights = [tf.Variable(1.0, dtype=tf.float32, trainable=False)
                            for _ in range(len(self.Tasks))]
        self.model = Model(inputs=model_input, outputs=outputs)
        self.model.dwm_weights = self.dwm_weights
        self.model.task_names = self.Tasks
        losses = {}
        metrics = {}
        loss_weights = {}
        if self.use_DWM:
            for i, name in enumerate(self.Tasks):
                loss_fn = self.create_weighted_loss(i)
                losses[f'model_output_{i}'] = loss_fn
                metrics[f'model_output_{i}'] = 'accuracy'
                loss_weights[f'model_output_{i}'] = 1.0
        else:
            loss = 'categorical_crossentropy'
            for i, name in enumerate(self.Tasks):
                losses[f'model_output_{i}'] = loss
                metrics[f'model_output_{i}'] = 'accuracy'
                loss_weights[f'model_output_{i}'] = 1.0
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )
        self.summary()

    def create_weighted_loss(self, task_idx):
        def loss(y_true, y_pred):
            loss_value = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            current_weight = self.dwm_weights[task_idx]
            weighted_loss_value = current_weight * loss_value
            return weighted_loss_value
        return loss

    def Outputs(self, x):
        model_outputs = []
        for i in range(len(self.n_class)):
            output = Dense(len(self.n_class[i]),
                           activation='softmax',
                           dtype='float32',
                           name=f'model_output_{i}')(x)
            model_outputs.append(output)
        return model_outputs

    def call(self, inputs):
        return self.model(inputs)

    def CNN_RNN_init(self, model_input):
        x = TimeDistributed(Conv2D(16,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   data_format='channels_last'))(model_input)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(64, activation='relu'))(x)
        x = TimeDistributed(Dropout(self.dropout_ratio))(x)
        return TimeDistributed(Dense(64, activation='relu'))(x)

    def CNN_RNN(self, model_input):
        model_input_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 0], axis=-1))(model_input)
        model_input_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=-1))(model_input)
        x1 = self.CNN_RNN_init(model_input_1)
        x2 = self.CNN_RNN_init(model_input_2)
        x = Concatenate(axis=-1)([x1, x2])
        if self.RNN_type == 'GRU':
            x = GRU(self.N_rnn, return_sequences=False)(x)
        elif self.RNN_type == 'LSTM':
            x = LSTM(self.N_rnn, return_sequences=False)(x)
        elif self.RNN_type == 'BiGRU':
            x = Bidirectional(GRU(self.N_rnn, return_sequences=False))(x)
        elif self.RNN_type == 'BiLSTM':
            x = Bidirectional(LSTM(self.N_rnn, return_sequences=False))(x)
        else:
            raise ValueError(f"Unsupported RNN type: {self.RNN_type}")
        x = Dropout(self.dropout_ratio)(x)
        return self.Outputs(x)

    def Transformer_init(self, x):
        x = PositionalEncoding(T_MAX=self.input_shape[0], dim=128)(x)
        for _ in range(4):
            x = encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=self.dropout_ratio)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.dropout_ratio)(x)
        return self.Outputs(x)

    def Transformer(self, model_input):
        x = Reshape((self.input_shape[0], -1))(model_input)
        x = Dense(128)(x)
        return self.Transformer_init(x)

    def CNN_Transformer(self, model_input):
        model_input_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 0], axis=-1))(model_input)
        model_input_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=-1))(model_input)
        x1 = self.CNN_RNN_init(model_input_1)
        x2 = self.CNN_RNN_init(model_input_2)
        x = Concatenate(axis=-1)([x1, x2])
        return self.Transformer_init(x)

    def RNN_Transformer(self, model_input):
        x = Reshape((self.input_shape[0], -1))(model_input)
        if self.RNN_type == 'GRU':
            x = GRU(self.N_rnn, return_sequences=True)(x)
        elif self.RNN_type == 'LSTM':
            x = LSTM(self.N_rnn, return_sequences=True)(x)
        elif self.RNN_type == 'BiGRU':
            x = Bidirectional(GRU(self.N_rnn, return_sequences=True))(x)
        elif self.RNN_type == 'BiLSTM':
            x = Bidirectional(LSTM(self.N_rnn, return_sequences=True))(x)
        else:
            raise ValueError(f"Unsupported RNN type: {self.RNN_type}")
        return self.Transformer_init(x)

    def CNN_RNN_Transformer(self, model_input):
        model_input_1 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 0], axis=-1))(model_input)
        model_input_2 = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=-1))(model_input)
        x1 = self.CNN_RNN_init(model_input_1)
        x2 = self.CNN_RNN_init(model_input_2)
        x = Concatenate(axis=-1)([x1, x2])
        if self.RNN_type == 'GRU':
            x = GRU(self.N_rnn, return_sequences=True)(x)
        elif self.RNN_type == 'LSTM':
            x = LSTM(self.N_rnn, return_sequences=True)(x)
        elif self.RNN_type == 'BiGRU':
            x = Bidirectional(GRU(self.N_rnn, return_sequences=True))(x)
        elif self.RNN_type == 'BiLSTM':
            x = Bidirectional(LSTM(self.N_rnn, return_sequences=True))(x)
        else:
            raise ValueError(f"Unsupported RNN type: {self.RNN_type}")
        return self.Transformer_init(x)

    def CNN(self, model_input):
        x = Conv3D(16,
                   kernel_size=(3, 3, 3),
                   activation='relu',
                   data_format='channels_last')(model_input)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_ratio)(x)
        x = Dense(64, activation='relu')(x)
        return self.Outputs(x)

    def RNN(self, model_input):
        x = Reshape((self.input_shape[0], -1))(model_input)
        if self.RNN_type == 'GRU':
            x = GRU(self.N_rnn, return_sequences=False)(x)
        elif self.RNN_type == 'LSTM':
            x = LSTM(self.N_rnn, return_sequences=False)(x)
        elif self.RNN_type == 'BiGRU':
            x = Bidirectional(GRU(self.N_rnn, return_sequences=False))(x)
        elif self.RNN_type == 'BiLSTM':
            x = Bidirectional(LSTM(self.N_rnn, return_sequences=False))(x)
        else:
            raise ValueError(f"Unsupported RNN type: {self.RNN_type}")
        return self.Outputs(x)

    def summary(self):
        return self.model.summary()

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 T_MAX,
                 dim):
        super(PositionalEncoding, self).__init__()
        self.T_MAX = T_MAX
        self.dim = dim
        pos = np.arange(self.T_MAX)[:, np.newaxis]
        i = np.arange(self.dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                           key_dim=head_size,
                                           dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x_ = LayerNormalization(epsilon=1e-6)(x)
    x_ = Dense(ff_dim, activation='relu')(x_)
    x_ = Dropout(dropout)(x_)
    x_ = Dense(inputs.shape[-1])(x_)
    return Add()([x_, x])

class LossHistory(Callback):
    def __init__(self, tasks, output_filename="loss_history.csv"):
        super().__init__()
        self.tasks = tasks
        self.output_path = os.path.join(os.getcwd(), output_filename)
        self.history = {task: [] for task in tasks}

    def on_epoch_end(self, epoch, logs=None):
        for i, task in enumerate(self.tasks):
            loss_key = f'model_output_{i}_loss'
            if loss_key in logs:
                self.history[task].append(logs[loss_key])
            else:
                self.history[task].append(None)
        self.save_loss_history()

    def save_loss_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_path, index=False)