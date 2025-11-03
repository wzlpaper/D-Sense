# This code is provided by Zhelun Wang. Stay Optimistic.
# Email: wzlpaper@126.com

import os
import tensorflow as tf
from D_Sense_DNN.data_loader import load_data, load_Gait_data, onehot_encoding, Confusion_matrix
from D_Sense_DNN.DNN_model import myModel, LossHistory
from D_Sense_DNN.DWM import DWM

'''
    Each task and model has been tested. If you cannot run the code,
    please check the following three points:
    
    1 Whether the environment configuration meets the requirements:
           Ubuntu: 22.04 LTS (Win11 Professional, WSL2),
           GPU: RTX 4060 (40GB),
           CPU: i9-14900HX,
           RAM: 64GB.
           TensorFlow: 2.18.0
           CUDA: 12.5
    2 Whether the ADP files are intact and the paths are correct
    3 This file should be at the same level as the D_Sense_DNN folder
'''

# Supported Wireless Sensing Tasks
tasks = {
        'In-domain-Gesture':              'In-domain Gesture recognition', # T = 0
        'In-domain-User':                 'In-domain User recognition', # T = 1
        'In-domain-Gesture-User':         'In-domain Gesture and User synchronized recognition', # T = 2
        'Cross-orientation-Gesture':      'Cross-orientation Gesture recognition', # T = 3
        'Cross-location-Gesture':         'Cross-location Gesture recognition', # T = 4
        'Cross-orientation-User':         'Cross-orientation User recognition', # T = 5
        'Cross-location-User':            'Cross-location User recognition', # T = 6
        'Cross-orientation-Gesture-User': 'Cross-orientation Gesture and User synchronized recognition', # T = 7
        'Cross-location-Gesture-User':    'Cross-location Gesture and User synchronized recognition', # T = 8
        'Orientation':                    'Orientation recognition', # T = 9
        'Location':                       'Location recognition', # T = 10
        'Gesture-Orientation':            'Gesture and Orientation synchronization recognition', # T = 11
        'Gesture-Location':               'Gesture and Location synchronization recognition', # T = 12
        'User-Orientation':               'User and Orientation synchronization recognition', # T = 13
        'User-Location':                  'User and Location synchronization recognition', # T = 14
        'Orientation-Location':           'Orientation and Location synchronization recognition', # T = 15
        'Gesture-User-Location':          'Gesture, User and Location synchronization recognition',# T = 16
        'Gesture-User-Orientation':       'Gesture, User and Orientation synchronization recognition', # T = 17
        'Gesture-Orientation-Location':   'Gesture, Orientation and Location synchronization recognition',# T = 18
        'User-Orientation-Location':      'User, Orientation and Location synchronization recognition', # T = 19
        'Four-task':                      'Gesture, User, Orientation and Location synchronization recognition', # T = 20
        'Gait':                           'Gait recognition', # T = 21
        'Track':                          'Track recognition', # T = 22
        'Gait-Track':                     'Gait and Track synchronization recognition' # T = 23
        }

# Available Models
# RNN type in {GRU LSTM BiGRU BiLSTM}
models = [
         'CNN-RNN', # M = 0
         'Transformer', # M = 1
         'CNN-Transformer', # M = 2
         'RNN-Transformer', # M = 3
         'CNN-RNN-Transformer', # M = 4
         'CNN', # M = 5
         'RNN' # M = 6
         ]

# Parameters
                                         # Note: The ADP folders for Gait-related tasks and Gesture-related tasks should be different.
test_set_ratio    = 0.2;                   ADP_dir         = '/home/wzl/paper/TMC/0.04'
                                         # domain_idx      = domains[I] 'I IN [0, 4]'
domains           = [1, 2, 3, 4, 5];       domain_idx      = domains[1]
gesture_cats      = [1, 2, 3, 4, 5, 6];    N_gesture       = len(gesture_cats)
user_cats         = [1, 2, 3];             N_user          = len(user_cats)
gait_cats         = [1, 2, 3, 4, 5, 6, 7]; N_gait          = len(gait_cats)
orientation_cats  = [1, 2, 3, 4, 5];       N_orientation   = len(orientation_cats)
track_cats        = [1, 2, 3, 4];          N_track         = len(track_cats)
location_cats     = [1, 2, 3, 4, 5];       N_location      = len(location_cats)
t_max             = 0;                     n_epochs        = 100
dropout_ratio     = 0.5;                   N_RNN, RNN_Type = 128, 'GRU'
n_batch           = 32;                    learning_rate   = 0.001
use_DWM           = True
# task_, model_   = list(tasks.items())[T], models[M] 'T IN [0, 23], M IN [0, 6]'
task_, model_     = list(tasks.items())[-4], models[0] # Select Task and Model
print(f"Selected task: {task_[1]}\nSelected model: {model_}")

# If your computer has a GPU, you can run the code in this module to utilize the GPU environment.
GPUs = tf.config.list_physical_devices('GPU')
if len(GPUs) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.config.experimental.set_memory_growth(GPUs[0], True)
    tf.random.set_seed(1)
else:
    print('No GPU available, using CPU...')

if task_[0] == 'Gait' or task_[0] == 'Track' or task_[0] == 'Gait-Track':
    data_loader = load_Gait_data(
                                 data_path=ADP_dir,
                                 user_cats=gait_cats,
                                 track_cats=track_cats,
                                 t_max=t_max,
                                 test_set_ratio=test_set_ratio,
                                 task=task_[0]
                                )
else:
    data_loader = load_data(
                            data_path=ADP_dir,
                            gesture_cats=gesture_cats,
                            user_cats=user_cats,
                            orientation_cats=orientation_cats,
                            location_cats=location_cats,
                            t_max=t_max,
                            test_set_ratio=test_set_ratio,
                            domain=domain_idx,
                            task=task_[0]
                            )

loader, T_max, Tasks, n_class = data_loader.get_loader()
data_train, data_test = loader['data_train'], loader['data_test']
print(data_train.shape, data_test.shape)

# ADP is a cross-domain feature of feature patterns, and there's no need to use a DNN to
# extract higher-level features every time. Therefore, you can save the pre-trained model
# and directly use it in similar tasks next time.
# save_model = Ture
# if save_model:
#     model = load_model('model_D_Sense_trained.h5')
#     model.summary()
# else:
#     model = myModel()
#     model.fit()
#     model.save('model_D_Sense_trained.h5')

# Create the DWM
DWM_ = DWM(
           task_names=Tasks,
           total_epochs=n_epochs,
           use_gradient_alignment=True,
           use_coba_mechanism=True,
           verbose=True
           )
loss_history = LossHistory(tasks=Tasks)

model = myModel(
                input_shape=(T_max, 20, 20, 2),
                n_class=n_class,
                N_rnn=N_RNN,
                learning_rate=learning_rate,
                dropout_ratio=dropout_ratio,
                RNN_type=RNN_Type,
                model_type=model_,
                Tasks=Tasks,
                use_DWM=use_DWM
                )

Train_Model = model.fit(
                   loader['data_train'],
                        [onehot_encoding(loader[f'{task}_label_train'], eval(f'N_{task.lower()}'))
                        for i, task in enumerate(Tasks)],
                        batch_size=n_batch,
                        epochs=n_epochs,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=True,
                        callbacks=[
                        [DWM_] if use_DWM
                        else [loss_history]
                        ]
                        )

label_test = {f'{task}_label_test': onehot_encoding(loader[f'{task}_label_test'], eval(f'N_{task.lower()}'))
              for task in Tasks}
pred_label = model.predict(loader['data_test'])
if len(Tasks) == 1:
    Confusion_matrix(pred_label, label_test, Tasks[0])
else:
    for i, task in enumerate(Tasks):
        Confusion_matrix(pred_label[i], label_test, task)