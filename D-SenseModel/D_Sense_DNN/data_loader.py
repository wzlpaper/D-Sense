# This code is provided by Zhelun Wang.
# Email: wzlpaper@126.com

import os
import numpy as np
import scipy.io as scio
import random
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

''' 
    The extraction and division of datasets for different tasks.
    In-domain tasks:
    'In-domain-Gesture',      # Task: In-domain gesture recognition
    'In-domain-User',         # Task: In-domain user recognition
    'In-domain-Gesture-User', # Task: In-domain gesture and user synchronized recognition

    Cross-domain tasks:
    'Cross-orientation-Gesture',      # Task: Cross-orientation gesture recognition
    'Cross-location-Gesture',         # Task: Cross-location gesture recognition
    'Cross-orientation-User',         # Task: Cross-orientation user recognition
    'Cross-location-User',            # Task: Cross-location user recognition
    'Cross-orientation-Gesture-User', # Task: Cross-orientation gesture and user synchronized recognition
    'Cross-location-Gesture-User',    # Task: Cross-location gesture and user synchronized recognition

    Single modality recognition tasks:
    'Orientation', # Task: Orientation recognition
    'Location',    # Task: Location recognition

    Multi-modality synchronization recognition tasks:
    'Gesture-Orientation',  # Task: Gesture and orientation synchronization recognition
    'Gesture-Location',     # Task: Gesture and location synchronization recognition
    'User-Orientation',     # Task: User and orientation synchronization recognition
    'User-Location',        # Task: User and location synchronization recognition
    'Orientation-Location', # Task: Orientation and location synchronization recognition

    Combined modality recognition tasks:
    'Gesture-User-Location',        # Task: Gesture, user, and location synchronization recognition
    'Gesture-User-Orientation',     # Task: Gesture, user, and orientation synchronization recognition
    'Gesture-Orientation-Location', # Task: Gesture, orientation, and location synchronization recognition
    'User-Orientation-Location',    # Task: User, orientation, and location synchronization recognition

    Four-task combined modality recognition:
    'Four-task' # Task: Gesture, user, orientation, and location synchronization recognition
    
    Gait recognition:
    'Gait'       # Task: Gait recognition
    'Track'      # Task: Track recognition
    'Gait-Track' # Task: Gait and Track synchronization recognition
'''

class load_data:
    def __init__(self,
                 data_path,
                 gesture_cats,
                 user_cats,
                 orientation_cats,
                 location_cats,
                 t_max,
                 test_set_ratio,
                 domain,
                 task
                 ):
        self.data_path = data_path
        self.gesture_cats = gesture_cats
        self.user_cats = user_cats
        self.orientation_cats = orientation_cats
        self.location_cats = location_cats
        self.t_max = t_max
        self.test_set_ratio = test_set_ratio
        self.domain = domain if domain is not None else random.randint(1, 5)
        self.task = task
        self.task_map = {
            'In-domain-Gesture': self.In_domain_Gesture,
            'In-domain-User': self.In_domain_User,
            'In-domain-Gesture-User': self.In_domain_Gesture_User,
            'Cross-orientation-Gesture': self.Cross_orientation_Gesture,
            'Cross-location-Gesture': self.Cross_location_Gesture,
            'Cross-orientation-User': self.Cross_orientation_User,
            'Cross-location-User': self.Cross_location_User,
            'Cross-orientation-Gesture-User': self.Cross_orientation_Gesture_User,
            'Cross-location-Gesture-User': self.Cross_location_Gesture_User,
            'Orientation': self.Orientation,
            'Location': self.Location,
            'Gesture-Orientation': self.Gesture_Orientation,
            'Gesture-Location': self.Gesture_Location,
            'User-Orientation': self.User_Orientation,
            'User-Location': self.User_Location,
            'Orientation-Location': self.Orientation_Location,
            'Gesture-User-Location': self.Gesture_User_Location,
            'Gesture-User-Orientation': self.Gesture_User_Orientation,
            'Gesture-Orientation-Location': self.Gesture_Orientation_Location,
            'User-Orientation-Location': self.User_Orientation_Location,
            'Four-task': self.Four_task
        }

    def get_loader(self):
        if self.task not in self.task_map:
            raise ValueError(f"Unsupported task: {self.task}")
        loader, T_max, Tasks = self.task_map[self.task]()
        return loader, T_max, Tasks, self.get_n_class(Tasks)

    def get_n_class(self, Tasks):
        n_class = []
        if 'Gesture' in Tasks:
            n_class.append(self.gesture_cats)
        if 'User' in Tasks:
            n_class.append(self.user_cats)
        if 'Orientation' in Tasks:
            n_class.append(self.orientation_cats)
        if 'Location' in Tasks:
            n_class.append(self.location_cats)
        return n_class

    def extract_data(self):
        paths = []
        for root, _, names in os.walk(self.data_path):
            for name in tqdm(names, desc="Loading files", ncols=100):
                path = os.path.join(root, name)
                try:
                    data_ = scio.loadmat(path)['ADP']
                    if data_.shape[0:2] != (20, 20):
                        continue
                    self.t_max = max(self.t_max, data_.shape[2])
                    paths.append(path)
                except Exception:
                    continue
        (data,
         gesture_label,
         user_label,
         orientation_label,
         location_label) = [], [], [], [], []
        for idx, path in tqdm(enumerate(paths),
                              total=len(paths),
                              desc="Processing data",
                              ncols=100):
            try:
                data_ = scio.loadmat(path)['ADP']
                label_parts = os.path.basename(path).split('-')
                user_label_ = int(re.sub(r'\D', '', label_parts[0]))
                gesture_label_ = int(re.sub(r'\D', '', label_parts[1]))
                location_label_ = int(re.sub(r'\D', '', label_parts[2]))
                orientation_label_ = int(re.sub(r'\D', '', label_parts[3]))
                if (user_label_ not in self.user_cats or
                    gesture_label_ not in self.gesture_cats or
                    location_label_ not in self.location_cats or
                    orientation_label_ not in self.orientation_cats):
                    continue
                t = data_.shape[-2]
                if t >= self.t_max:
                    data_pad = data_[..., :self.t_max]
                else:
                    pad_width = [(0, 0)] * data_.ndim
                    pad_width[-2] = (self.t_max - t, 0)
                    data_pad = np.pad(data_, pad_width, mode='constant', constant_values=0)
                data.append(data_pad)
                gesture_label.append(gesture_label_)
                user_label.append(user_label_)
                orientation_label.append(orientation_label_)
                location_label.append(location_label_)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
        return (np.swapaxes(np.swapaxes(np.array(data), 1, 3), 2, 3), # (@, T_MAX, 20_1, 20_2, 2)
                np.array(gesture_label),
                np.array(user_label),
                np.array(orientation_label),
                np.array(location_label),
                self.t_max)

    def single_task_data(self):
        paths = []
        for root, _, names in os.walk(self.data_path):
            for name in names:
                path = os.path.join(root, name)
                try:
                    data_ = scio.loadmat(path)['ADP']
                    if data_.shape[0:2] != (20, 20):
                        continue
                    self.t_max = max(self.t_max, data_.shape[2])
                    paths.append(path)
                except Exception:
                    continue
        valid_paths = []
        for path in paths:
            try:
                label_parts = os.path.basename(path).split('-')
                user_label_ = int(re.sub(r'\D', '', label_parts[0]))
                if user_label_ == 1:
                    valid_paths.append(path)
            except Exception:
                continue
        # print(np.array(valid_paths).shape)
        (data,
         gesture_label,
         orientation_label,
         location_label) = [], [], [], []
        for idx, path in tqdm(enumerate(valid_paths),
                              total=len(valid_paths),
                              desc="Processing data",
                              ncols=100):
            try:
                data_ = scio.loadmat(path)['ADP']
                label_parts = os.path.basename(path).split('-')
                gesture_label_ = int(re.sub(r'\D', '', label_parts[1]))
                location_label_ = int(re.sub(r'\D', '', label_parts[2]))
                orientation_label_ = int(re.sub(r'\D', '', label_parts[3]))
                if (gesture_label_ not in self.gesture_cats or
                        location_label_ not in self.location_cats or
                        orientation_label_ not in self.orientation_cats):
                    continue
                t = data_.shape[-2]
                if t >= self.t_max:
                    data_pad = data_[..., :self.t_max]
                else:
                    pad_width = [(0, 0)] * data_.ndim
                    pad_width[-2] = (self.t_max - t, 0)
                    data_pad = np.pad(data_, pad_width, mode='constant', constant_values=0)
                data.append(data_pad)
                gesture_label.append(gesture_label_)
                orientation_label.append(orientation_label_)
                location_label.append(location_label_)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
        if len(data) == 0:
            raise ValueError("No samples found with user_label == 1")
        return (np.swapaxes(np.swapaxes(np.array(data), 1, 3), 2, 3), # (@, T_MAX, 20_1, 20_2, 2)
                np.array(gesture_label),
                np.array(orientation_label),
                np.array(location_label),
                self.t_max)

    def In_domain_Gesture(self):
        data, label, _, _, T_max = self.single_task_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['Gesture']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Gesture_label_train': label_train,
            'Gesture_label_test': label_test
        }, T_max, Tasks

    def In_domain_User(self):
        data, _, label, _, _, T_max = self.extract_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['User']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'User_label_train': label_train,
            'User_label_test': label_test
        }, T_max, Tasks

    def In_domain_Gesture_User(self):
        data, gesture_label, user_label, _, _, T_max = self.extract_data()
        [data_train,
         data_test,
         gesture_label_train,
         gesture_label_test,
         user_label_train,
         user_label_test] = train_test_split(data,
                                             gesture_label,
                                             user_label,
                                             test_size=self.test_set_ratio)
        Tasks = ['Gesture', 'User']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Gesture_label_train': gesture_label_train,
            'Gesture_label_test': gesture_label_test,
            'User_label_train': user_label_train,
            'User_label_test': user_label_test
        }, T_max, Tasks

    def Cross_orientation_Gesture(self):
        data, gesture_label, orientation_label, _, T_max = self.single_task_data()
        test_mask = orientation_label == self.domain
        Tasks = ['Gesture']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'Gesture_label_train': gesture_label[~test_mask],
            'Gesture_label_test': gesture_label[test_mask]
        }, T_max, Tasks

    def Cross_location_Gesture(self):
        data, gesture_label, _, location_label, T_max = self.single_task_data()
        test_mask = location_label == self.domain
        Tasks = ['Gesture']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'Gesture_label_train': gesture_label[~test_mask],
            'Gesture_label_test': gesture_label[test_mask]
        }, T_max,  Tasks

    def Cross_orientation_User(self):
        data, _, user_label, orientation_label, _, T_max = self.extract_data()
        test_mask = orientation_label == self.domain
        Tasks = ['User']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'User_label_train': user_label[~test_mask],
            'User_label_test': user_label[test_mask]
        }, T_max, Tasks

    def Cross_location_User(self):
        data, _, user_label, _, location_label, T_max = self.extract_data()
        test_mask = location_label == self.domain
        Tasks = ['User']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'User_label_train': user_label[~test_mask],
            'User_label_test': user_label[test_mask]
        }, T_max, Tasks

    def Cross_orientation_Gesture_User(self):
        data, gesture_label, user_label, orientation_label, _, T_max = self.extract_data()
        test_mask = orientation_label == self.domain
        Tasks = ['Gesture', 'User']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'Gesture_label_train': gesture_label[~test_mask],
            'Gesture_label_test': gesture_label[test_mask],
            'User_label_train': user_label[~test_mask],
            'User_label_test': user_label[test_mask]
        }, T_max, Tasks

    def Cross_location_Gesture_User(self):
        data, gesture_label, user_label, _, location_label, T_max = self.extract_data()
        test_mask = location_label == self.domain
        Tasks = ['Gesture', 'User']
        return {
            'data_train': data[~test_mask],
            'data_test': data[test_mask],
            'Gesture_label_train': gesture_label[~test_mask],
            'Gesture_label_test': gesture_label[test_mask],
            'User_label_train': user_label[~test_mask],
            'User_label_test': user_label[test_mask]
        }, T_max, Tasks

    def Orientation(self):
        data, _, label, _, T_max = self.single_task_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['Orientation']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Orientation_label_train': label_train,
            'Orientation_label_test': label_test
        }, T_max, Tasks

    def Location(self):
        data, _, _, label, T_max = self.single_task_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['Location']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Location_label_train': label_train,
            'Location_label_test': label_test
        }, T_max, Tasks

    def Four_task(self):
        (data,
         gesture_label,
         user_label,
         orientation_label,
         location_label,
         T_max) = self.extract_data()
        [data_train,
         data_test,
         gesture_label_train,
         gesture_label_test,
         user_label_train,
         user_label_test,
         orientation_label_train,
         orientation_label_test,
         location_label_train,
         location_label_test] = train_test_split(data,
                                                 gesture_label,
                                                 user_label,
                                                 orientation_label,
                                                 location_label,
                                                 test_size=self.test_set_ratio)
        Tasks = ['Gesture', 'User', 'Orientation', 'Location']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Gesture_label_train': gesture_label_train,
            'Gesture_label_test': gesture_label_test,
            'User_label_train': user_label_train,
            'User_label_test': user_label_test,
            'Orientation_label_train': orientation_label_train,
            'Orientation_label_test': orientation_label_test,
            'Location_label_train': location_label_train,
            'Location_label_test': location_label_test
        }, T_max, Tasks

    def Gesture_Orientation(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Gesture', 'Orientation']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Gesture_label_train': task_data['Gesture_label_train'],
            'Gesture_label_test': task_data['Gesture_label_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test']
        }, T_max, Tasks

    def Gesture_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Gesture', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Gesture_label_train': task_data['Gesture_label_train'],
            'Gesture_label_test': task_data['Gesture_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

    def User_Orientation(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['User', 'Orientation']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'User_label_train': task_data['User_label_train'],
            'User_label_test': task_data['User_label_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test']
        }, T_max, Tasks

    def User_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['User', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'User_label_train': task_data['User_label_train'],
            'User_label_test': task_data['User_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

    def Orientation_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Orientation', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

    def Gesture_User_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Gesture', 'User', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Gesture_label_train': task_data['Gesture_label_train'],
            'Gesture_label_test': task_data['Gesture_label_test'],
            'User_label_train': task_data['User_label_train'],
            'User_label_test': task_data['User_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

    def Gesture_User_Orientation(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Gesture', 'User', 'Orientation']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Gesture_label_train': task_data['Gesture_label_train'],
            'Gesture_label_test': task_data['Gesture_label_test'],
            'User_label_train': task_data['User_label_train'],
            'User_label_test': task_data['User_label_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test']
        }, T_max, Tasks

    def Gesture_Orientation_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['Gesture', 'Orientation', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'Gesture_label_train': task_data['Gesture_label_train'],
            'Gesture_label_test': task_data['Gesture_label_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

    def User_Orientation_Location(self):
        task_data, T_max, _ = self.Four_task()
        Tasks = ['User', 'Orientation', 'Location']
        return {
            'data_train': task_data['data_train'],
            'data_test': task_data['data_test'],
            'User_label_train': task_data['User_label_train'],
            'User_label_test': task_data['User_label_test'],
            'Orientation_label_train': task_data['Orientation_label_train'],
            'Orientation_label_test': task_data['Orientation_label_test'],
            'Location_label_train': task_data['Location_label_train'],
            'Location_label_test': task_data['Location_label_test']
        }, T_max, Tasks

class load_Gait_data:
    def __init__(self,
                 data_path,
                 user_cats,
                 track_cats,
                 t_max,
                 test_set_ratio,
                 task
                 ):
        self.data_path = data_path
        self.user_cats = user_cats
        self.track_cats = track_cats
        self.t_max = t_max
        self.test_set_ratio = test_set_ratio
        self.task = task
        self.task_map = {
            'Gait': self.Gait,
            'Track': self.Track,
            'Gait-Track': self.Gait_Track
        }

    def get_loader(self):
        if self.task not in self.task_map:
            raise ValueError(f"Unsupported task: {self.task}")
        loader, T_max, Tasks = self.task_map[self.task]()
        return loader, T_max, Tasks, self.get_n_class(Tasks)

    def get_n_class(self, Tasks):
        n_class = []
        if 'Gait' in Tasks:
            n_class.append(self.user_cats)
        if 'Track' in Tasks:
            n_class.append(self.track_cats)
        return n_class

    def extract_data(self):
        paths = []
        for root, _, names in os.walk(self.data_path):
            for name in tqdm(names, desc="Loading files", ncols=100):
                path = os.path.join(root, name)
                try:
                    data_ = scio.loadmat(path)['ADP']
                    if data_.shape[0:2] != (20, 20):
                        continue
                    self.t_max = max(self.t_max, data_.shape[2])
                    paths.append(path)
                except Exception:
                    continue
        (data,
         user_label,
         track_label) = [], [], []
        for idx, path in tqdm(enumerate(paths),
                              total=len(paths),
                              desc="Processing data",
                              ncols=100):
            try:
                data_ = scio.loadmat(path)['ADP']
                label_parts = os.path.basename(path).split('-')
                user_label_ = int(re.sub(r'\D', '', label_parts[0]))
                track_label_ = int(re.sub(r'\D', '', label_parts[1]))
                if (user_label_ not in self.user_cats or
                    track_label_ not in self.track_cats):
                    continue
                t = data_.shape[-2]
                if t >= self.t_max:
                    data_pad = data_[..., :self.t_max]
                else:
                    pad_width = [(0, 0)] * data_.ndim
                    pad_width[-2] = (self.t_max - t, 0)
                    data_pad = np.pad(data_, pad_width, mode='constant', constant_values=0)
                data.append(data_pad)
                user_label.append(user_label_)
                track_label.append(track_label_)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
        return (np.swapaxes(np.swapaxes(np.array(data), 1, 3), 2, 3), # (@, T_MAX, 20_1, 20_2, 2)
                np.array(user_label),
                np.array(track_label),
                self.t_max)

    def Gait(self):
        data, label, _, T_max = self.extract_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['Gait']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Gait_label_train': label_train,
            'Gait_label_test': label_test
        }, T_max, Tasks

    def Track(self):
        data, _, label, T_max = self.extract_data()
        [data_train,
         data_test,
         label_train,
         label_test] = train_test_split(data,
                                        label,
                                        test_size=self.test_set_ratio)
        Tasks = ['Track']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Track_label_train': label_train,
            'Track_label_test': label_test
        }, T_max, Tasks

    def Gait_Track(self):
        data, user_label, track_label, T_max = self.extract_data()
        [data_train,
         data_test,
         user_label_train,
         user_label_test,
         track_label_train,
         track_label_test] = train_test_split(data,
                                             user_label,
                                             track_label,
                                             test_size=self.test_set_ratio)
        Tasks = ['Gait', 'Track']
        return {
            'data_train': data_train,
            'data_test': data_test,
            'Gait_label_train': user_label_train,
            'Gait_label_test': user_label_test,
            'Track_label_train': track_label_train,
            'Track_label_test': track_label_test
        }, T_max, Tasks

# You can write these two functions in a new .py file,
# but make sure to import them in main.py!
def onehot_encoding(label, n_cats):
    return np.eye(n_cats)[np.array(label).astype('int32') - 1]

def Confusion_matrix(pred_label, label_test, task):
    pred_label_ = np.argmax(pred_label, axis=-1) + 1
    true_label_ = np.argmax(label_test[f'{task}_label_test'], axis=-1) + 1
    conf_matrix_ = confusion_matrix(true_label_, pred_label_)
    conf_matrix_sum = conf_matrix_.sum(axis=1)[:, np.newaxis]
    conf_matrix = np.divide(conf_matrix_.astype('float'),
                            conf_matrix_sum,
                            where=conf_matrix_sum != 0)
    print(f"Confusion Matrix for task {task}:\n" + "\n".join(map(str, conf_matrix)))
    mean_accuracy = np.mean(np.diag(conf_matrix))
    print(f"Mean Accuracy for task {task}: {mean_accuracy}")
    return mean_accuracy