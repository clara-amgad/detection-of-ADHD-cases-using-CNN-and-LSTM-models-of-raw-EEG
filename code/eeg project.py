#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mne


# In[2]:


get_ipython().system('pip install tensorflow')


# In[3]:


import os
import glob
import mne
import numpy as np
import scipy.io as sio
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import display


# In[4]:


def convert_mat_to_fif(main_directory, input_folder_names, output_folder_names):
    
    for input_folder, output_folder in zip(input_folder_names, output_folder_names):
        input_folder_path = os.path.join(main_directory, input_folder)       
        output_directory_path = os.path.join(main_directory, output_folder)
        os.makedirs(output_directory_path, exist_ok=True)
        
        mat_files = glob.glob(os.path.join(input_folder_path, '*.mat'))
        
        ch_types = ['eeg'] * 19
        ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10',
                    'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15', 'EEG16', 'EEG17', 'EEG18', 'EEG19']
        
        for mat_file in mat_files:
            mat_contents = sio.loadmat(mat_file)
            electrode_data = np.array(mat_contents[os.path.splitext(os.path.basename(mat_file))[0]])
            data = [i for i in electrode_data.T]  # Transpose the data
    
            raw_info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
            raw = mne.io.RawArray(data, info=raw_info)
    
            output_filename = os.path.splitext(os.path.basename(mat_file))[0] + '.fif'
            output_filepath = os.path.join(output_directory_path, output_filename)
            raw.save(output_filepath, overwrite=True)
            print(f"Saved {output_filepath}")
    
    print("Processing complete!")


main_directory = r'C:\Users\AL-MOTAHEDA\Downloads\signal'

input_folder_names = ['ADHD_part2', 'ADHD_part1', 'Control_part1', 'Control_part2']

output_folder_names = ['ADHD_part2_fif', 'ADHD_part1_fif', 'Control_part1_fif', 'Control_part2_fif']

# Call the function
convert_mat_to_fif(main_directory, input_folder_names, output_folder_names)


# In[5]:


def apply_highpass_filter(raw, cutoff_freq):

    filtered_data = raw.copy()
    
    # Apply the high-pass filter
    filtered_data.filter(l_freq=cutoff_freq, h_freq=None, fir_design='firwin', phase='zero-double')
    
    return filtered_data


def apply_notch_filter(filtered_raw,fnotch):
    
    # Apply the notch filter
    filtered_raw.notch_filter(freqs=fnotch, picks=None, fir_design='firwin', phase='zero')
    return filtered_raw


def apply_bandpass_filter(filtered_raw, lowcut, highcut, fs, order=5):
    
    
    filtered_raw.filter(l_freq=lowcut, h_freq=highcut, picks=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                          n_jobs=1, method='fir', iir_params=None, phase='zero', fir_window='hamming',
                          fir_design='firwin', pad='reflect_limited', verbose=None)
    
    return filtered_raw



# In[6]:


def filters(main_directory, input_folder_names, output_folder_names):
    for input_folder, output_folder in zip(input_folder_names, output_folder_names):
        input_folder_path = os.path.join(main_directory, input_folder)
        output_directory_path = os.path.join(main_directory, output_folder)
        os.makedirs(output_directory_path, exist_ok=True)
        fif_files = glob.glob(os.path.join(input_folder_path, '*.fif'))
        label = 1 if 'ADHD' in input_folder else 0
        for fif_file in fif_files:
            raw = mne.io.read_raw_fif(fif_file, preload=True)
            cutoff_freq = 0.5
            filtered_raw = apply_highpass_filter(raw, cutoff_freq)
            
            lowcut = 2
            highcut = 50
            fs = 128
            filtered_raw = apply_bandpass_filter(filtered_raw, lowcut, highcut, fs, order=5)
            
            fnotch = 50  # Notch filter frequency ( remove 50 Hz power line noise)
            filtered_raw = apply_notch_filter(filtered_raw,fnotch)
            
            output_filename = f'label_{label}_' + os.path.basename(fif_file).replace('.fif', '_filtered.fif')
            output_filepath = os.path.join(output_directory_path, output_filename)
            filtered_raw.save(output_filepath, overwrite=True)
            print(f"Saved {output_filepath}")
    print("Processing complete!")

main_directory = r'C:\Users\AL-MOTAHEDA\Downloads\signal'

input_folder_names = ['ADHD_part2_fif', 'ADHD_part1_fif', 'Control_part1_fif', 'Control_part2_fif']

output_folder_names = ['ADHD_part2_filtered_fif', 'ADHD_part1_filtered_fif', 'Control_part1_filtered_fif', 'Control_part2_filtered_fif']

# Call the function
filters(main_directory, input_folder_names, output_folder_names)


# In[7]:


raw = mne.io.read_raw_fif(r'C:\Users\AL-MOTAHEDA\Downloads\signal\ADHD_part1_fif\v10p.fif' ,preload=True)
raw.plot(duration=20, n_channels=19, scalings=1000, color='blue')
plt.show()


# In[8]:


raw.compute_psd(fmax=64).plot(picks="data", exclude="bads")


# In[9]:


filtered=mne.io.read_raw_fif(r'C:\Users\AL-MOTAHEDA\Downloads\signal\ADHD_part1_filtered_fif\label_1_v10p_filtered.fif', preload=True)
filtered.plot(duration=30, n_channels=19, scalings=1000 , color='green')  # Plot filtered data for 5 seconds
plt.show()


# In[10]:


filtered.compute_psd(fmax=64).plot(picks="data", exclude="bads")


# In[11]:


pip install imblearn


# In[15]:


import os
import glob
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam

# Constants
SAMPLE_RATE = 512  # Hz
EPOCH_DURATION = 30  # seconds
EPOCH_LENGTH = EPOCH_DURATION * SAMPLE_RATE  # in samples

def split_data(main_directory, output_folder_names, test_size=0.3):
    labeled_files = []
    labels = []

    for output_folder in output_folder_names:
        output_folder_path = os.path.join(main_directory, output_folder)
        fif_files = glob.glob(os.path.join(output_folder_path, 'label_*.fif'))
        
        for fif_file in fif_files:
            labeled_files.append(fif_file)
            label = 1 if 'label_1' in fif_file else 0
            labels.append(label)
    
    train_files, test_files, train_labels, test_labels = train_test_split(
        labeled_files, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return train_files, test_files, train_labels, test_labels

def load_eeg_data(filenames, labels, epoch_length=EPOCH_LENGTH):
    data = []
    new_labels = []
    for file, label in zip(filenames, labels):
        raw = mne.io.read_raw_fif(file, preload=True)
        raw_data = raw.get_data()

        n_epochs = raw_data.shape[1] // epoch_length
        for i in range(n_epochs):
            start = i * epoch_length
            end = start + epoch_length
            epoch_data = raw_data[:, start:end]
            data.append(epoch_data)
            new_labels.append(label)
    
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data, np.array(new_labels)

def build_eeg_cnn(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(filters=16, kernel_size=(10, 1), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=16, kernel_size=(4, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=32, kernel_size=(1, 128), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(1, 64))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=32, kernel_size=(1, 64), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(1, 32))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

main_directory = r'C:\Users\AL-MOTAHEDA\Downloads\signal'

output_folder_names = [
    'ADHD_part2_filtered_fif', 
    'ADHD_part1_filtered_fif', 
    'Control_part1_filtered_fif', 
    'Control_part2_filtered_fif'
]

train_files, test_files, train_labels, test_labels = split_data(main_directory, output_folder_names)

X_train, y_train = load_eeg_data(train_files, train_labels, epoch_length=EPOCH_LENGTH)
X_test, y_test = load_eeg_data(test_files, test_labels, epoch_length=EPOCH_LENGTH)
# Reshape data for SMOTE
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Apply SMOTE
#to balance the class distribution and improve model performance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)

# Reshape back to original shape
X_train_res = X_train_res.reshape(X_train_res.shape[0], 19, EPOCH_LENGTH, 1)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train_res shape: {X_train_res.shape}')
print(f'y_train_res shape: {y_train_res.shape}')

input_shape = (19, EPOCH_LENGTH, 1)
model = build_eeg_cnn(input_shape)

adam_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    X_train_res, y_train_res,
    epochs=30,
    batch_size=16,  # Reduce batch size to avoid memory issues
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

precision_adhd = precision_score(y_test, y_pred, pos_label=1)
precision_control = precision_score(y_test, y_pred, pos_label=0)
recall_adhd = recall_score(y_test, y_pred, pos_label=1)
recall_control = recall_score(y_test, y_pred, pos_label=0)
f1_adhd = f1_score(y_test, y_pred, pos_label=1)
f1_control = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_pred_prob)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Control', 'ADHD'])

print(f'Precision (ADHD): {precision_adhd}')
print(f'Precision (Control): {precision_control}')
print(f'Sensitivity (Recall for ADHD):{recall_adhd}')
print(f'Specificity (Recall for Control): {recall_control}')
print(f'F1-score (ADHD): {f1_adhd}')
print(f'F1-score (Control): {f1_control}')
print(f'AUC: {auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[16]:


model.summary()


# In[17]:


print(len(train_files), len(test_files))


# In[19]:


predicted_labels = model.predict(X_test)
predicted_classes = (predicted_labels > 0.5).astype(int).flatten()

num_samples = X_test.shape[0]
num_channels = X_test.shape[1]
num_subplots = num_samples + 1  

start_index = 1000  
end_index = 2000    

plt.figure(figsize=(12, 3 * num_subplots))
for i in range(num_samples):
    plt.subplot(num_subplots, 1, i + 1)
    flattened_data = X_test[i].flatten()[start_index:end_index]
    plt.plot(flattened_data, label=f'True Label: {y_test[i]}, Predicted Label: {predicted_classes[i]}')
    plt.xlabel('Time')
    plt.ylabel('EEG Data')
    plt.title(f'Sample {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()


# In[20]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import glob

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

conf_matrix = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Control', 'ADHD'])
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[21]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import glob


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[25]:


#LSTM model 
import os
import glob
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam

# Constants
SAMPLE_RATE = 512  # Hz
EPOCH_DURATION = 30  # seconds
EPOCH_LENGTH = EPOCH_DURATION * SAMPLE_RATE  # in samples

def split_data(main_directory, output_folder_names, test_size=0.3):
    labeled_files = []
    labels = []

    for output_folder in output_folder_names:
        output_folder_path = os.path.join(main_directory, output_folder)
        fif_files = glob.glob(os.path.join(output_folder_path, 'label_*.fif'))
        
        for fif_file in fif_files:
            labeled_files.append(fif_file)
            label = 1 if 'label_1' in fif_file else 0
            labels.append(label)
    
    train_files, test_files, train_labels, test_labels = train_test_split(
        labeled_files, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return train_files, test_files, train_labels, test_labels

def load_eeg_data(filenames, labels, epoch_length=EPOCH_LENGTH):
    data = []
    new_labels = []
    for file, label in zip(filenames, labels):
        raw = mne.io.read_raw_fif(file, preload=True)
        raw_data = raw.get_data()

        n_epochs = raw_data.shape[1] // epoch_length
        for i in range(n_epochs):
            start = i * epoch_length
            end = start + epoch_length
            epoch_data = raw_data[:, start:end]
            data.append(epoch_data)
            new_labels.append(label)
    
    data = np.array(data)
    return data, np.array(new_labels)

def build_eeg_lstm(input_shape):
    inputs = Input(shape=input_shape)
    
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    lstm_model = Model(inputs=inputs, outputs=outputs)
    
    return lstm_model

main_directory = r'C:\Users\AL-MOTAHEDA\Downloads\signal'

output_folder_names = [
    'ADHD_part2_filtered_fif', 
    'ADHD_part1_filtered_fif', 
    'Control_part1_filtered_fif', 
    'Control_part2_filtered_fif'
]

train_files, test_files, train_labels, test_labels = split_data(main_directory, output_folder_names)

X_train, y_train = load_eeg_data(train_files, train_labels, epoch_length=EPOCH_LENGTH)
X_test, y_test = load_eeg_data(test_files, test_labels, epoch_length=EPOCH_LENGTH)

# Reshape data for SMOTE
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train)

# Reshape back to original shape
X_train_res = X_train_res.reshape(X_train_res.shape[0], 19, EPOCH_LENGTH)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train_res shape: {X_train_res.shape}')
print(f'y_train_res shape: {y_train_res.shape}')

input_shape = (19, EPOCH_LENGTH)
lstm_model = build_eeg_lstm(input_shape)

adam_optimizer = Adam(learning_rate=0.0001)
lstm_model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(
    X_train_res, y_train_res,
    epochs=30,
    batch_size=16,  # Reduce batch size to avoid memory issues
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

loss, accuracy = lstm_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

y_pred_prob = lstm_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

precision_adhd = precision_score(y_test, y_pred, pos_label=1)
precision_control = precision_score(y_test, y_pred, pos_label=0)
recall_adhd = recall_score(y_test, y_pred, pos_label=1)
recall_control = recall_score(y_test, y_pred, pos_label=0)
f1_adhd = f1_score(y_test, y_pred, pos_label=1)
f1_control = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_pred_prob)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Control', 'ADHD'])

print(f'Precision (ADHD): {precision_adhd}')
print(f'Precision (Control): {precision_control}')
print(f'Sensitivity (Recall for ADHD):{recall_adhd}')
print(f'Specificity (Recall for Control): {recall_control}')
print(f'F1-score (ADHD): {f1_adhd}')
print(f'F1-score (Control): {f1_control}')
print(f'AUC: {auc}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[26]:


lstm_model.summary()


# In[27]:


predicted_labels = lstm_model.predict(X_test)
predicted_classes = (predicted_labels > 0.5).astype(int).flatten()

num_samples = X_test.shape[0]
num_channels = X_test.shape[1]
num_subplots = num_samples + 1  

start_index = 1000  
end_index = 2000    

plt.figure(figsize=(12, 3 * num_subplots))
for i in range(num_samples):
    plt.subplot(num_subplots, 1, i + 1)
    flattened_data = X_test[i].flatten()[start_index:end_index]
    plt.plot(flattened_data, label=f'True Label: {y_test[i]}, Predicted Label: {predicted_classes[i]}')
    plt.xlabel('Time')
    plt.ylabel('EEG Data')
    plt.title(f'Sample {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()


# In[28]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import glob

y_pred = lstm_model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

conf_matrix = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Control', 'ADHD'])
disp.plot(cmap=plt.cm.Reds)
plt.show()


# In[29]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import glob

y_pred_prob = lstm_model.predict(X_test)
y_pred_classes = (y_pred_prob > 0.5).astype(int).flatten()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='pink', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:




