### project title 
Detection of ADHD cases using CNN and LTSM models.

### description
This study proposes a novel convolutional neural network (CNN) structure in conjunction with 
classical machine learning models, utilizing the raw electroencephalography (EEG) signal as the input to 
diagnose attention deficit hyperactivity disorder (ADHD) in children. The proposed EEG-based approach does 
not require transformation or artifact rejection techniques.


### installation
1. MNE:
      - A Python package for processing EEG data.
   
2. TensorFlow:
      - An open-source machine learning framework.

3. scikit-learn:
      - A machine learning library for Python.
        
4. imbalanced-learn:
      - A Python package for dealing with imbalanced datasets.
        
5. matplotlib:
      - A plotting library for Python.
        
6. os (Operating System Interface):
   - Provides a portable way of using operating system-dependent functionality, such as reading or writing files, manipulating paths, and managing directories.
   - Commonly used functions include os.path.join(), os.listdir(), os.makedirs(), os.path.exists(), etc.

7. glob:
   - Provides a Unix-style pathname pattern expansion.
   - Useful for finding files and directories whose names match a specific pattern (e.g., using wildcards like * or ?).
   - Often used in file search and manipulation tasks, such as iterating over a directory to find files with a certain extension.

8. numpy (Numerical Python):
   - Fundamental package for scientific computing in Python.
   - Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
   - Widely used in numerical computations, linear algebra, statistics, and data manipulation.

9. scipy.io:
   - Subpackage of SciPy, a library for mathematics, science, and engineering.
   - Provides functions for reading and writing data from various file formats, including MATLAB files, WAV files, and more.
   - Useful for loading and saving data in scientific research and engineering applications.

10. pandas:
   - Data manipulation and analysis library for Python.
   - Provides high-performance, easy-to-use data structures and data analysis tools.
   - Widely used for handling structured data, including reading and writing data from various file formats (e.g., CSV, Excel), data cleaning, transformation, and analysis.

11. IPython.display:
   - Part of IPython, an enhanced interactive Python shell.
   - Provides functions for displaying rich media (e.g., images, HTML, audio, video) directly in the IPython environment.
   - Useful for visualizing data, generating interactive output, and enhancing the interactive computing experience.

12. keras (now integrated into TensorFlow as tf.keras):
   - High-level neural networks API, originally developed as an independent library and later integrated into TensorFlow.
   - Provides a simple, modular interface for building and training deep learning models.
   - Allows for easy prototyping of neural networks with a focus on user-friendliness, modularity, and extensibility.

### Availability of data and material 
[The data which was used in this code ](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

### preprocessing
-The high-pass filter:  It helps in removing slow drifts and low-frequency noise such as movement artifacts and baseline wander, which are often caused by patient movements or electrode movements enhancing the clarity of the EEG signals.

-Notch Filter:Eliminate Power Line Noise ,(the power line frequency is 50 Hz can cause a significant amount of noise in the signal ) 

-Band-Pass Filter:Isolate Relevant Frequency Bands: EEG signals contain various frequency components that correspond to different types of brain activity. For example, delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), and gamma (30-100 Hz) bands are commonly analyzed in EEG studies, Focus on Specific Brain Activities.


### Models
#### (1) CNN model
##### structure
1. Input Layer:
   - Shape: (19 channels, 512 samples per channel, 1)
   - This layer receives EEG data represented as 19 channels with 512 samples each, with a depth of 1 (monochromatic).

2. Convolutional Layers:
   - Four convolutional layers are employed, each followed by batch normalization, average pooling, and dropout regularization.
   - The convolutional layers use filters of varying kernel sizes to capture different aspects of the input data.
   - Activation function: ReLU (Rectified Linear Unit) is used to introduce non-linearity.

3. Flatten Layer:
   - Converts the output of the convolutional layers into a one-dimensional vector.

4. Dense Layers:
   - Two fully connected (dense) layers are added after flattening to learn complex patterns from the extracted features.
   - Dropout regularization is applied to mitigate overfitting.
   - Activation function: ReLU is used in the first dense layer.

5. Output Layer:
   - Dense layer with a single neuron representing the binary classification output (ADHD or Control).
   - Activation function: Sigmoid is used to obtain a probability output in the range [0, 1], indicating the likelihood of belonging to a particular class.
##### training and evaluation 
(1)Training:
.Adam optimizer with a learning rate of 0.0001 is used for optimization.
.Binary cross-entropy loss function is utilized, suitable for binary classification tasks.
.The model is trained for 30 epochs with a batch size of 16, using early stopping to prevent overfitting.                                                                                                                                                                       
(2)evalution:

![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/850ec07c-a9c2-45c7-99a6-22a5f2f59658) 
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/bfa89fac-0300-4df9-b243-e14a069611be)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/8213af41-d696-4cec-9fe9-d41e513eda77)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/9aa12385-cc63-40f8-8f86-7678c72de7f3)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/ecba62ec-acf7-40e0-a99b-8bd10864eb12)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/6eebd3b4-79f4-4da7-9151-4ab970cee6fb)

#### (2) LSTM model  
##### structure
(1)Input Layer:
-Shape: (19 channels, 512 samples per channel)
-This layer receives EEG data represented as 19 channels with 512 samples each.                                        
(2)LSTM Layers:
-Two LSTM layers are employed to capture temporal dependencies and long-term patterns in the EEG signals.
-Each LSTM layer consists of 64 memory cells.
-Batch normalization is applied to stabilize and accelerate the training process.
-Dropout regularization is used to prevent overfitting by randomly dropping a fraction of the connections during training.                                                                                                                            
(3)Dense Layers:
-One fully connected (dense) layer is added after the LSTM layers to process the extracted features.
-Dropout regularization is applied to further prevent overfitting.                                                                       (4)Output Layer:
-Dense layer with a single neuron representing the binary classification output (ADHD or Control).
-Activation function: Sigmoid is used to obtain a probability output in the range [0, 1], indicating the likelihood of belonging to a particular class.
##### training and evaluation 
(1)Training:
Adam optimizer with a learning rate of 0.0001 is used for optimization.
Binary cross-entropy loss function is utilized, suitable for binary classification tasks.
The model is trained for 30 epochs with a batch size of 16, using early stopping to prevent overfitting. 
(2)evaluation:
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/a92b4c50-8970-477b-ae05-9c57eb919bad)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/5835d6e4-6e58-467f-b6c0-b4ad5a42b881)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/3c433c5c-815c-42f0-b988-a5c233db4b17)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/9132aa3b-aa92-41f0-9be8-c8b36538fc7a)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/1df6e852-6e34-4521-9437-9b3bb0b7c898)
![image](https://github.com/clara-amgad/detection-of-ADHD-cases-using-CNN-and-LSTM-models-of-raw-EEG/assets/170095494/104fb5c6-50ad-4e6f-8ee1-382efe18caca)

### reference
[paper](https://doi.org/10.1016/j.cmpbup.2022.100080)
