<h1 align="center">D-Sense: Expanding Gesture Recognition via Wi-Fi</h1>
This code repository provides a basic implementation of D-Sense.

## Cite the Paper
*Z. Wang, Y. Liu and Z. Tao, "D-Sense: Expanding Gesture Recognition via Wi-Fi," in IEEE Transactions on Mobile Computing, doi: 10.1109/TMC.2026.3709191.*

## Introduction
D-Sense is a general-purpose wireless sensing system based on Wi-Fi signals that supports multiple wireless sensing tasks. By leveraging Channel State Information (CSI) from gesture data, D-Sense enables both in-domain and cross-domain gesture recognition, in-domain and cross-domain user authentication, orientation recognition, and localization. Using CSI data from gait signals, it further supports user authentication and trajectory recognition. In addition, we conduct several extended experiments on D-Sense.

In this repository, we release the code for extracting the Absolute Distance Profile (ADP) (```/ADP_Estimates``` folder) and the sensing and recognition models (```/D-SenseModel``` folder). The following sections provide a description of this codebase.

<p align="center">
<strong>Overall architecture of the D-Sense system.</strong><br>
<img src="Image/Fig.1.jpg" width="800"/>
</p>

## Preparations
This section introduces the requirements for running this codebase.

### Preparations for ADP Estimation
We recommend using MATLAB R2023b or later to extract ADP. The procedure is as follows:

- Download the ```/ADP_Estimates``` directory from this repository.
- Launch MATLAB and set the working directory to ```/ADP_Estimates```.
- Add ```/ADP_Estimates/CSI_to_DFS``` and ```/ADP_Estimates/generate_ADP``` to the MATLAB path using the "Add Folder and Subfolders" function.
### Preparations for D-Sense Model Training

#### Hardware
- We recommend using an NVIDIA RTX 4060 (8GB) or higher GPU.
- A minimum of 32 GB RAM is recommended.

#### Software
- The D-SenseModel is trained using TensorFlow 2.18.0. Since TensorFlow ≥ 2.11.0 does not support native GPU acceleration on Windows, we recommend using WSL2 to set up an Ubuntu 22.04 LTS environment on Windows 11, where GPU acceleration can be properly enabled. If you are using a server with a native Linux system, this requirement can be ignored.

#### Environment Setup
**1. Create a Conda environment named D-Sense based on Python 3.10 and activate it**
```bash
conda create -n D-Sense python=3.10 -y
conda activate D-Sense
```

**2. Install TensorFlow**
```bash
pip install --upgrade pip
pip install tensorflow==2.18.0
```

**3. Install CUDA**

Install CUDA 12.5 from the official NVIDIA [website](https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local), or install it via the following commands:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

**4. Install additional dependencies**
```bash
pip install numpy scipy pandas scikit-learn tqdm
```

> ❗ Ensure that the ```/D-SenseModel/D_Sense_DNN``` folder is located at the same level as ```/D-SenseModel/main.py```.

## ADP Estimation
ADP estimation is implemented in MATLAB and executed on the CPU. In our experiments, we use an Intel i9-14900HX processor. Before running ```/ADP_Estimates/ADP_main.m```, the following parameters need to be configured:
```matlab
params = {
    1;                    % Doppler frequency resolution.
    100;                  % DFS time dimension sampling (to reduce computational power consumption).
    'N=sigma';            % Spatial representation rule ('N=sigma' or 'delta').
    [20, 20];             % Dimensions of the ADP.
    '1~9';                % Area index (see the paper).
    'CSI\';               % CSI save path.
    'DFS\';               % DFS save path.
    'ADP\';               % ADP save path.
    'D\';                 % D save path.
    -60;                  % Minimum Doppler frequency.
    60;                   % Maximum Doppler frequency.
    'Gesture'             % Task ('Gesture' or 'Gait').
};
```
<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Params</th>
      <th align="center">Description</th>
      <th align="center">Example Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">params{1}</td>
      <td align="center">Doppler Frequency Resolution</td>
      <td align="center"><code>1</code></td>
    </tr>
    <tr>
      <td align="center">params{2}</td>
      <td align="center">DFS Time Dimension Sampling</td>
      <td align="center"><code>100</code></td>
    </tr>
    <tr>
      <td align="center">params{3}</td>
      <td align="center">Spatial Representation Rule</td>
      <td align="center"><code>N=0.01(sigma)</code> / <code>N=delta</code></td>
    </tr>
    <tr>
      <td align="center">params{4}</td>
      <td align="center">ADP Dimensions</td>
      <td align="center"><code>[20, 20]</code></td>
    </tr>
    <tr>
      <td align="center">params{5}</td>
      <td align="center">Area Index</td>
      <td align="center"><code>1~9</code></td>
    </tr>
    <tr>
      <td align="center">params{6}</td>
      <td align="center">CSI Save Path</td>
      <td align="center"><code>CSI\</code></td>
    </tr>
    <tr>
      <td align="center">params{7}</td>
      <td align="center">DFS Save Path</td>
      <td align="center"><code>DFS\</code></td>
    </tr>
    <tr>
      <td align="center">params{8}</td>
      <td align="center">ADP Save Path</td>
      <td align="center"><code>ADP\</code></td>
    </tr>
    <tr>
      <td align="center">params{9}</td>
      <td align="center">D Save Path</td>
      <td align="center"><code>D\</code></td>
    </tr>
    <tr>
      <td align="center">params{10}</td>
      <td align="center">Minimum Doppler Frequency</td>
      <td align="center"><code>-60</code></td>
    </tr>
    <tr>
      <td align="center">params{11}</td>
      <td align="center">Maximum Doppler Frequency</td>
      <td align="center"><code>60</code></td>
    </tr>
    <tr>
      <td align="center">params{12}</td>
      <td align="center">Task</td>
      <td align="center"><code>Gesture</code> / <code>Gait</code></td>
    </tr>
  </tbody>
</table>

</div>

After configuring the parameters, run ```/ADP_Estimates/ADP_main.m```. Upon completion, the Doppler frequency data for each CSI sample will be saved in the ```DFS/``` directory, the generated ADP data will be saved in the ```ADP/``` directory, and the corresponding Frequency-Distance Translation Tensor for the selected area index will be saved in the ```D/``` directory.

> ❗ The Doppler Frequency Shift (DFS) is generated implicitly. Before the ADP for the selected region is fully generated, DFS data cannot be accessed or visualized during runtime. Once all ADP data have been generated, the DFS results can be viewed without any restriction. If early access to DFS is required, please refer to the corresponding DFS data in the [Widar3.0 dataset](https://tns.thss.tsinghua.edu.cn/widar3.0/).

Each ```.mat``` file in the ```ADP/``` directory is saved as a 3-D tensor after applying the configured scaling and sampling settings. Taking params{4} = [120, 120] as an example, and without downsampling, the following configuration allows clear visualization of the dynamic variations in ADP power:

<div align="center">

<table border="1">

  <tr>
    <th align="center">MATLAB Tools (implay)</th>
    <th align="center">Recommended Parameters</th>
    <th align="center">Visualization Results</th>
  </tr>

  <tr>
    <td align="center">Magnification</td>
    <td align="center"><code>800%</code></td>
    <td align="center" rowspan="3">
      <img src="Image/Fig.2.gif" width="240"><br>
    </td>
  </tr>

  <tr>
    <td align="center">Color Map</td>
    <td align="center"><code>parula(256)</code></td>
  </tr>

  <tr>
    <td align="center">Frame Rate</td>
    <td align="center"><code>20 fps</code></td>
  </tr>

</table>

</div>

The physical interpretation of the power distribution can be found in our [paper](https://doi.org/10.1109/TMC.2026.3709191):

<p align="center">
<img src="Image/Fig.3.jpg" width="800"/>
</p>

## D-Sense Model
D-Sense model is built based on the TensorFlow framework. All experiments are conducted on Ubuntu 22.04 LTS with an NVIDIA RTX 4060 GPU for training and evaluation. Before running ```/D-SenseModel/main.py```, the following parameters should be properly configured:

```python
test_set_ratio   = 0.2
ADP_dir          = '/ADP'
domains          = [1, 2, 3, 4, 5, 6]
domain_idx       = domains[1]
gesture_cats     = [1, 2, 3, 4, 5, 6]
user_cats        = [1, 2, 3]
gait_cats        = [1, 2, 3, 4, 5, 6, 7]
orientation_cats = [1, 2, 3, 4, 5]
track_cats       = [1, 2, 3, 4]
location_cats    = [1, 2, 3, 4, 5]
N_gesture        = len(gesture_cats)
N_user           = len(user_cats)
N_gait           = len(gait_cats)
N_orientation    = len(orientation_cats)
N_track          = len(track_cats)
N_location       = len(location_cats)
t_max            = 0
n_epochs         = 100
dropout_ratio    = 0.5
N_RNN, RNN_Type  = 128, 'GRU'
n_batch          = 32
learning_rate    = 0.001
use_DWM          = True
task_, model_    = list(tasks.items())[-1], models[0]
```

<div align="center">

<table>
  <tr>
    <th align="center">Parameter</th>
    <th align="center">Description</th>
    <th align="center">Example Value</th>
  </tr>

  <tr>
    <td align="center">test_set_ratio</td>
    <td align="center">Ratio of Test Split</td>
    <td align="center"><code>0.2</code></td>
  </tr>

  <tr>
    <td align="center">ADP_dir</td>
    <td align="center">Directory of ADP Dataset</td>
    <td align="center"><code>/ADP</code></td>
  </tr>

  <tr>
    <td align="center">domains</td>
    <td align="center">Available Domain Indices</td>
    <td align="center"><code>[1,2,3,4,5,6]</code></td>
  </tr>

  <tr>
    <td align="center">domain_idx</td>
    <td align="center">Selected Domain Index</td>
    <td align="center"><code>domains[1]</code></td>
  </tr>

  <tr>
    <td align="center">gesture_cats</td>
    <td align="center">Gesture Categories</td>
    <td align="center"><code>[1,2,3,4,5,6]</code></td>
  </tr>

  <tr>
    <td align="center">user_cats</td>
    <td align="center">User Categories</td>
    <td align="center"><code>[1,2,3]</code></td>
  </tr>

  <tr>
    <td align="center">gait_cats</td>
    <td align="center">Gait Categories</td>
    <td align="center"><code>[1,2,3,4,5,6,7]</code></td>
  </tr>

  <tr>
    <td align="center">orientation_cats</td>
    <td align="center">Orientation Categories</td>
    <td align="center"><code>[1,2,3,4,5]</code></td>
  </tr>

  <tr>
    <td align="center">track_cats</td>
    <td align="center">Tracking Categories</td>
    <td align="center"><code>[1,2,3,4]</code></td>
  </tr>

  <tr>
    <td align="center">location_cats</td>
    <td align="center">Location Categories</td>
    <td align="center"><code>[1,2,3,4,5]</code></td>
  </tr>

  <tr>
    <td align="center">n_epochs</td>
    <td align="center">Training Epochs</td>
    <td align="center"><code>100</code></td>
  </tr>

  <tr>
    <td align="center">dropout_ratio</td>
    <td align="center">Dropout Ratio</td>
    <td align="center"><code>0.5</code></td>
  </tr>

  <tr>
    <td align="center">RNN</td>
    <td align="center">Hidden Size / Type</td>
    <td align="center"><code>128 / GRU</code></td>
  </tr>

  <tr>
    <td align="center">batch_size</td>
    <td align="center">Batch Size</td>
    <td align="center"><code>32</code></td>
  </tr>

  <tr>
    <td align="center">learning_rate</td>
    <td align="center">Learning Rate</td>
    <td align="center"><code>0.001</code></td>
  </tr>

  <tr>
    <td align="center">use_DWM</td>
    <td align="center">Enable DWM Module</td>
    <td align="center"><code>True</code></td>
  </tr>

  <tr>
    <td align="center">task_</td>
    <td align="center">Task Selection</td>
    <td align="center"><code>list(tasks.items())[-1]</code></td>
  </tr>

  <tr>
    <td align="center">model_</td>
    <td align="center">Model Selection</td>
    <td align="center"><code>models[0]</code></td>
  </tr>
  
</table>

</div>

The supported models, tasks, and their corresponding indices are summarized in the table below:

<div align="center">

<table border="1" style="border-collapse:collapse;">

  <tr>
    <th>Model Index</th>
    <th>Model Name</th>
  </tr>

  <tr><td align="center">M = 0</td><td align="center">CNN-RNN</td></tr>
  <tr><td align="center">M = 1</td><td align="center">Transformer</td></tr>
  <tr><td align="center">M = 2</td><td align="center">CNN-Transformer</td></tr>
  <tr><td align="center">M = 3</td><td align="center">RNN-Transformer</td></tr>
  <tr><td align="center">M = 4</td><td align="center">CNN-RNN-Transformer</td></tr>
  <tr><td align="center">M = 5</td><td align="center">CNN</td></tr>
  <tr><td align="center">M = 6</td><td align="center">RNN</td></tr>

  <tr>
    <th>Task Index</th>
    <th>Description</th>
  </tr>
  
  <tr><td align="center">T = 0</td><td align="center">In-Domain Gesture Recognition</td></tr>
  <tr><td align="center">T = 1</td><td align="center">In-Domain User Recognition</td></tr>
  <tr><td align="center">T = 2</td><td align="center">In-Domain Gesture and User Synchronized Recognition</td></tr>
  <tr><td align="center">T = 3</td><td align="center">Cross-Orientation Gesture Recognition</td></tr>
  <tr><td align="center">T = 4</td><td align="center">Cross-Location Gesture Recognition</td></tr>
  <tr><td align="center">T = 5</td><td align="center">Cross-Orientation User Recognition</td></tr>
  <tr><td align="center">T = 6</td><td align="center">Cross-Location User Recognition</td></tr>
  <tr><td align="center">T = 7</td><td align="center">Cross-Orientation Gesture and User Synchronized Recognition</td></tr>
  <tr><td align="center">T = 8</td><td align="center">Cross-Location Gesture and User Synchronized Recognition</td></tr>
  <tr><td align="center">T = 9</td><td align="center">Orientation Recognition</td></tr>
  <tr><td align="center">T = 10</td><td align="center">Location Recognition</td></tr>
  <tr><td align="center">T = 11</td><td align="center">Gesture and Orientation Synchronized Recognition</td></tr>
  <tr><td align="center">T = 12</td><td align="center">Gesture and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 13</td><td align="center">User and Orientation Synchronized Recognition</td></tr>
  <tr><td align="center">T = 14</td><td align="center">User and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 15</td><td align="center">Orientation and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 16</td><td align="center">Gesture, User and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 17</td><td align="center">Gesture, User and Orientation Synchronized Recognition</td></tr>
  <tr><td align="center">T = 18</td><td align="center">Gesture, Orientation and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 19</td><td align="center">User, Orientation and Location Synchronized Recognition</td></tr>
  <tr><td align="center">T = 20</td><td align="center">Four-Task Synchronized Recognition</td></tr>
  <tr><td align="center">T = 21</td><td align="center">Gait Recognition</td></tr>
  <tr><td align="center">T = 22</td><td align="center">Track Recognition</td></tr>
  <tr><td align="center">T = 23</td><td align="center">Gait and Track Synchronized Recognition</td></tr>
  <tr><td align="center">T = 24</td><td align="center">Cross-Location Orientation Recognition</td></tr>
  <tr><td align="center">T = 25</td><td align="center">Cross-User Orientation Recognition</td></tr>
  <tr><td align="center">T = 26</td><td align="center">Cross-Gesture Orientation Recognition</td></tr>
  <tr><td align="center">T = 27</td><td align="center">Cross-Orientation Location Recognition</td></tr>
  <tr><td align="center">T = 28</td><td align="center">Cross-Gesture Location Recognition</td></tr>
  <tr><td align="center">T = 29</td><td align="center">Cross-User Location Recognition</td></tr>

</table>

</div>

After adjusting or selecting the parameters and completing training, the confusion matrix of the model under the selected configuration will be displayed based on its performance on the test set.

## Partial Sensing Accuracy

### Experimental Accuracy of the D-Sense Across Different Sensing Tasks

<p align="center">
<img src="Image/Fig.4/Fig.4(legend).jpg" width="500"/>
</p>

<p align="center">
<strong>Gesture recognition.</strong><br>
<img src="Image/Fig.4/Fig.4(a).jpg" width="500"/>
</p>

<p align="center">
<strong>User identification.</strong><br>
<img src="Image/Fig.4/Fig.4(b).jpg" width="500"/>
</p>

<p align="center">
<strong>Orientation recognition.</strong><br>
<img src="Image/Fig.4/Fig.4(c).jpg" width="500"/>
</p>

<p align="center">
<strong>Localization.</strong><br>
<img src="Image/Fig.4/Fig.4(d).jpg" width="500"/>
</p>

### Cross-domain Gesture and User Synchronized Recognition

<div align="center">

<table>
  <tr>
    <th>Target Label</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
  </tr>

  <tr>
    <td align="center">Gesture</td>
    <td align="center">In-domain</td>
    <td align="center">98.33%</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td align="center">Location</td>
    <td align="center">84.83%</td>
    <td align="center">88.83%</td>
    <td align="center">85.50%</td>
    <td align="center">85.33%</td>
    <td align="center">88.83%</td>
  </tr>

  <tr>
    <td align="center">Orientation</td>
    <td align="center">61.50%</td>
    <td align="center">79.00%</td>
    <td align="center">85.17%</td>
    <td align="center">82.17%</td>
    <td align="center">75.58%</td>
  </tr>

  <tr style="height:8px;">
    <td colspan="6"></td>
  </tr>

  <tr>
    <td align="center">User</td>
    <td align="center">In-domain</td>
    <td align="center">94.67%</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td align="center">Location</td>
    <td align="center">76.33%</td>
    <td align="center">82.33%</td>
    <td align="center">83.67%</td>
    <td align="center">81.33%</td>
    <td align="center">86.33%</td>
  </tr>

  <tr>
    <td align="center">Orientation</td>
    <td align="center">68.67%</td>
    <td align="center">77.67%</td>
    <td align="center">80.67%</td>
    <td align="center">80.67%</td>
    <td align="center">80.09%</td>
  </tr>

</table>

</div>

### Synchronized Multi-Subtask Recognition

<div align="center">

<table>
  <tr>
    <th>Task</th>
    <th>Gesture</th>
    <th>User</th>
    <th>Orientation</th>
    <th>Location</th>
  </tr>

  <tr>
    <td align="center">Gesture</td>
    <td align="center">-</td>
    <td align="center">94.67%</td>
    <td align="center">95.60%</td>
    <td align="center">84.40%</td>
  </tr>

  <tr>
    <td align="center">User</td>
    <td align="center">97.33%</td>
    <td align="center">-</td>
    <td align="center">95.20%</td>
    <td align="center">82.40%</td>
  </tr>

  <tr>
    <td align="center">Orientation</td>
    <td align="center">96.67%</td>
    <td align="center">93.67%</td>
    <td align="center">-</td>
    <td align="center">83.40%</td>
  </tr>

  <tr>
    <td align="center">Location</td>
    <td align="center">96.33%</td>
    <td align="center">94.33%</td>
    <td align="center">94.20%</td>
    <td align="center">-</td>
  </tr>

  <tr style="height:8px;">
    <td colspan="6"></td>
  </tr>

  <tr>
    <td align="center">Gesture--User</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">94.40%</td>
    <td align="center">83.80%</td>
  </tr>

  <tr>
    <td align="center">Gesture--Orientation</td>
    <td align="center">-</td>
    <td align="center">93.00%</td>
    <td align="center">-</td>
    <td align="center">85.40%</td>
  </tr>

  <tr>
    <td align="center">Gesture--Location</td>
    <td align="center">-</td>
    <td align="center">93.00%</td>
    <td align="center">95.20%</td>
    <td align="center">-</td>
  </tr>

  <tr>
    <td align="center">User--Orientation</td>
    <td align="center">97.17%</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">85.20%</td>
  </tr>

  <tr>
    <td align="center">User--Location</td>
    <td align="center">96.33%</td>
    <td align="center">-</td>
    <td align="center">95.40%</td>
    <td align="center">-</td>
  </tr>

  <tr>
    <td align="center">Orientation--Location</td>
    <td align="center">96.17%</td>
    <td align="center">91.00%</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>

</table>

</div>

### Cross-domain Gesture Recognition

<p align="center">
<img src="Image/Fig.5.jpg" width="1000"/>
</p>

### Multi-Task Gait Recognition

<p align="center">
<img src="Image/Fig.7(a).jpg" width="300"/>
</p>

### Multi-Task Recognition Accuracy of Different Sensing Areas

<p align="center">
<img src="Image/Fig.8(a).jpg" width="300"/>
</p>

<div align="center">

<table>
  <tr>
    <th>Area</th>
    <th>Gesture</th>
    <th>User</th>
    <th>Ori.</th>
    <th>Loc.</th>
    <th>Gait</th>
    <th>Track</th>
  </tr>

  <tr>
    <td align="center">1</td>
    <td align="center">94.76%</td>
    <td align="center">87.64%</td>
    <td align="center">83.62%</td>
    <td align="center">70.00%</td>
    <td align="center">89.35%</td>
    <td align="center">99.46%</td>
  </tr>

  <tr>
    <td align="center">2</td>
    <td align="center">95.20%</td>
    <td align="center">88.62%</td>
    <td align="center">88.53%</td>
    <td align="center">77.69%</td>
    <td align="center">92.89%</td>
    <td align="center">99.50%</td>
  </tr>

  <tr>
    <td align="center">3</td>
    <td align="center">95.84%</td>
    <td align="center">88.62%</td>
    <td align="center">88.18%</td>
    <td align="center">76.00%</td>
    <td align="center">90.20%</td>
    <td align="center">99.49%</td>
  </tr>

  <tr>
    <td align="center">4</td>
    <td align="center">94.12%</td>
    <td align="center">90.69%</td>
    <td align="center">88.34%</td>
    <td align="center">73.93%</td>
    <td align="center">94.77%</td>
    <td align="center">99.95%</td>
  </tr>

  <tr>
    <td align="center">5</td>
    <td align="center">96.83%</td>
    <td align="center">92.45%</td>
    <td align="center">91.13%</td>
    <td align="center">83.20%</td>
    <td align="center">92.60%</td>
    <td align="center">100%</td>
  </tr>

  <tr>
    <td align="center">6</td>
    <td align="center">96.46%</td>
    <td align="center">90.93%</td>
    <td align="center">88.24%</td>
    <td align="center">79.41%</td>
    <td align="center">95.34%</td>
    <td align="center">100%</td>
  </tr>

  <tr>
    <td align="center">7</td>
    <td align="center">93.36%</td>
    <td align="center">90.52%</td>
    <td align="center">84.87%</td>
    <td align="center">75.74%</td>
    <td align="center">94.33%</td>
    <td align="center">99.67%</td>
  </tr>

  <tr>
    <td align="center">8</td>
    <td align="center">96.56%</td>
    <td align="center">90.80%</td>
    <td align="center">93.69%</td>
    <td align="center">85.04%</td>
    <td align="center">94.65%</td>
    <td align="center">100%</td>
  </tr>

</table>

</div>

### Cross-Domain Location and Orientation Recognition

<div align="center">

<table>
  <tr>
    <th>Target Label</th>
    <th>1/Push</th>
    <th>2/Sweep</th>
    <th>3/Clap</th>
    <th>4/Slide</th>
    <th>5/Z</th>
  </tr>

  <tr>
    <td align="center">CL-O</td>
    <td align="center">69.17%</td>
    <td align="center">77.17%</td>
    <td align="center">71.17%</td>
    <td align="center">70.17%</td>
    <td align="center">90.33%</td>
  </tr>

  <tr>
    <td align="center">CU-O</td>
    <td align="center">67.37%</td>
    <td align="center">61.92%</td>
    <td align="center">60.07%</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>

  <tr>
    <td align="center">CG-O</td>
    <td align="center">35.60%</td>
    <td align="center">70.80%</td>
    <td align="center">38.60%</td>
    <td align="center">28.80%</td>
    <td align="center">50.20%</td>
  </tr>

  <tr>
    <td align="center">CO-L</td>
    <td align="center">32.83%</td>
    <td align="center">35.00%</td>
    <td align="center">35.17%</td>
    <td align="center">45.00%</td>
    <td align="center">32.67%</td>
  </tr>

  <tr>
    <td align="center">CG-L</td>
    <td align="center">40.60%</td>
    <td align="center">43.40%</td>
    <td align="center">20.20%</td>
    <td align="center">35.00%</td>
    <td align="center">34.60%</td>
  </tr>

  <tr>
    <td align="center">CU-L</td>
    <td align="center">45.40%</td>
    <td align="center">40.11%</td>
    <td align="center">44.11%</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>

</table>

</div>

## Effectiveness of the DWM

The main experiments in this paper adopt a CNN-GRU network with the Dynamic Weighting Mechanism (DWM):

<p align="center">
<img src="Image/Fig.9.jpg" width="400"/>
</p>

### Multi-Task Loss Curve and DWM Weight Distribution

<div align="center">
    <img src="Image/Fig.6(a).jpg" width="210">
    <img src="Image/Fig.6(b).jpg" width="210">
    <img src="Image/Fig.7(b).jpg" width="230">
</div>

## Citation

If our work has been helpful to your research, please consider citing:

```bibtex
@ARTICLE{11592660,
  author={Wang, Zhelun and Liu, Ying and Tao, Zhiyong},
  journal={IEEE Transactions on Mobile Computing},
  title={D-Sense: Expanding Gesture Recognition via Wi-Fi},
  year={2026},
  pages={1-18},
  doi={10.1109/TMC.2026.3709191}
}
```

## Acknowledgments

We would like to thank [Widar3.0](https://doi.org/10.1109/TPAMI.2021.3105387) for its inspiration and for providing valuable support in terms of the dataset.

## Contact

If you have any questions, please feel free to contact me at [wzlpaper@126.com](mailto:wzlpaper@126.com).
