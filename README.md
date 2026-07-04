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
