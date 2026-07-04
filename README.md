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
      <th align="center">No.</th>
      <th align="center">Description</th>
      <th align="center">Example Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td align="center">Doppler frequency resolution</td>
      <td align="center"><code>1</code></td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="center">DFS time dimension sampling</td>
      <td align="center"><code>100</code></td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="center">Spatial representation rule</td>
      <td align="center"><code>N=sigma</code> / <code>delta</code></td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td align="center">ADP dimensions</td>
      <td align="center"><code>[20, 20]</code></td>
    </tr>
    <tr>
      <td align="center">5</td>
      <td align="center">Area index</td>
      <td align="center"><code>1~9</code></td>
    </tr>
    <tr>
      <td align="center">6</td>
      <td align="center">CSI save path</td>
      <td align="center"><code>CSI\</code></td>
    </tr>
    <tr>
      <td align="center">7</td>
      <td align="center">DFS save path</td>
      <td align="center"><code>DFS\</code></td>
    </tr>
    <tr>
      <td align="center">8</td>
      <td align="center">ADP save path</td>
      <td align="center"><code>ADP\</code></td>
    </tr>
    <tr>
      <td align="center">9</td>
      <td align="center">D save path</td>
      <td align="center"><code>D\</code></td>
    </tr>
    <tr>
      <td align="center">10</td>
      <td align="center">Minimum Doppler frequency</td>
      <td align="center"><code>-60</code></td>
    </tr>
    <tr>
      <td align="center">11</td>
      <td align="center">Maximum Doppler frequency</td>
      <td align="center"><code>60</code></td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td align="center">Task</td>
      <td align="center"><code>Gesture</code> / <code>Gait</code></td>
    </tr>
  </tbody>
</table>

</div>
