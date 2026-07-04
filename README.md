# D-Sense: Expanding Gesture Recognition via Wi-Fi
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

### MATLAB (```/ADP_Estimates```)
We recommend using MATLAB R2023b or newer versions.

### Python (```)
