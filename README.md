# SynergyNet_mindspore

***
Implemented the SynergyNet model based on mindspore.

The SynergyNet pipeline contains two stages. The first stage includes a preliminary 3DMM regression from images and a multi-attribute feature aggregation (MAFA) for landmark refinement. The second stage contains a landmark-to 3DMM regressor to reveal the embedded facial geometry in sparse landmarks.

The architectural definition of each network refers to the following papers:

[1] C. -Y. Wu, Q. Xu and U. Neumann, "Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry," 2021 International Conference on 3D Vision (3DV), 2021, pp. 453-463, doi: 10.1109/3DV53792.2021.00055.

[<a href="https://arxiv.org/abs/2110.09772">paper</a>]

## Pretrained models

***

### Facial Alignment on AFLW2000-3D (NME of facial landmarks)

The following table lists SynergyNet AFLW2000-3D checkpoints. The model verifies the accuracy
of Top-1 and Top-5.

|  | | MindSpore | MindSpore |  MindSpore |  MindSpore  | Pytorch_official | Pytorch_official | Pytorch_official |Pytorch_official || |
|:-----:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:-----------:|:-----------:|
| Model | Dataset |  [ 0, 30] | [30, 60] | [60, 90] | [ 0, 90] |[ 0, 30] | [30, 60] |[60, 90] |[ 0, 90] |Download | Config |
| SynergyNet | AFLW2000-3D | 2.656 |3.316|4.268|3.413|2.656|3.316|4.268|3.413|

### Face orientation estimation on AFLW2000-3D (MAE of Euler angles)

|  | | MindSpore | MindSpore |  MindSpore |  MindSpore  | Pytorch_official | Pytorch_official |  Pytorch_official |Pytorch_official || |
|:-----:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Model | Dataset | Yaw  | Pitch | Roll | Mean MAE |Yaw  | Pitch | Roll | Mean MAE |Download | Config |
| SynergyNet | AFLW2000-3D |3.566  |4.059 |2.539|3.388|3.566  |4.059|2.539|3.388|

## Examples

***

### Eval

- The following configuration for eval.

  ```shell
  python synergynet_aflw2000_eval.py  --root ./aflw2000_data
  ```

  output:

  ```text
  Facial Alignment on AFLW2000-3D(NME):
  [ 0, 30] Mean:2.656 Std:1.194
  [30, 60] Mean:3.316 Std:1.924
  [60, 90] Mean:4.268 Std:2.569
  [ 0, 90] Mean:3.413 Std:0.662
  Face orientation estimation:
  Mean MAE = 3.388 (in deg), [yaw,pitch,roll] = [3.566, 4.059, 2.539]
  ```
  
**Acknowledgement**

The project is developed on [<a href="https://choyingw.github.io/works/SynergyNet">SynergyNet_torch</a>]. Thank them for their wonderful work. 
