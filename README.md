<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# MotionNet

## Features

MotionNet for Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps.

## Installation
- CUDA >= 9.0
- Python 3
- pyquaternion, Matplotlib, PIL, numpy, cv2, tqdm (and other relevant packages which can be easily installed with pip or conda)
- PyTorch >= 1.1
  - Note that, the MGDA-related code currently can only run on PyTorch 1.1 (due to the official implementation of [MGDA](https://github.com/intel-isl/MultiObjectiveOptimization)). Such codes include `min_norm_solvers.py`, `train_single_seq_MGDA.py` and `train_multi_seq_MGDA.py`.

### Setup:
1. To run the code, first need to add the path to the root folder. For example:
```
export PYTHONPATH=/home/pwu/PycharmProjects/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/pwu/PycharmProjects/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
```
2. Data preparation (suppose we are now at the folder `MotionNet`):
   - Download the [nuScenes data](https://www.nuscenes.org/).
   - Run command `python data/gen_data.py --root /path/to/nuScenes/data/ --split train --savepath /path/to/the/directory/for/storing/the/preprocessed/data/`. This will generate preprocessed training data. Similarly we can prepare the validation and test data.
   - See `README.md` in the `data` folder for more details.
3. Suppose the generated preprocessed data are in folder `/data/nuScenes_preprocessed`, then:
   - To train the model trained with spatio-temporal losses: `python train_multi_seq.py --data /data/nuScenes_preprocessed --batch 8 --nepoch 45 --nworker 4 --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --log`. This command will train the model with spatio-temporal consistency losses. See the code for more details.
   - To train the model with MGDA framework: `python train_multi_seq_MGDA.py --data /data/nuScenes_preprocessed --batch 8 --nepoch 70 --nworker 4 --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --reg_weight_cls 2.0 --log`.
   - The pre-trained model for `train_multi_seq.py` can be downloaded from https://zenodo.org/record/8215963
   - The pre-trained model for `train_multi_seq_MGDA.py` can be downloaded from https://zenodo.org/record/8215963
   - The files `train_single_seq.py` and `train_single_seq_MGDA.py` train MotionNet exactly in the same manner, except without utilizing spatio-temporal consistency losses.
4. After obtaining the trained model, e.g., `model.pth` for `train_multi_seq.py`, we can evaluate the model performance as follows:
   - Run `python eval.py --data /path/to/the/generated/test/data --model model.pth --split test --log . --bs 1 --net MotionNet`. This will test the performance of MotionNet.

### Visualization
To visualize the results:
1. Generate the predicted results into .png images: run `python plots.py --data /path/to/nuScenes/data/ --version v1.0-trainval --modelpath model.pth --net MotionNet --nframe 10 --savepath logs`
2. Assemble the generated .png images into `.gif` or `.mp4`: `python plots.py --data /path/to/nuScenes/data/ --version v1.0-trainval --modelpath model.pth --net MotionNet --nframe 10 --savepath logs --video --format gif`

### Files
- `train_single_seq.py` and `train_single_seq_MGDA.py` train the model without using spatio-temporal consistency losses.
- `train_multi_seq.py` and `train_multi_seq_MGDA.py` train the model with spatio-temporal losses.
- `model.py` contains the definition of MotionNet.
- `min_norm_solvers.py` is for multi-objective optimization framework (MGDA).
- `eval.py` contains several metrics for evaluating the model performance.
- `plots.py` contains utilities for generating the predicted images/videos.
- `data/data_utils.py` includes the utility functions for preprocessing the nuScenes data.
- `data/gen_data.py` generates the preprocessed nuScenes data.
- `data/nuscenes_dataloader.py` the dataloader for model training/validation/testing.
- `nuscenes-devkit` this folder is based on the nuScenes official API, and modified to include many other utilities.

## Citation

If you use the software, please cite the following ([TR2020-068](https://merl.com/publications/TR2020-068)):

```BibTex
@inproceedings{wu2020motionnet,
  title={MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps},
  author={Wu, Pengxiang and Chen, Siheng and Metaxas, Dimitris},
  booktitle={CVPR},
  year={2020}
}
```

## Contact

Tim K Marks (<tmarks@merl.com>)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as listed below:

```
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

Files in `nuscenes-devkit` were adapted from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit):

```
Copyright (C) 2018, Oscar Beijbom, Qiang Xu, Alex Lang

SPDX-License-Identifier: CC-BY-NC-SA-4.0
```
