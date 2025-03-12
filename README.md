# DISCO
This repository contains the official implementation associated with the paper: 

<b>DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Control</b>.

In European Conference on Computer Vision 2024 (ECCV 2024)

[[arXiv](https://arxiv.org/abs/2407.14758)]


## Install

We use ```python3.8``` in this work. You can setup the environment and install dependencies in ```requirements.txt``` via

```
conda create -n disco python==3.8
conda activate disco
pip install -r requirements.txt
```

## Download Data

Next, download [ALFRED DATA](https://github.com/askforalfred/alfred/tree/master/data) using the ```download_data.sh``` script. The Lite json version is ok in this repository.

```
cd data
sh download_data.sh json
```

It will finally in the following structure:

```
----data
   | ----json_2.1.0
      | ----train
      | ----valid_seen
      | ----valid_unseen
      | ----tests_seen
      | ----tests_unseen
   | ----splits
      | ----oct21.json
```

## Download checkpoints

Then, download pretrained checkpoints from [Google Drive](https://drive.google.com/drive/folders/10yMWoHfqXcHNlnV9QDOe36R26RlvQ3VC?usp=sharing). Unzip the file and put them into ```weights```


## Headless Machines

For headless machines, use the script ```startx.py``` to launch a headless render to enable the unity simulation.

```
python startx.py --gpu 0 --display [DISPLAY]
```

## Running

Now, everything is prepared if you set up correctly. You can run DISCO using the folling commands

```
python run.py --n_proc 4 --split valid_seen   --x_display [DISPLAY] --name [EXP_NAME]
python run.py --n_proc 4 --split valid_unseen --x_display [DISPLAY] --name [EXP_NAME] 
```
The ```[DISPLAY]``` number should equal to the number when you start the X server

The result will be saved into ```./logs/[EXP_NAME]```.

## Evaluation

Last, you can evaluate the performances via

```
python evaluate.py --name [EXP_NAME] --split valid_unseen
python evaluate.py --name [EXP_NAME] --split valid_seen
```

## Citation
If you find our work useful, please cite:
```
@misc{xu2024disco,
      title={DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Control}, 
      author={Xinyu Xu and Shengcheng Luo and Yanchao Yang and Yong-Lu Li and Cewu Lu},
      year={2024},
      eprint={2407.14758},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14758}, 
}
```

