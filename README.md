On Python 3.12

Install Torch w/ right CUDA version

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


Install `segment-anything` :

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install required modules :
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx wheel ninja
```

Install `GroundingDINO` :

```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

cd Grounded-Segment-Anything

pip install --no-build-isolation -e GroundingDINO
```

Also clone Grounded-SAM-2 if you want to use sam2 programs.

```
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```

Then install `segment-anything-2` :

```
cd Grounded-SAM-2
pip install -e .
```

If you are on linux, you can download GroundingDINO models with the shell script `Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh` and SAM2 models with `Grounded-SAM-2/checkpoints/download_ckpts.sh`.

Also grab a config file for GroundingDINO in Grounded-SAM-2/grounding_dino/groundingdino/config 

---

See respective READMEs on official repos for more installation details.

--- 

By default, sam2 programs will use config files present in the Grounded-SAM-2/sam2/configs directory. Hydra will look for these files in the directory where the sam2 program is run. If you want to use a different config location, you may struggle with the hydra config search path.