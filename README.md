# Dinov2-guided diffusion

Gonna try guiding stable-diffusion on spherical distance loss from Dinov2 embeddings.  
If you see a lot of ImageBind code it's because this repository is a fork of [imagebind-guided-diffusion](https://github.com/Birch-san/imagebind-guided-diffusion).

## Setup

Clone repository (including ImageBind submodule):

```bash
git clone https://github.com/Birch-san/dino-guided-diffusion.git
cd imagebind-guided-diffusion
```

### [Option 1] Conda env

Create a Conda environment.

```bash
conda create -n dino-guide python=3.11
conda activate dino-guide
```

### [Option 2] Virtualenv

```bash
python3.11 -m venv venv
. venv/bin/activate
pip install wheel
```

## Install dependencies

Install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

## Run:

From root of repository:

```bash
python -m scripts.guidance_play
```