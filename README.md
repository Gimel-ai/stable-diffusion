<aside>
💡 Original github repo [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

</aside>

<aside>
💡 My repo - [https://github.com/Gimel-ai/stable-diffusion](https://github.com/Gimel-ai/stable-diffusion)

</aside>

## Requirements:

- Ubuntu 22.04 with, Nvidia 510, cuda 11.7, docker and nvidia docker installed.
- Nvidia GPU with at least 8GB Vram that works with the FP16 version of the model, better to have 11GB or more
- RTX 2080 supper was tested, but newer cards can be used, if you present an error with pytorch version not supporting your GPU because using the 30 series, then delete and update pytorch.
- Git installed and Jupyter notebooks, make sure to use jupyter lab.
- Anaconda need to be installed

---

## Downloading the repo and installing dependencies

```bash
# run with sudo if necessary 

git clone https://github.com/Gimel-ai/stable-diffusion.git

## Install few dependencies 
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .

## 
```

## Huggin faces model download and run

```bash
pip install --upgrade diffusers transformers scipy

# Login to hf cli 
huggingface-cli login

# To get the token go to your account settings and create a token and copy it to the terminal then press enter 

# Create a notebook or .py code and paste the code below 

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Model selection 
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Load the model to the GPU for faster inference 
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# Prompt sections where we inster the output of pic we want
prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("astronaut_rides_horse.png")
```

## To remove the diffusion filter for NFSW content

```bash
# add the code below before the prompt section of the model, full example below. 
def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Model selection 
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Load the model to the GPU for faster inference 
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# Remove limitation to create content NFSW 
def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker

# Prompt sections where we inster the output of pic we want
prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("astronaut_rides_horse.png")

```

## Add this line of code if your GPU has less than 11GB Vram so the model will fit.

```bash
# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

```

# Running stable diffusion in a Docker container

```bash
sudo docker run --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --rm 4c14b66a4c09
```

Once inside the container we will use the cells on top to download the model and run the same commands to install dependencies and so on, except for Pytorch since this is installed in the container already. 

```bash
git clone https://github.com/Gimel-ai/stable-diffusion.git
cd stable-diffusion/

pip install --upgrade diffusers transformers scipy
pip install "ipywidgets>=7,<8"

# Login to hf cli 
huggingface-cli login

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

# create the .py file to run the model 

vim run.py 

# Copy the code below and save the file
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Model selection 
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Load the model to the GPU for faster inference 
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

# Remove limitation to create content NFSW 
def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker

# Prompt sections where we inster the output of pic we want
prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("astronaut_rides_horse.png")

### launch the file to create your image, change the prompt as you wish
python3 run.py 

```

# Jupyter notebook - easy way inside the container `This is the easiest way` for begginers`

```bash
# Launch the container with jupyter enable from local or remote computer

sudo docker run --gpus all -p "8888:8889" -it --ulimit memlock=-1 --ulimit stack=67108864 --rm 4c14b66a4c09

# Follow the code like on the steps below to download and configure stable diffusion inside the container
jupyter lab --port=8889 --ip=0.0.0.0 --allow-root --no-browser 

# launch a browser, either on local machine or remote like from your laptop. 
http://192.168.1.113:8888/lab 

# Navigate to the model folder and open the run.ipynb notebook and follow the steps to create images

 
```

## Dockerhub image ready to use and launch `Beginners`

```bash
# Download the docker image from my repo and launch the container

sudo docker pull 
sudo docker run --gpus all -p "8888:8889" -it --ulimit memlock=-1 --ulimit stack=67108864 --rm 4c14b66a4c09

# Lunch the jupyter notebook once inside the container
jupyter lab --port=8889 --ip=0.0.0.0 --allow-root --no-browser

# Navigate to the stablefussion folder 
# Open the run.ipynb notebook and follow the steps to start creatig images
```
