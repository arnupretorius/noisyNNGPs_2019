# Code: On the expected behaviour of noise regularised neural networks as Gaussian processes.

This repository provides the code to reproduce the results in the paper: "On the expected behaviour of noise regularised neural networks as Gaussian processes."

## Basic requirements to plot Figures 1, 3 and 4

To reproduce Figures 1-4, all that is required is `numpy`, `pandas`, `seaborn` and `matplotlib`.

## Requirements to regenerate the results for Figures 1 and 3: 

To regenerate the results for Figures 1 and 3, follow the instructions given below.

### Installation

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Step 2. Pull deepo image to duplicate the research environment.

```bash
docker pull ufoym/deepo:tensorflow-py27-cu90
```
#### Step 3. Clone the research code repository. 
```bash
git clone https://github.com/arnupretorius/noisyNNGPs_2019.git
```

### Usage

Change directory to the cloned repository on your local machine and run the bash script to start the docker container with the correct environment.
```bash
env_up.sh
```

Next change directory to `noisy_nngps`, and run the following script.
```bash
run_exp.sh
```

