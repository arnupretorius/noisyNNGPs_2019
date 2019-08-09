# Code: On the expected behaviour of noise regularised neural networks as Gaussian processes.

This repository provides the code to reproduce the results in the paper: "On the expected behaviour of noise regularised neural networks as Gaussian processes."

The code was written by Arnu Pretorius. Large portions of the code was originally adapted from code that was made available by Lee et al. (2018) at https://github.com/brain-research/nngp.

## Basic requirements to plot Figures 1, 3 and 4

To reproduce Figures 1, 3 and 4 all that is required is `numpy`, `pandas`, `seaborn` and `matplotlib`. Each figure corresponds to a notebook in the repository.

## Requirements to regenerate the results for Figures 1 and 3: 

To regenerate the results for Figures 1 and 3, follow the instructions given below.

### Installation

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Step 2. Pull the deepo image to duplicate the research environment.

```bash
docker pull ufoym/deepo:tensorflow-py27-cu90
```
#### Step 3. Clone the research code repository. 
```bash
git clone https://github.com/arnupretorius/noisyNNGPs_2019.git
```

### Usage

Change directory to the cloned repository on your local machine and run the bash script `env_up.sh` to start the docker container with the correct environment.

Next change directory to `noisy_nngps`, and run the following script `run_exp.sh`.

### References

- Pretorius, A., Kamper, H., & Kroon, S. On the expected behaviour of noise regularised neural networks as Gaussian processes. Under review: NeurIPS, 2019.
- Lee, J., Bahri, Y., Novak, R., Schoenholz, S.S., Pennington, J. and Sohl-Dickstein, J. Deep neural networks as gaussian processes. ICLR, 2018.
