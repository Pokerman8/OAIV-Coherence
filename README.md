# OAIV-Coherence
Official pytorch implementation of OAIV-Coherence(Enhancing Seam Carving with Object Awareness for Improved Visual Coherence)

<img src="./framework.png">


## Introduction
Seam carving is a classic topic in computer graphics, playing a crucial role in reducing misalignment caused by differences in captured perspectives and object motion in image stitching. Traditional seam carving methods fail to consider semantic information, resulting in disrupted foreground continuity. We propose a deep learning-based framework that leverages semantic priors of foreground objects, introducing a novel loss function to preserve semantic integrity and enhance visual coherence. Additionally, We propose two specialized real-world datasets to evaluate our method. Experimental results demonstrate significant improvements in image quality, addressing traditional technique limitations and providing robust support for practical applications.

## Preparation

### Install

We implement this work with UWSL2(Ubuntu24.04), 3090, and CUDA11.

1. Git clone this repository.
   ```
  git clone https://github.com/Pokerman8/OAIV-Coherence.git
  ```

3. Create a new conda environment
  ```
  conda env create -f environment.yml
  ```




### DATASET

Download the dataset from []().



## Testing

### test with our pretrain model

