# Neural Waveshaping Synthesis

This repository is the official implementation of [Neural Waveshaping Synthesis](#). 

[Colab Notebook](https://colab.research.google.com/github/anon9119/anon-project-1/blob/main/colab/NEWT_Timbre_Transfer.ipynb)

## Requirements

To install:

```setup
pip install -r requirements.txt
pip install -e .
```

We recommend installing in a virtual environment.

## Data

We trained our checkpoints on the [URMP](http://www2.ece.rochester.edu/projects/air/projects/URMP.html) dataset.
Once downloaded, the dataset can be preprocessed using `scripts/create_urmp_dataset.py`. 
This will consolidate recordings of each instrument within the dataset and preprocess them according to the pipeline in the paper.

```bash
python scripts/create_urmp_dataset.py \
  --gin-file gin/data/urmp_4second_crepe.gin \ 
  --data-directory /path/to/urmp \
  --output-directory /path/to/output \
  --device cuda:0  # torch device string for CREPE model
```

Alternatively, you can supply your own dataset and use the general `create_dataset.py` script:

```bash
python scripts/create_dataset.py \
  --gin-file gin/data/urmp_4second_crepe.gin \ 
  --data-directory /path/to/dataset \
  --output-directory /path/to/output \
  --device cuda:0  # torch device string for CREPE model
```

## Training

To train a model on the URMP dataset, use this command:

```bash
python scripts/train.py \
  --gin-file gin/train/train_newt.gin \
  --dataset-path /path/to/processed/urmp \
  --urmp \
  --instrument vn \  # select URMP instrument with abbreviated string
  --load-data-to-memory
```

Or to use a non-URMP dataset:
```bash
python scripts/train.py \
  --gin-file gin/train/train_newt.gin \
  --dataset-path /path/to/processed/data \
  --load-data-to-memory
```
