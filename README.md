# Thrivata

This repository contains the machine learning model used for action recognition. It uses a modified version of the model in this [TensorFlow tutorial](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet) for 3D transfer learning using MoviNet. The model uses a pre-trained MoviNet model in a frozen state with an extra layer added at the end to tailor the model to the specific classification task. Many of the model parameters may be modified in the form of the *config* json file. The goal of this model is to be able to recognize actions from any angle, so a diverse dataset with samples from different persectives is strongly recommended. Data should be kept online and downloaded from a specified URL when the model is run to prevent storage issues. The pre-trained MoviNet model is also downloaded at runtime.

## Contributors
Mark Zheng (2023)

## How to Run
In terminal, run "**py run.py config.json**" where config.json is the name of your desired config file.

## Usage
* Define a config file appropriate for your desired task. Please refer to "config.json" for an example of the formatting.
* If required, implement additional functions to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Run *run.py* as  specified in the "How to Run" section.
* Save any logs, stats, and plots into *./figures/exp* dir where "exp" is the name of the specific process being run.

## Files

* `run.py`: Main driver class used to run model
* `config`: Directory containing config files for running models
* `data`: Directory for storing data used to train models
* `figures`: Directory for storing generated figures
* `model_downloads`: Directory for storing pre-trained models
* `notebooks`: Directory containing Jupyter Notebooks
* `src`: Parent directory for project source code
    * `datasets`: Directory for loading data
        * `load_datasets.py`: Helper functions used to download and unpack data
    * `models`: Directory for model related functions
        * `train_models.py`: Model training and loading pre-trained models
    * `preprocessing`: Directory for preprocessing data
        * `build_features.py`: Helper functions preparing data to be inputted into the model
    * `visualize`: Directory containing source code for graphs, charts, etc.
        * `visualize.py`: Functions to generate graphs