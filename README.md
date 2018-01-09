# aml-keras-image-recognition

A sample [Azure Machine Learning services](https://azure.microsoft.com/en-us/services/machine-learning-services/) project for Transfer Learning-based custom image recognition by utilizing Keras.

## Prerequisites

1. An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
2. An installed copy of Azure Machine Learning Workbench with a workspace created. ([Create Azure Machine Learning Preview accounts and install Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation))
3. This example could be run on any compute context.

## Usage

This project uses Training dataset, Test dataset and Validation dataset.

For preparing this directory structure,
run `train_test_split.py` like below on your dataset and one whole dataset is split into 3 directories (training, testing, validation).

`python -m scripts.train_test_split --image_dir data/my_photos --output_dir data/my_split --pct_test 10 --pct_validation 20 --seed 1337`

Then compress them into one zip file and name it as "image_dataset.zip".

Store it under "data" container in Azure Blob Storage and add the following references to your .runconfig file to load dataset into your Azure Machine Learning compute target automatically for your training:

```
EnvironmentVariables:
  "STORAGE_ACCOUNT_NAME": "<YOUR_AZURE_STORAGE_ACCOUNT_NAME>"
  "STORAGE_ACCOUNT_KEY": "<YOUR_AZURE_STORAGE_ACCOUNT_KEY>"
```

Run `train_keras.py` like below from Azure Machine Learning.

`az ml experiment submit -c YOUR_VM_TARGET .\train_keras.py --gpu 2 --use_weights True --score True --learning_rates 0.001 0.0005 0.00002 --epochs 20 10 10 --model_type Xception --num_batches_to_score 100`

All of run histories, logs and trained models are managed by Azure Machine Learning Services.

[This documentation](train_keras.py) tells you how to set up GPU VMs for Azure Machine Learning Services.

## Scripts

There are three scripts in this module - one for splitting your image data, one for training your model, and the final for scoring your model and getting a picture of the confusion matrix.

- `train_test_split.py`
   - Splits an existing directory of image files (with subdirectories named for the labels)
   - Uses hashing on the filenames to ensure that even if you add more files, existing files will hash into the same buckets they did previously
- `train_keras.py`
   - Trains a Keras model for image recognition based on Transfer Learning. See [Keras Applications](https://keras.io/applications/) for a sense of the pre-built models that are available. 
   - As new models are launched or you develop/find them, it's easy to add them to this script and use them as base models for your own transfer-learned versions.
   - Models are automatically named based on the hyperparameters given, and can also have a timestamp appended. However, this can be overridden if needed.
   - Data augmentation flags are supported, and include the methods supported by Keras (flipping horizontally and vertically, rotation, zooming, and shearing).
   - You can bundle training and scoring if desired.
   - As part of the training, a Markdown file describing the model is produced. If you are also doing scoring, details on the performance will also be included.
- `score_keras.py`
   - Takes an existing trained model and runs multiple passes over the test-set of images, generating a confusion matrix and precision, recall and F-score values.
- All scripts support the `--help` flag to give you details on usage, and should guide you in required vs. optional parameters. If you have any issues, please feel free to reach out.

# Examples

## `train_test_split.py`

- `python -m scripts.train_test_split --image_dir data/my_photos --output_dir data/my_split --pct_test 10 --pct_validation 20 --seed 1337`
   - Splits images in `data/my_photos` into output directory `data/my_split`
   - 10% of the images in each class go into `testing`, 20% into `validation`, and the other 70% into `training`

## `train_keras.py`

- `python -m scripts.train_keras --image_dir data/my_split --model_type InceptionV3 --batch_size 8 --learning_rates 0.001 0.001 0.0005 --epochs 1 1 1 --use_weights True --score True --model_dir ./models --gpu 1`
   - Trains a new model based on a pre-trained InceptionV3 instance.
   - Since `output_model` is not specified, the name is imputed from the hyperparameters.
   - Uses class-weights based on inverse class distribution.
   - Scores the results and writes out confusion matrix and description markdown file.

## `score_keras.py`

- `python -m scripts.score_keras --model_dir ./models --model_name InceptionV3_avg-1024_relu_lr001-001-0005_wts_e1-1-1 --num_batches 20 --image_dir ./data/my_split`
   - Scores an existing model (`./models/InceptionV3_avg-1024_relu_lr001-001-0005_wts_e1-1-1.h5`)
   - Uses images from `./data/my_split/testing` for evaluation.
   - Stores confusion matrix and scores into `./models`, into a `.png` and `.csv` respectively.

# LICENSE

See `LICENSE`. Overall code is licensed under MIT license. `train_test_split.py` is Apache v2.0 licensed because it is adapted from TensorFlow code (see comments in that file).