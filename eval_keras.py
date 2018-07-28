'''
Evaluates one or more images with an existing model.
'''

from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from datetime import datetime
import os
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import argparse
import logging
from pathlib import Path

logger = logging.getLogger('eval_keras')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

FLAGS = None


def load_model_and_classmap(model_file, class_map_file):
    model = load_model(str(model_file))
    with class_map_file.open('r', encoding='utf-8') as cmfp:
        reader = csv.reader(cmfp)
        next(reader, None)  # Skip header
        class_map = dict((int(row[1]), row[0]) for row in reader)

    return model, class_map


def evaluate_model(model, img_file, class_map, top_n=3):
    img = image.load_img(img_file, target_size=(299, 299))
    imarr = image.img_to_array(img)
    imarr = np.expand_dims(
        imarr, axis=0)  # add zeroth axis of single element for "batch"
    pred = model.predict(imarr)
    top_idx = len(class_map.keys()) - top_n
    pred_ordered = np.argpartition(pred, top_idx, axis=1)
    best_n_idx = pred_ordered[0][top_idx:]
    return list(zip([class_map[ix] for ix in best_n_idx], \
        pred_ordered[0][top_idx:], pred[0][best_n_idx]))[::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models',
        help=
        'model_dir + model_name + "h5" == full model file path. Default is ./models.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help=
        'Name of model file (model_dir + model_name + ".h5" == full file path).'
    )
    parser.add_argument(
        '--image',
        type=str,
        nargs='+',
        required=True,
        help='Image to evaluate, may specify more than one.')
    FLAGS, _ = parser.parse_known_args()
    model_path = Path(FLAGS.model_dir) / (FLAGS.model_name + ".h5")
    class_map_path = Path(FLAGS.model_dir) / (
        FLAGS.model_name + "_classes.csv")
    m, cm = load_model_and_classmap(model_path, class_map_path)
    logger.info('Loaded model from {}'.format(model_path))
    for img in FLAGS.image:
        logger.info(evaluate_model(m, img, cm))
