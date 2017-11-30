'''
Scores an existing model using one or more batches of images from a testing set.

Outputs a confusion matrix, and precision, recall, F-score and support metrics.
'''

from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from datetime import datetime
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import argparse
import logging

logger = logging.getLogger('score_keras')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

FLAGS = None

def plot_confusion_matrix(cm, classes, outfile,
                          title='Confusion Matrix',
                          cmap=plt.cm.BuPu,
                          size_inches=6):
    """
    Plots the confusion matrix and stores into outfile.
    """
    fig = plt.figure()
    fig.set_size_inches(size_inches, size_inches)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    class_labels = [ '{} ({})'.format(classes[idx], idx) for idx in range(len(classes))] 
    plt.xticks(tick_marks, class_labels, rotation=90)
    plt.yticks(tick_marks, class_labels)

    halfway = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > halfway else 'black')

    plt.tight_layout()
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)

def load_model_and_images(model_file, image_dir, seed):
    testpath = image_dir

    imagegen = image.ImageDataGenerator()
    test_gen = image.DirectoryIterator(testpath, imagegen, target_size=(224, 224), seed=seed, shuffle=True)

    model = load_model(model_file)
    # Get classes sorted by their value
    classes = [x[0] for x in sorted(test_gen.class_indices.items(), key=lambda x: x[1])]

    return model, test_gen, classes

def evaluate_model(model, image_gen, classes, num_batches, output_metrics, output_image, top_n=None, beta=1.):
    for batch_num in range(num_batches):
        logger.info('Processing batch %d of %d' % (batch_num, num_batches))
        cur_x, cur_y = image_gen.next()
        cur_y_pred = model.predict(cur_x)
        if batch_num == 0:
            y_actual = cur_y
            y_pred = cur_y_pred
        else:
            y_actual = np.concatenate((y_actual, cur_y))
            y_pred = np.concatenate((y_pred, cur_y_pred))
    y_v = np.argmax(y_actual, axis=1)
    y_p_v = np.argmax(y_pred, axis=1)
    metrics = {}
    if output_image:
        logger.info('Writing confusion matrix to %s' % output_image)
        plot_confusion_matrix(confusion_matrix(y_v, y_p_v), classes, output_image)
    if output_metrics:
        logger.info('Writing metrics to %s' % output_metrics)
        precision, recall, fscore, support = precision_recall_fscore_support(y_v, y_p_v, beta=beta)
        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F-Score': fscore,
            'Support': support
        }
        if top_n:
            top_idx = len(classes) - top_n
            y_p_ordered = np.argpartition(y_pred, top_idx, axis=1)
            cnt = 0
            # Is there a numpy way to do this without iterating?
            # I was looking at np.where, but can't figure out how to 
            #  broadcast actuals against predictions
            for actual, pred in zip(y_v, y_p_ordered):
                if actual in pred[top_idx:]:
                    # print('Found {} at index {} of {}'.format(actual, np.where(pred==actual)[0], pred))
                    cnt += 1
                #else:
                    # print('Failed to find {} before {}, found at index {} of {}'.format(actual, top_idx, np.where(pred==actual)[0], pred))
            tot = len(y_v)
            logger.info('Top {}: {} of {}'.format(top_n, cnt, tot))
            top_n_value = float(cnt) / tot
            metrics['Top_{}'.format(top_n)] = top_n_value
            logger.info('Precision: {}, Recall: {}, F-Score: {}, Support: {}, Top_{}: {}'\
                .format(precision, recall, fscore, support, top_n, top_n_value))
        else:
            logger.info('Precision: {}, Recall: {}, F-Score: {}, Support: {}'.format(precision, recall, fscore, support))
        os.makedirs(os.path.dirname(output_metrics), exist_ok=True)
        with open(output_metrics, 'w', encoding='utf-8') as fp:
            if top_n:
                fp.write('Precision, Recall, F_Score, Top_{}, Support\n'.format(top_n))
                fp.write('{}, {}, {}, {}, {}\n'.format(precision, recall, fscore, top_n_value, support))
            else:
                fp.write('Precision, Recall, F_Score, Support\n')
                fp.write('{}, {}, {}, {}\n'.format(precision, recall, fscore, support))
    return metrics, y_actual, y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models',
        help='model_dir + output_model + "h5" == full output model file path. Default is ./models.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of model file (model_dir + model_name + ".h5" == full file path).')
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Base path to images, will use "testing" subdirectory underneath.')
    parser.add_argument(
        '--output_img',
        type=str,
        default=None,
        help='Path for output confusion matrix. Defaults to (model_dir + model_name + "_cm.png")')
    parser.add_argument(
        '--output_metrics',
        type=str,
        default=None,
        help='Path for output metrics. Defaults to (model_dir + model_name + "_metrics.csv")')
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='Number of batches to iterate for testing. Defaults to 10.')
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Set seed for randomization of image iteration')
    FLAGS, _ = parser.parse_known_args()
    model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name + ".h5")
    cm_path = FLAGS.output_img if FLAGS.output_img else os.path.join(FLAGS.model_dir, FLAGS.model_name + "_cm.png")
    metrics_path = FLAGS.output_metrics if FLAGS.output_metrics else os.path.join(FLAGS.model_dir, FLAGS.model_name + "_metrics.csv")
    img_path = FLAGS.image_dir
    if os.path.exists(os.path.join(img_path, "testing")):
        img_path = os.path.join(img_path, "testing")
    m, i, c = load_model_and_images(model_path, img_path, FLAGS.seed)
    logger.info('Loaded model from {}'.format(model_path))
    logger.info('Evaluating model using images from {}'.format(img_path))
    evaluate_model(m, i, c, FLAGS.num_batches, metrics_path, cm_path, top_n=1)
