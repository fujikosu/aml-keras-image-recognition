'''
Trains an image recognition model using Keras.

Uses Transfer Learning from an existing Keras "Application" (see https://keras.io/applications/), and trains as follows:
- Freeze all layers in the base model
- Add a new pooling layer (either global max or avg) if asked (see --pooling)
- Add one or more new dense layers (see --dense_layers) with activation functions (see --activation)
- Train the model using an optimizers[0] for epochs[0] epochs using learning_rates[0]
- Then unfreeze the model, and train for len(epochs)-1 additional rounds, for epochs[1:] epochs using optimizers[1:] and learning_rates[1:].
- Class weighting may be used (see --use_weights), and if so will weight proportionally to 1. - (# class N / total #).

It then writes the model file to --model_dir using a composite name for all model parameters, and a Markdown description alongsize. You can then use score_keras.py to evaluate the model.
'''
import matplotlib
# Force matplotlib to not use Xwindows backend otherwise it will crash on servers without Xwindows
matplotlib.use('Agg')
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import score_keras

from datetime import datetime
import os
import argparse
import logging
from distutils.util import strtobool
from utils.misc.azure_utils import load_file_from_blob
from utils.misc.zip_helper import unzip_file
from azureml.logging import get_azureml_logger
import time
import multiprocessing

FLAGS = None

logger = logging.getLogger('train_keras')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
aml_run_logger = get_azureml_logger()

activations = {
    'relu': 'relu',
    'prelu': lambda: PReLU(),
    'lrelu': lambda: LeakyReLU(),
    'elu': 'elu',
    'selu': 'selu',
    'tanh': 'tanh',
    'softmax': 'softmax'
}

optimizer_types = {
    'SGD': lambda lr: SGD(lr=lr), 
    'RMSprop': lambda lr: RMSprop(lr=lr), 
    'Adagrad': lambda lr: Adagrad(lr=lr), 
    'Adadelta': lambda lr: Adadelta(lr=lr), 
    'Adam': lambda lr: Adam(lr=lr), 
    'Adamax': lambda lr: Adamax(lr=lr), 
    'Nadam': lambda lr: Nadam(lr=lr)
}

model_types = {
    'InceptionV3': {
        'model': lambda: InceptionV3(weights='imagenet', include_top=False),
        'img_size': 299
    },
    'ResNet50': {
        'model': lambda: ResNet50(weights='imagenet', include_top=False),
        'img_size': 224
    },
    'VGG19': {
        'model': lambda: VGG19(weights='imagenet', include_top=False),
        'img_size': 224
    },
    'Xception': {
        'model': lambda: Xception(weights='imagenet', include_top=False),
        'img_size': 299
    },
    'InceptionResNetV2': {
        'model': lambda: InceptionResNetV2(weights='imagenet', include_top=False),
        'img_size': 299
    }
}

pooling_types = {
    'avg': lambda x: GlobalAveragePooling2D()(x),
    'max': lambda x: GlobalMaxPooling2D()(x),
    'none': lambda x: x
}

def build_model_name(options):
    ts = '' if not options.add_timestamp_suffix else datetime.now().strftime('_%Y%m%dT%H%M%S')
    mn = options.model_type
    topology = options.pooling + '-' + '-'.join(map(str, options.dense_layers))
    af = options.activation
    opts = '-'.join(options.optimizers)
    lrs = '-'.join([str(lr)[2:] for lr in options.learning_rates])
    wts = 'wts' if options.use_weights else 'nowts'
    epochs = '-'.join(map(str, options.epochs))
    return '{}_{}_{}_opts-{}_lr{}_{}_e{}{}'.format(mn, topology, af, opts, lrs, wts, epochs, ts)


def write_model_desc(options, model_path, model_name, classes, weights, train_gen, cm_path, metrics):
    desc_file = os.path.join(model_path, model_name + '_desc.md')
    logger.info('Writing model description to {}'.format(desc_file))
    with open(desc_file, 'w', encoding='utf-8') as fp:
        fp.write('# Model Details\n\n')
        fp.write('#### Model Type: {}\n\n'.format(options.model_type))
        fp.write(
            'Using transfer learning, removed last layer and added a {} pooling layer\n\n'.
            format(options.pooling))
        fp.write('Then added {} dense layers with [{}] nodes.\n\n'.format(
            len(options.dense_layers), ', '.join(
                map(str, options.dense_layers))))
        fp.write(
            '\nWe used {} as our activation function for all added layers.\n\n'.
            format(options.activation))
        if options.use_weights:
            fp.write(
                'We use class weighting to try and mitigate the imbalanced nature of the classes involved:\n\n'
            )
            for k in weights:
                fp.write('- {}: {}\n'.format(k, weights[k]))
        else:
            fp.write('No class weighting was used.\n')
        fp.write('\n\n# Training Details\n\n')
        fp.write(
            'We go through an initial training with frozen weights for all layers of the base model, using {} with a learning rate of {}, for {} epochs.\n\n'.
            format(options.optimizers[0], options.learning_rates[0], options.epochs[0]))
        fp.write(
            'After that, we unfreeze all layers and retrain {} times, using the following optimizers/learning rates/epochs:\n\n'.
            format(len(options.epochs) - 1))
        for opt, lr, epoch \
                in zip(options.optimizers[1:], options.learning_rates[1:], options.epochs[1:]):
            fp.write('- Using {} with Learning Rate {} for {} epochs'.format(opt, lr, epoch))
        fp.write('\n\n{} training images were used in {} classes.'.format(
            len(train_gen.classes), train_gen.num_classes))
        if cm_path or metrics:
            fp.write('\n\n# Scoring and Evaluation\n\n')
            if cm_path:
                fp.write('### Confusion Matrix:\n\n')
                fp.write('![Confusion Matrix](./{})\n\n'.format(os.path.basename(cm_path)))
            if metrics:
                logger.info('Metrics:')
                logger.info(metrics)
                fp.write('### Evaluation Metrics (on Test Set)\n\n')
                for metric in metrics.keys():
                    logger.info('Writing metric {}'.format(metric))
                    vals = metrics[metric]
                    # Metric is per-class
                    if hasattr(vals, '__iter__') and len(vals) == len(classes):
                        fp.write('- {}:\n'.format(metric))
                        #vals_by_class = dict(zip(classes, vals))
                        #aml_run_logger.log(metric, vals_by_class)
                        for idx in range(len(vals)):
                            fp.write('    - {}: {}\n'.format(classes[idx], vals[idx]))
                    else:
                        fp.write('- {}: {}\n'.format(metric, metrics[metric]))
                        try:
                            aml_run_logger.log(metric, metrics[metric])
                        except:
                            logger.warn('Failed to log metric {} to AzureML'.format(metric))

def load_images(img_path, flip, rotate, zoom, shear, batch_size, img_size,
                seed):
    logger.info(
        'Loading training data from {}. {}, rotate={}, zoom={}, shear={}. Batch size={}. Image size={}.'.
        format(img_path, 'flip' if flip else 'no flip', rotate, zoom, shear,
               batch_size, img_size))
    training = image.ImageDataGenerator(
        horizontal_flip=flip,
        vertical_flip=flip,
        rotation_range=rotate,
        zoom_range=zoom,
        shear_range=shear)
    train_gen = image.DirectoryIterator(
        os.path.join(img_path, 'training'),
        training,
        batch_size=batch_size,
        target_size=(img_size, img_size),
        seed=seed)
    validation = image.ImageDataGenerator()
    valid_gen = image.DirectoryIterator(
        os.path.join(img_path, 'validation'),
        validation,
        batch_size=batch_size,
        target_size=(img_size, img_size),
        seed=seed)
    return train_gen, valid_gen


def train_model(img_path,
                model_type,
                tf_log_dir,
                flip=False,
                rotate=0.,
                zoom=0.,
                shear=0.,
                batch_size=32,
                pooling='max',
                dense_layers=[1024],
                optimizers=['RMSprop', 'SGD'],
                learning_rates=[0.001, 0.005],
                activation='relu',
                epochs=[5, 5],
                use_weights=False,
                seed=1337,
                gpu=1):
    callbacks = None
    if tf_log_dir:
        # NOTE: Cannot write histograms when using generators as of Keras 2.0.8
        # If this changes, alter the histogram_freq below.
        logger.info('Logging to {} for TensorBoard'.format(tf_log_dir))
        callbacks = [
            TensorBoard(
                log_dir=tf_log_dir, histogram_freq=0, batch_size=batch_size)
        ]

    model_details = model_types[model_type]
    base_model = model_details['model']()
    img_size = model_details['img_size']
    train_gen, valid_gen = load_images(img_path, flip, rotate, zoom, shear,
                                       batch_size, img_size, seed)

    if use_weights:
        vals, freqs = np.unique(train_gen.classes, return_counts=True)
        wts = {}
        tot = freqs.sum()
        for i in range(len(vals)):
            wts[i] = 1. - float(freqs[i]) / tot
    else:
        wts = None
        # wts = dict(
        # zip(range(train_gen.num_classes), [1.] * train_gen.num_classes))
    logger.info('Using class weights {}'.format(wts))

    # Add new dense layers and softmax
    activation_fn = activations[activation]
    x = base_model.output
    x = pooling_types[pooling](x)
    for num_nodes in dense_layers:
        if type(activation_fn) is str:
            x = Dense(num_nodes, activation=activation_fn)(x)
        else:
            x = Dense(num_nodes)(x)
            x = activation_fn()(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        model = Model(inputs=base_model.input, outputs=predictions)
    logger.info(
        'Adding {} dense layers with {} nodes, {} pooling, {} activation.'.
        format(len(dense_layers), dense_layers, pooling, activation))

    # freeze all convolutional layers from base model
    for layer in base_model.layers:
        layer.trainable = False

    cpu_count = multiprocessing.cpu_count()

    logger.info('Initial training using Optimizer {} and LR {}'.format(\
        optimizers[0], learning_rates[0]))
    logger.info('Use {} GPUs'.format(gpu))
    aml_run_logger.log("cpu count", cpu_count)


    if gpu > 1:
        gpu_model = multi_gpu_model(model, gpus=gpu)
        batch_size = batch_size * gpu
        gpu_model.compile(
            optimizer=optimizer_types[optimizers[0]](learning_rates[0]),
            loss='categorical_crossentropy')
        start_time = time.perf_counter()
        gpu_model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen.classes) / batch_size,
            epochs=epochs[0],
            validation_data=valid_gen,
            validation_steps=len(valid_gen.classes) / batch_size,
            class_weight=wts,
            use_multiprocessing=True,
            workers=cpu_count,
            callbacks=callbacks)
        execution_time = time.perf_counter() - start_time
        aml_run_logger.log("Initial training execution time", execution_time)
    else:
        model.compile(
            optimizer=optimizer_types[optimizers[0]](learning_rates[0]),
            loss='categorical_crossentropy')
        start_time = time.perf_counter()
        model.fit_generator(
            train_gen,
            steps_per_epoch=len(train_gen.classes) / batch_size,
            epochs=epochs[0],
            validation_data=valid_gen,
            validation_steps=len(valid_gen.classes) / batch_size,
            class_weight=wts,
            use_multiprocessing=True,
            workers=cpu_count,
            callbacks=callbacks)
        execution_time = time.perf_counter() - start_time
        aml_run_logger.log("Initial training execution time", execution_time)

    num_to_unfreeze = -1 * (len(dense_layers) + 1)
    for layer in model.layers[:num_to_unfreeze]:
        layer.trainable = False
    for layer in model.layers[num_to_unfreeze:]:
        layer.trainable = True

    for optimizer, lr, epoch in zip(optimizers[1:], learning_rates[1:], epochs[1:]):
        logger.info('Training {} epochs using Optimizer {} and LR {}'.format(epoch, optimizer, lr))
        if gpu > 1:
            gpu_model = multi_gpu_model(model, gpus=gpu)
            gpu_model.compile(
                optimizer=optimizer_types[optimizer](lr),
                loss='categorical_crossentropy')
            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            start_time = time.perf_counter()
            gpu_model.fit_generator(
                train_gen,
                steps_per_epoch=len(train_gen.classes) / batch_size,
                epochs=epoch,
                validation_data=valid_gen,
                validation_steps=len(valid_gen.classes) / batch_size,
                class_weight=wts,
                use_multiprocessing=True,
                workers=cpu_count,
                callbacks=callbacks)
            execution_time = time.perf_counter() - start_time
            aml_run_logger.log("Second training execution time", execution_time)
        else:
            model.compile(
                optimizer=optimizer_types[optimizer](lr),
                loss='categorical_crossentropy')
            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            start_time = time.perf_counter()
            model.fit_generator(
                train_gen,
                steps_per_epoch=len(train_gen.classes) / batch_size,
                epochs=epoch,
                validation_data=valid_gen,
                validation_steps=len(valid_gen.classes) / batch_size,
                class_weight=wts,
                use_multiprocessing=True,
                workers=cpu_count,
                callbacks=callbacks)
            execution_time = time.perf_counter() - start_time
            aml_run_logger.log("Second training execution time", execution_time)
    return model, wts, train_gen, img_size


def evaluate(model_root, model, images, image_size, num_batches, seed, top_n=None):
    imagegen = image.ImageDataGenerator()
    test_gen = image.DirectoryIterator(
        os.path.join(images, 'testing'),
        imagegen,
        target_size=(image_size, image_size),
        seed=seed,
        shuffle=True)
    # Get classes sorted by their value
    classes = [x[0] for x in sorted(test_gen.class_indices.items(), key=lambda x: x[1])]
    metrics_path = model_root + "_metrics.csv"
    cm_path = model_root + "_cm.png"
    metrics, _, _ = score_keras.evaluate_model(model, test_gen, classes, num_batches,
                                               metrics_path, cm_path, top_n=top_n)
    class_map = model_root + "_classes.csv"
    try:
        with open(class_map, 'w', encoding='utf-8') as cmfp:
            cmfp.write('Class,ID\n')
            for i, c in enumerate(classes):
                cmfp.write('"{}",{}\n'.format(c, i))
    except:
        logger.warn('Failed to write class map file.')
    return classes, cm_path, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--flip',
        type=strtobool,
        default=False,
        help=
        'Whether to augment training images with flips (horiz and vert). Defaults to False.'
    )
    parser.add_argument(
        '--rotate',
        type=float,
        default=0.,
        help='Degrees to rotate (for training augmentation). Defaults to 0.')
    parser.add_argument(
        '--zoom',
        type=float,
        default=0.,
        help='Pct to zoom in/out (for training augmentation). Defaults to 0.')
    parser.add_argument(
        '--shear',
        type=float,
        default=0.,
        help='Shear range (for training augmentation). Defaults to 0.')
    parser.add_argument(
        '--tensorflow_logs',
        type=str,
        default='./outputs/tf_logs/',
        help='Path to output tensorflow logs. Defaults to ./tf_logs/.')
    parser.add_argument(
        '--model_type',
        type=str,
        default='InceptionV3',
        choices=model_types.keys(),
        help=
        'Type of pre-trained model to use. See https://keras.io/applications/. Defaults to InceptionV3.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./outputs/models',
        help=
        'model_dir + output_model + ".h5" == full output model file path. Defaults to ./models.')
    parser.add_argument(
        '--output_model',
        default=None,
        type=str,
        help=
        'Filename (sans prefix) for saved model. Defaults to structured combination of training parameters.')
    parser.add_argument(
        '--add_timestamp_suffix',
        default=False,
        type=strtobool,
        help='Turn on/off the timestamp suffix on model (and desc, and cm, and metrics). Defaults to False.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size. Defaults to 32.')
    parser.add_argument(
        '--dense_layers',
        type=int,
        nargs='+',
        default=[1024],
        help='Number of nodes per added dense layer. Defaults to [1024].')
    parser.add_argument(
        '--pooling',
        type=str,
        default='avg',
        choices=pooling_types.keys(),
        help=
        'Type of pooling layer to add between featurization layer and new dense layer(s). Defaults to avg.'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=activations.keys(),
        help=
        'Activation function to use for additional dense layers. Defaults to relu.'
    )
    parser.add_argument(
        '--optimizers',
        type=str,
        nargs='+',
        default=['RMSprop', 'SGD'],
        help='Optimzers to use for training. Defaults to RMSProp for initial training and SGD for subsequent.'
    )
    parser.add_argument(
        '--learning_rates',
        type=float,
        nargs='+',
        default=[0.001, 0.005],
        help='Learning rates. Defaults to [0.001, 0.005].')
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        default=[5, 5],
        help=
        'Number of epochs to use for each training session. Defaults to [5, 5].'
    )
    parser.add_argument(
        '--use_weights',
        type=strtobool,
        default=False,
        help='Use class weights relative to frequencies. Defaults to False.')
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for directory iteration. Defaults to 1337.')
    parser.add_argument(
        '--score',
        type=strtobool,
        default=False,
        help='Score the model after training using score_keras.')
    parser.add_argument(
        '--num_batches_to_score',
        type=int,
        default=10,
        help='If scoring, how many batches to score.')
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='Number of epochs to use for each training session.')
    FLAGS, _ = parser.parse_known_args()
    if len(FLAGS.learning_rates) != len(FLAGS.epochs):
        raise Exception('Must provide as many LRs as Epochs.')
    model_name = FLAGS.output_model if FLAGS.output_model else build_model_name(
        FLAGS)
    logger.info('Model name {}'.format(model_name))
    shared_data_path = os.path.join(os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'])
    container_name = "data"
    zip_file_name = "output_all.zip"
    data_dir = "data"
    load_file_from_blob(container_name, zip_file_name, os.path.join(shared_data_path, zip_file_name))
    unzip_file(os.path.join(os.path.join(shared_data_path, zip_file_name)),os.path.join(shared_data_path,data_dir))
    FLAGS.image_dir = os.path.join(shared_data_path,data_dir,zip_file_name.split(".")[0])
    trained_model, weights, training_data, im_sz = train_model(
        FLAGS.image_dir,
        FLAGS.model_type,
        FLAGS.tensorflow_logs,
        flip=FLAGS.flip,
        rotate=FLAGS.rotate,
        zoom=FLAGS.zoom,
        shear=FLAGS.shear,
        batch_size=FLAGS.batch_size,
        pooling=FLAGS.pooling,
        activation=FLAGS.activation,
        dense_layers=FLAGS.dense_layers,
        optimizers=FLAGS.optimizers,
        learning_rates=FLAGS.learning_rates,
        epochs=FLAGS.epochs,
        use_weights=FLAGS.use_weights,
        seed=FLAGS.seed,
        gpu=FLAGS.gpu)
    aml_run_logger.log('model_name', model_name)
    model_root = os.path.join(FLAGS.model_dir, model_name)
    model_file = model_root + '.h5'
    logger.info('Saving model to {}'.format(model_file))
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    trained_model.save(model_file)
    aml_run_logger.log("hyperparameters", {
        "flip": FLAGS.flip,
        "rotate": FLAGS.rotate,
        "zoom": FLAGS.zoom,
        "shear": FLAGS.shear,
        "batch_size": FLAGS.batch_size,
        "pooling": FLAGS.pooling,
        "activation": FLAGS.activation,
        "dense_layers": FLAGS.dense_layers,
        "optimizers": FLAGS.optimizers,
        "learning_rates": FLAGS.learning_rates,
        "epochs": FLAGS.epochs,
        "weights": weights
    })
    classes = None
    cm_path = None
    metrics = None
    if FLAGS.score:
        logger.info('Model and description saved. Evaluating and scoring.')
        classes, cm_path, metrics = evaluate(model_root, trained_model, FLAGS.image_dir, im_sz,
            FLAGS.num_batches_to_score, FLAGS.seed, top_n=3)
    write_model_desc(FLAGS, FLAGS.model_dir, model_name, classes, weights,
                        training_data, cm_path, metrics)
