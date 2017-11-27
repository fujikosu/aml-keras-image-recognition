#
# NOTE: Some of this code was adapted from tensorflow/examples/image_retraining/retrain.py
#       As such, although the repo as a whole is MIT Licensed, _this particular script_
#       is licensed under the Apache 2 license that TensorFlow uses.
#       HOWEVER, I have extensively altered this script to remove all TensorFlow dependencies.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Split an image directory into train, validation, and test sets.

Note that the split is done using hashing on the filenames so that subsequent runs with additional images in the input set will wind up hashing the resulting images into the same train/test/validation sets. The split is done per-class, so it is a stratified (balanced) split.

The output directory, once finished, will contain subdirectories named 'training', 'testing', and 'validation' - this allows the subsequent train_keras and score_keras scripts to assume inputs making their commands more streamlined.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import hashlib
import re
import glob
from shutil import copyfile
import logging

logger = logging.getLogger('score_keras')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2**27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
    if not os.path.exists(image_dir):
        logger.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [os.path.basename(x) for x in glob.glob(image_dir + '/*') if os.path.isdir(x)]
    # The root directory comes first, so skip it.
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        logger.info("Looking for images in '{}'".format(os.path.join(image_dir, dir_name)))
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            logger.warning('No files found')
            continue
        if len(file_list) < 20:
            logger.warning('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            logger.warning(
                'WARNING: Folder {} has more than {} images. Some images will never be selected.'
                    .format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(
                hash_name.encode(errors='replace')).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (
                    testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

def mkdir(root, dirname):
    path = os.path.join(root, dirname)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def divide_images():
    img_dir = FLAGS.image_dir
    testing_pct = FLAGS.pct_test
    validation_pct = FLAGS.pct_validation
    out_dir = FLAGS.output_dir
    image_lists = create_image_lists(img_dir, testing_pct, validation_pct)
    class_count = len(image_lists.keys())
    if class_count == 0:
        logger.error('No valid folders of images found at ' +
                         FLAGS.image_dir)
        return -1
    if class_count == 1:
        logger.error('Only one valid folder of images found at ' +
                         FLAGS.image_dir +
                         ' - multiple classes are needed for classification.')
        return -1
    mkdir('', out_dir)
    train_dir = mkdir(out_dir, 'training')
    test_dir = mkdir(out_dir, 'testing')
    val_dir = mkdir(out_dir, 'validation')
    for cl in image_lists.keys():
        td_cl = mkdir(train_dir, cl)
        te_cl = mkdir(test_dir, cl)
        v_cl = mkdir(val_dir, cl)
        indir = os.path.join(img_dir, image_lists[cl]['dir'])
        for img in image_lists[cl]['training']:
            copyfile(os.path.join(indir, img), os.path.join(td_cl, img))
        for img in image_lists[cl]['testing']:
            copyfile(os.path.join(indir, img), os.path.join(te_cl, img))
        for img in image_lists[cl]['validation']:
            copyfile(os.path.join(indir, img), os.path.join(v_cl, img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to folders of labeled images.')
    parser.add_argument(
        '--output_dir', type=str, required=True, help='Where to save the divided images.')
    parser.add_argument(
        '--pct_test',
        type=int,
        default=10,
        help='Percentage of input set to use for training.')
    parser.add_argument(
        '--pct_validation',
        type=int,
        default=20,
        help='Percentage of images to use in validation.')
    parser.add_argument(
        '--seed', type=float, default=1337, help='Random seed.')
    FLAGS, _ = parser.parse_known_args()
    divide_images()
