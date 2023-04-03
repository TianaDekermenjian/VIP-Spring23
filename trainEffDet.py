import os
import random
import shutil
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# The label map as a dictionary (zero is reserved):
label_map = {1: 'Basketball'}

# Specify dataset
images_in = "Dataset/withmade_dataset1/withmade_images"
annotations_in = "Dataset/withmade_dataset1/withmade_labelsxml"

train_dir, val_dir, test_dir = split_dataset(images_in, annotations_in, val_split=0.2, test_split=0.2, out_path='split-dataset')

train_data = object_detector.DataLoader.from_pascal_voc(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'annotations'), label_map=label_map)
validation_data = object_detector.DataLoader.from_pascal_voc(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'annotations'), label_map=label_map)
test_data = object_detector.DataLoader.from_pascal_voc(os.path.join(test_dir, 'images'), os.path.join(test_dir, 'annotations'), label_map=label_map)

# Check
print(f'train count: {len(train_data)}')
print(f'validation count: {len(validation_data)}')
print(f'test count: {len(test_data)}')

# Specify EfficientDet Model
spec = object_detector.EfficientDetLite0Spec()

model = object_detector.create(train_data=train_data, model_spec=spec, validation_data=validation_data, epochs=50, batch_size=10, train_whole_model=True)

def split_dataset(images_path, annotations_path, val_split, test_split, out_path):

    # Args:
    #   images_path: Path to the directory with your images (JPGs).
    #   annotations_path: Path to a directory with your VOC XML annotation files,
    #   with filenames corresponding to image filenames.
    #   val_split: Fraction of data to reserve for validation (float between 0 and 1).
    #   test_split: Fraction of data to reserve for test (float between 0 and 1).
    # Returns:
    #   The paths for the split images/annotations (train_dir, val_dir, test_dir)

    _, dirs, _ = next(os.walk(images_path))

    train_dir = os.path.join(out_path, 'train')
    val_dir = os.path.join(out_path, 'validation')
    test_dir = os.path.join(out_path, 'test')

    IMAGES_TRAIN_DIR = os.path.join(train_dir, 'images')
    IMAGES_VAL_DIR = os.path.join(val_dir, 'images')
    IMAGES_TEST_DIR = os.path.join(test_dir, 'images')
    os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
    os.makedirs(IMAGES_TEST_DIR, exist_ok=True)

    ANNOT_TRAIN_DIR = os.path.join(train_dir, 'annotations')
    ANNOT_VAL_DIR = os.path.join(val_dir, 'annotations')
    ANNOT_TEST_DIR = os.path.join(test_dir, 'annotations')
    os.makedirs(ANNOT_TRAIN_DIR, exist_ok=True)
    os.makedirs(ANNOT_VAL_DIR, exist_ok=True)
    os.makedirs(ANNOT_TEST_DIR, exist_ok=True)

    # Get all filenames for this dir, filtered by filetype
    filenames = os.listdir(os.path.join(images_path))
    filenames = [os.path.join(images_path, f) for f in filenames if (f.endswith('.jpg'))]
    # Shuffle the files, deterministically
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    # Get exact number of images for validation and test; the rest is for training
    val_count = int(len(filenames) * val_split)
    test_count = int(len(filenames) * test_split)
    for i, file in enumerate(filenames):
        source_dir, filename = os.path.split(file)
        annot_file = os.path.join(annotations_path, filename.replace("jpg", "xml"))
        if i < val_count:
            shutil.copy(file, IMAGES_VAL_DIR)
            shutil.copy(annot_file, ANNOT_VAL_DIR)
        elif i < val_count + test_count:
            shutil.copy(file, IMAGES_TEST_DIR)
            shutil.copy(annot_file, ANNOT_TEST_DIR)
        else:
            shutil.copy(file, IMAGES_TRAIN_DIR)
            shutil.copy(annot_file, ANNOT_TRAIN_DIR)
    return (train_dir, val_dir, test_dir)

