"""
Usage:
python ./scripts/generate_tfrecord.py --csv_input=C:/Users/Yan/Desktop/placas/dataset/dataset/annotations.csv  --output_path=C:/Users/Yan/Desktop/placas/TensorFlow/workspace/data/treino.record --image_dir=C:/Users/Yan/Desktop/placas/dataset/dataset
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from Dictionary import Dictionary

flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

class TFRecord:
    dictionary = Dictionary()
    
    def __init__(self, annotations, output_path, image_dir):
        self.annotations = annotations
        self.output_path = output_path
        self.image_dir = image_dir

    # TO-DO replace this with label map
    def class_text_to_int(self, row_label):
        return self.dictionary.values().index(row_label)

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def generate_label_map(self, output_path='./'):
        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdirs(output_path)

        with open(f'{output_path}/label_map.pbtxt', 'w') as f:
            for key in self.dictionary.keys():
                f.writelines('item{\n')
                f.writelines(f'\tid:{key}\n')
                f.writelines(f'\tname:{self.dictionary.values()[key]}\n')
                f.writelines('}\n')
            f.close()

    def run(self):
        writer = tf.compat.v1.python_io.TFRecordWriter(self.output_path)
        path = os.path.join(self.image_dir)
        examples = pd.read_csv(self.annotations)
        grouped = self.split(examples, 'filename')

        for group in grouped:
            tf_example = self.create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()

        output_path = os.path.join(os.getcwd(), self.output_path)


if __name__ == '__main__':

parser = argparse.ArgumentParser(description='Generate TF RECORD')
parser.add_argument('-a', '--annotations', required=True, nargs=1, dest='annotations', action='store', help='File path for the CSV file with annotations about the images (file name, xmin, ymin, xmax, ymax, width, height).')
parser.add_argument('-o', '--out_file', required=True, nargs=1, dest='output_path', action='store', help='Output path for the file with the TFRecord data.')
parser.add_argument('-im', '--imdir', required=True, nargs=1, dest='image_dir', action='store', help='Folder to the images.')

parser.add_argument('-lbl', '--label_map', required=False, nargs=1, dest='label_map', action='store', help='Output path for the label map file.')

args = parser.parse_args()
tfr = TFRecord(args.annotations, args.output_path, args.image_dir)

if args.label_map is not None:
    tf.generate_label_map(args.label_map)

tfr.tf.compat.v1.app.run()
