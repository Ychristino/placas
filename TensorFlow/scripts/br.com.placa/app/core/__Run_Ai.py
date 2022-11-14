import os
import sys
import tensorflow as tf
from tqdm import tqdm

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


class Run_Ai:

	def __init__(self, scripts_path='./models/research', config_path='./workspace/models/export/efficientdet_d0_v1/pipeline.config', model_path='./workspace/models/export/efficientdet_d0_v1/checkpoint', label_map_path='./workspace/data/label_map.pbtxt'):
		self.path2scripts = scripts_path
		self.path2config = config_path
		self.path2model = model_path 
		self.path2label_map = label_map_path	

	def gpu_test(self):
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
		
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here
		# specify which device you want to work on.
		# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
		os.environ["CUDA_VISIBLE_DEVICES"]="0"

		if tf.test.gpu_device_name():
			print('GPU found')
		else:
			print("No GPU found")

	def init_configuration(self):
		sys.path.insert(0, self.path2scripts) # making scripts in models/research available for import

		# do not change anything in this cell
		self.configs = config_util.get_configs_from_pipeline_file(self.path2config) # importing config
		self.model_config = self.configs['model'] # recreating model config
		self.detection_model = model_builder.build(model_config=self.model_config, is_training=False) # importing model


		self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
		self.ckpt.restore(os.path.join(self.path2model, 'ckpt-0')).expect_partial()

		self.category_index = label_map_util.create_category_index_from_labelmap(self.path2label_map,use_display_name=True)

		return self.detection_model, self.category_index


	def run(self):
		self.gpu_test()
		detection_model, category_index = self.init_configuration()
		return detection_model, category_index