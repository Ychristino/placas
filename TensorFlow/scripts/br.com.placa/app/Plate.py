import cv2
import numpy as np
import tensorflow as tf

from __Image_Filter import Image_Filter
from __Run_Ai import Run_Ai
from Filter import Filter
from Image_Crop import Image_Crop
import tensorflow as tf

from object_detection.utils import visualization_utils as viz_utils


class Plate:
    Filter = Filter()
    Crop = None

    def __init__(self, file_path=None):
        if file_path is not None:
            self.file_path = file_path
        AI = Run_Ai()
        self.detection_model, self.category_index = AI.run()

    def open(self, file_path=None, gscale=False, th=None, noise=None):
        if file_path is not None:
            self.file_path = file_path

        self.image = cv2.imread(self.file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.Crop = Image_Crop(self.image)
        
        if gscale:
            self.image = Image_Filter().grayscale(self.image)
        if th is not None:
            self.image = Image_Filter().threshhold(self.image, th)
        if noise is not None:
            self.image = Image_Filter().noise(self.image, noise)

        return self.image

    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def find(self, image):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
        for key, value in detections.items()}

        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = (detections['detection_classes'] + 1).astype(np.int64)

        return detections, num_detections

    def draw_boxes(self, image, detections, box_th=0.25, max_box=200, normalize_coordinates=True, agnostic_mode=False, line_size=5):
    	viz_utils.visualize_boxes_and_labels_on_image_array(
			image,
			detections['detection_boxes'],
			detections['detection_classes'],
			detections['detection_scores'],
			self.category_index,
			use_normalized_coordinates=normalize_coordinates,
			max_boxes_to_draw=max_box,
			min_score_thresh=box_th,
			agnostic_mode=agnostic_mode,
			line_thickness=line_size)

    	return image

if __name__ == '__main__':
    print('Opss... Wrong way.')
    print('This is class is not acessable directly... you should turn back.')