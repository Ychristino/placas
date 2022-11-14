from Plate import Plate
from Extract_Data import Extract_Data
from Coordinates import Coordinates
from Ocr import OCR
class Main:


	def run(self, input_file):
		plate = Plate(input_file)
		original_image = plate.open()

		detections, num_detections = plate.find(original_image)

		detections = plate.Filter.classes(detections, [1])
		detections = plate.Filter.score(detections, min_score=0.60)
		detections = plate.Filter.intersection(detections)


		image_with_boxes = plate.draw_boxes(original_image, detections)
		
		index = 0
		for box in detections['detection_boxes']:
			cropped_image = plate.Crop.crop(top=box[Coordinates.TOP], right=box[Coordinates.RIGHT], bottom=box[Coordinates.BOTTOM], left=box[Coordinates.LEFT])
			Extract_Data().save_jpg(cropped_image, f'cropped_{index}' , '../dataset')
			print(OCR(image=cropped_image).basic_jpg_read())	
			index += 1
		# Extract_Data().save_jpg(image_with_boxes, 'teste', '../dataset')

if __name__ == '__main__':
	print("Inicio")
	Main().run('../dataset/test_images/img_10.jpg')
	print("Fim")