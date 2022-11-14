from PIL import Image
import numpy as np
from matplotlib import cm

class Image_Crop:

	def __init__(self, image):
		self.image = image

	def crop(self, image=None, top=0, right=None, bottom=None, left=0):
		if image is not None:
			self.image = image
		
		image = Image.fromarray(self.image)
		
		w, h = image.size
		if right is None:
			right = w
		if bottom is None:
			bottom = h

		# Normalize coordinates
		top, bottom = top * h, bottom * h
		right, left = right * w, left * w

		return np.array(image.crop((left, top, right, bottom)))


if __name__ == '__main__':
	print('Opss... Wrong way.')
	print('This is class is not acessable directly... you should turn back.')