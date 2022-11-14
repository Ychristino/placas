import cv2
import pytesseract
import numpy as np
from __Image_Filter import Image_Filter

class OCR:
	def __init__(self, file_path=None, image=None):
		if file_path is not None:
			self.image = cv2.imread(file_path)
		elif image is not None:
			self.image = image
		else:
			print("An image or a file_path should be passed.")

	## Realiza a conversão da imagem para texto.
	def read(self, image):
		pytesseract.pytesseract.tesseract_cmd = (r'./virtual_env/scripts/Tesseract-OCR/tesseract')
		text = pytesseract.image_to_string(image)
		return text

	## Realiza a leitura com parametros padrões, geralmente, o suficiente para a conversão...
	## Processo por binzarização, converte a imagem para gray_scale e após, "arredondando" os valores pelo thresh hold.
	## Para otimizar a leitura ou realizar a leitura de arquivos de menor qualidade, utilizar uma leitura mais detalhada.
	## A leitura mais detalhada pode ser realizada chamando as funções como "remove_noise" para ajustar a qualidade imagem.
	def basic_jpg_read(self):
		self.image = Image_Filter().grayscale(self.image)
		self.image = Image_Filter().threshhold(self.image)

		return self.read(self.image)
		
if __name__ == '__main__':
	print('Opss... Wrong way.')
	print('This is class is not acessable directly... you should turn back.')