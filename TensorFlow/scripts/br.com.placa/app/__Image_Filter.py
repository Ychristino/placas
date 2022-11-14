import cv2

class Image_Filter:

	## Converte a imagem para preto e branco, a fim de facilitar a leitura dos caracteres.
	## Função importante para uma boa leitura, se necessário a utilização de imagens, verificar uso.
	def grayscale(self, image):
		return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	## Remove ruídos da imagem, buscando otimizar a leitura quado necessário em imagens de baixa qualidade.
	## Recebe os parâmetros de arquivo de entrada e nível de ruído para remoção.
	def remove_noise(self, image, noise):
		return cv2.medianBlur(image, noise)

	## Otimiza leitura da imagem por meio de "separação".
	## Busca truncar separação de valores para otimizar o reconhecimentos dos caracteres.
	def threshhold(self, image, th=0.5):
		return cv2.threshold(image, th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

if __name__ == '__main__':
	print('Opss... Wrong way.')
	print('This is class is not acessable directly... you should turn back.')