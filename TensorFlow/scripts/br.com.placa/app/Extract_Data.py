import os
import csv
from matplotlib import pyplot as plt

class Extract_Data:

	def save_jpg(self, image, filename, output_path='./'):
		try:
			if not os.path.exists(os.path.dirname(output_path)):
				os.mkdirs(output_path)

			plt.imsave(f'{output_path}/{filename}.jpg', image)
			return True
		except:
			return False

	def plot_img(self, image):
		plt.imshow(image)
		plt.show()

	def save_csv(self, data, filename, header=[], output_path='./', mode='w'):
		if not os.path.exists(os.path.dirname(output_path)):
			os.mkdirs(output_path)
		
		out_file = open(f'{output_path}/{filename}.csv', mode, newline='')
		csv_file = csv.writer(out_file)

		try:
			csv_file.writerow(header)

			for row in data:
				csv_file.writerow(row)

			return True
		
		except:
			return False
		
		finally:
			out_file.close()
			
if __name__ == '__main__':
    print('Opss... Wrong way.')
    print('This is class is not acessable directly... you should turn back.')