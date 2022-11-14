import random
import pathlib
import csv 
import argparse

class Select_Random_Data:

	## Abre arquivo CSV e retorna vetor com as linhas
	def open_file(path, remove_header=False):
		assert pathlib.Path(path).suffix == 'csv'

		file = open(path)
		csv_file = csv.reader(file)   
		data = []

		for row in csv_file: 
		  data.append(row)
		file.close()
		
		##remove cabecalho
		if(remove_header):
			data.pop(0)

		return data

	## Separa aleatoriamente e escreve CSV com os dados passados
	def random_separate_data(data, filename, output_path, percent=1, cabecalho=''):
		data_path = open(f"{output_path}/{filename}.csv","w+", newline='')
		write_csv = csv.writer(data_path)
		
		if (cabecalho != ''):
			write_csv.writerow(cabecalho)
		
		for img in range(int(len(data)*percent)):
			pos = random.randint(0, len(data) - 1)
			write_csv.writerow(data[pos])
			data.pop(pos)
		
		data_path.close()

if __name__ == '__main__':

	cabecalho = "filename","width","height","class","xmin","ymin","xmax","ymax"

	parser = argparse.ArgumentParser(description='Random Data Generator')
	parser.add_argument('-t', '--training', nargs='?', type=int, default=0.7, dest='training_percent', help='Percentual de imagens que serão utilizadas para treinamento da I.A. O valor deve ser indicado entre 0 e 1. Ao utlizar o valor 0, não será gerado arquivo de saída.')
	parser.add_argument('-v', '--validate', nargs='?', type=int, default=0.2, dest='validation_percent', action='store', help='Percentual de imagens que serão utilizadas para validação do treinamento da I.A. O valor deve ser indicado entre 0 e 1. Ao utlizar o valor 0, não será gerado arquivo de saída.')
	parser.add_argument('-i', '--inference', nargs='?', type=int, default=0.1, dest='inference_percent', action='store', help='Percentual de imagens que serão utilizadas para teste do treinamento realizado da I.A. O valor deve ser indicado entre 0 e 1. Ao utlizar o valor 0, não será gerado arquivo de saída.')
	parser.add_argument('--header', nargs='*', default=cabecalho, dest='header', action='store', help='Header para o arquivo. Deve ser informado separado por espaços.')
	parser.add_argument('-rh', '--remove_header', nargs=1, type=bool, default=True, dest='remove_header', action='store', help='Remove a primeira linha do arquivo original para não causar erro. Se necessário a adição de Header, deverá ser informado no argumento -h ou --header.')
	parser.add_argument('-f', '--file', required=True, nargs='?', dest='input_file', action='store', help='Diretório do arquivo que contém as informações a serem distribuídas.')
	parser.add_argument('-o', '--out', required=True, nargs='?', dest='out_path', action='store', help='Dirtetório para saída do arquivo. Será gerado arquivo com o nome "Treino, Validacao e Teste com a extensão CSV no local de saída indicado.')

	args = parser.parse_args()
	random_data = Select_Random_Data()

	
	data = Select_Random_Data().open_file(args.input_file, remove_header=args.remove_header)
	if (args.training_percent > 0):
		random_data.random_separate_data(data, args.input_file, args.out_path, args.training_percent, args.header);
	if (args.validation_percent > 0):
		random_data.random_separate_data(data, args.input_file, args.out_path, args.validation_percent, args.header);
	if (args.inference_percent > 0):
		random_data.random_separate_data(data, args.input_file, args.out_path, args.inference_percent, args.header);