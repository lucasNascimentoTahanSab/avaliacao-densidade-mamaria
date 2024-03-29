from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo, showwarning

import os

from PIL import ImageTk, Image

import mahotas
import numpy
import pandas

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import time

from sklearn.utils import resample
from descriptorsScreen import DescriptorsScreen
from metricsScreen import MetricsScreen

from imageDescribed import ImageDescribed

descriptors = [
    'energy_or_uniformity',
    'contrast',
    'correlation',
    'variance',
    'texture_homogeneity',
    'sum_average',
    'sum_variance',
    'sum_entropy',
    'entropy',
    'diff_variance',
    'diff_entropy',
    'info_measures_of_correlation',
    'info_measures_of_correlation',
    'birads'
]


class Application:
    def __init__(self, root):
        self.root = root
      
        self.selected_image = None
        self.shades_of_gray = 32

        self.birads_1_images = list()
        self.birads_2_images = list()
        self.birads_3_images = list()
        self.birads_4_images = list()

        self.birads_1_path = 'images/1/'
        self.birads_2_path = 'images/2/'
        self.birads_3_path = 'images/3/'
        self.birads_4_path = 'images/4/'
        
        self.birads_1_dataframe = pandas.DataFrame()
        self.birads_2_dataframe = pandas.DataFrame()
        self.birads_3_dataframe = pandas.DataFrame()
        self.birads_4_dataframe = pandas.DataFrame()

        self.file_types = (
            ('PNG Files', '*.png'),
            ('JPG Files', '*.jpg')
        )
        
        self.radiuses = [1, 2, 4, 8, 16]
        self.number_haralick_descriptors = 13
        
        self.train_descriptors = pandas.DataFrame() 
        self.train_birads = pandas.DataFrame()
         
        self.test_descriptors = pandas.DataFrame() 
        self.test_birads = pandas.DataFrame()
        
        self.trained = False
        
        # Hiperparâmetros para o modelo SVM
        self.C = [0.1, 1, 10, 100, 1000]
        self.gamma = ['scale', 'auto']
        self.kernel = ['rbf', 'poly', 'linear']

        # Melhor combinação de hiperparâmetros para o modelo SVM encontrada 
        self.svm_classifier = svm.SVC(C=self.C[2], gamma=self.gamma[0], kernel=self.kernel[2])

    # Método responsável pela abertura de página para seleção da imagem a ser 
    # analisada e posicionamento da imagem na plataforma.
    def open_file(self):
        file_name = filedialog.askopenfilename(
            title='Escolha uma imagem',
            filetypes=self.file_types,
            initialdir=os.path.normpath('images/')
        )

        if not file_name:
            return

        # Criação do objeto Image a partir do path para o arquivo escolhido.
        self.selected_image = Image.open(file_name)
        photo_image = ImageTk.PhotoImage(self.selected_image)

        # Criação do objeto Frame para posicionamento da imagem na tela.
        frame = Frame(self.root, width=128, height=128)
        frame.pack()
        
        frame.place(anchor='center', relx=0.5, rely=0.5)

        # Criação do objeto Label para posicionamento da imagem no frame na tela.
        label = Label(frame, image=photo_image)
        label.image = photo_image
        label.pack()

    # Método responsável pela leitura dos diretórios de cada classe BIRADS para 
    # posterior treino e teste.
    def read_directories(self):
        showinfo(
            'Lendo diretórios...',
            'Aguarde enquanto a leitura do diretório de imagens de treino/teste é efetuada.'
        )

        self.birads_1_images = self.get_described_images_from_directory(self.birads_1_path, 1)
        self.birads_2_images = self.get_described_images_from_directory(self.birads_2_path, 2)
        self.birads_3_images = self.get_described_images_from_directory(self.birads_3_path, 3)
        self.birads_4_images = self.get_described_images_from_directory(self.birads_4_path, 4)

        showinfo(
            'Leitura de diretórios efetuada',
            'A leitura do diretório de imagens foi efetuada com sucesso!'
        )

    # Método responsável pela leitura e retorno das imagens do diretório informado 
    # e mapeamento para a classe BIRADS indicada.
    def get_described_images_from_directory(self, path, birads_class):
        images = list()

        for file in os.listdir(path):
            if file.endswith('.png') or file.endswith('.jpg'):
                image = Image.open(path + file)
                image_described = ImageDescribed(image, birads=birads_class)
                images.append(image_described)

        return images

    # Método responsável pela construção e treino do classificador SVM, apresentação 
    # da acurácia do modelo e matriz de confusão.
    def train_svm_classifier(self):
        if not self.birads_1_images:
            return showwarning(
                'Não é possível treinar o classificador',
                'É necessário realizar a leitura do diretório de imagens antes de treinar o classificador.'
            )

        showinfo(
            'Treinando Classificador',
            'Aguarde enquanto o classificador é treinado.'
        )

        self.calculate_images_descriptors()
        self.fill_classifier_sets()
        
        self.calculate_svm_train_time()
        self.calculate_svm_test_time()

        self.plot_svm_confusion_matrix()

        self.calculate_svm_classifier_accuracy()
        self.calculate_svm_classifier_specificity()

        self.display_classifier_metrics()

    # Método responsável pelo preenchimento dos dataframes de descritores
    # de cada BIRADS registrada.
    def calculate_images_descriptors(self):
        self.birads_1_dataframe = self.get_birads_dataframe(self.birads_1_images)
        self.birads_2_dataframe = self.get_birads_dataframe(self.birads_2_images)
        self.birads_3_dataframe = self.get_birads_dataframe(self.birads_3_images)
        self.birads_4_dataframe = self.get_birads_dataframe(self.birads_4_images)
        
    # Método responsável pela geração dos dataframes de descritores
    # das imagens de acordo com as imagens recebidas.
    def get_birads_dataframe(self, birads_images):
        described_birads_images = self.get_described_images_by_birads(birads_images)
        
        return pandas.DataFrame(described_birads_images, columns=descriptors)

    # Método responsável pela obtenção dos descritores das imagens 
    # recebidas por classe BIRADS. 
    def get_described_images_by_birads(self, described_images):
        images_descriptors = []

        for described_image in described_images:
            described_image.descriptors = self.get_image_descriptors(described_image.image)
            image_descriptors = described_image.descriptors
            image_descriptors.append(described_image.birads)
            images_descriptors.append(image_descriptors)

        return images_descriptors

    # Método responsável pelo cálculo dos descritores de Haralick da 
    # imagem recebida, retornando as médias de cada descritor por raio 
    # (determinados no contrutor).
    def get_image_descriptors(self, image):
        resampled_image = self.get_resampled_image_shades_of_gray(image)
        haralick_descriptors_by_radius: numpy.ndarray = numpy.empty(
            shape=(len(self.radiuses), self.number_haralick_descriptors)
        )

        for i in range(len(self.radiuses)):
            haralick_descriptors: numpy.ndarray = numpy.array(
                mahotas.features.haralick(resampled_image, distance=self.radiuses[i])
            )
            
            haralick_descriptors = haralick_descriptors.mean(axis=0)
            haralick_descriptors_by_radius[i] = haralick_descriptors

        return haralick_descriptors_by_radius.mean(axis=0).tolist()
      
    # Método responsável pelo cálculo e obtenção da imagem recebida
    # reamostrada nos tons de cinza definidos no construtor.
    def get_resampled_image_shades_of_gray(self, image):
        resample_ratio = int(round(255 / self.shades_of_gray))
        resampled_image = numpy.round(numpy.array(image) / resample_ratio)
        
        return resampled_image.astype(numpy.uint8)

    def resampling_shades_of_gray_interface(self):

        self.resample_screen = Tk()
        self.slider = Scale(self.resample_screen, from_=2, to=32, orient=HORIZONTAL)
        self.slider.pack()

        Button(self.resample_screen, text='Resample', command=self.resampling_shades_of_gray).pack() 

        self.resample_screen.mainloop()

    def resampling_shades_of_gray(self):
        self.shades_of_gray = self.slider.get()
        if self.selected_image is not None:
            resample_ratio = int(round(255 / self.shades_of_gray))
            resampled_image = numpy.round(numpy.array(self.selected_image) / resample_ratio)
            plt.imshow(resampled_image, cmap='Greys', interpolation='none')
            plt.show()   
                
        self.resample_screen.destroy()
        
    # Método responsável pelo preenchimento dos conjuntos de descritores e
    # classes BIRADS de treino (75%) e teste (25%).
    def fill_classifier_sets(self):
        splitted_data = self.get_splitted_data()
        train_set = pandas.concat(splitted_data['train'])
        test_set = pandas.concat(splitted_data['test'])

        self.train_descriptors = self.get_set_descriptors(train_set)
        self.train_birads = self.get_set_birads(train_set)
        self.test_descriptors = self.get_set_descriptors(test_set) 
        self.test_birads = self.get_set_birads(test_set)
      
    # Método responsável pela obtenção dos dados separados em treino (75%) 
    # e teste (25%) para treinamento e testes do classificador.
    def get_splitted_data(self):
        birads_1_train, birads_1_test = train_test_split(self.birads_1_dataframe, test_size=0.25, random_state=42)
        birads_2_train, birads_2_test = train_test_split(self.birads_2_dataframe, test_size=0.25, random_state=42)
        birads_3_train, birads_3_test = train_test_split(self.birads_3_dataframe, test_size=0.25, random_state=42)
        birads_4_train, birads_4_test = train_test_split(self.birads_4_dataframe, test_size=0.25, random_state=42)
        
        return {
          'train': [birads_1_train, birads_2_train, birads_3_train, birads_4_train],
          'test': [birads_1_test, birads_2_test, birads_3_test, birads_4_test]
        }
        
    # Método responsável pela obtenção dos descritores, excluindo a coluna
    # de classificação BIRADS.
    def get_set_descriptors(self, set):
        return set.drop('birads', axis=1)
      
    # Método responsável pela obtenção de todas as classificações BIRADS
    # presentes no conjunto recebido.
    def get_set_birads(self, set):
        return set['birads']
        
    # Método responsável pelo treinamento do classificador a partir das variáveis
    # de treino estabelecidas.
    def calculate_svm_train_time(self):
        start_fit_model = time.time()
        
        self.svm_classifier.fit(self.train_descriptors, self.train_birads)
        self.trained = True
        
        end_fit_model = time.time()

        self.fit_time = end_fit_model - start_fit_model
        
        
    # Método responsável por testar o classificador a partir das variáveis de
    # teste estabelecidas.
    def calculate_svm_test_time(self):
        start_predict = time.time()
        
        self.predicted_birads = self.svm_classifier.predict(self.test_descriptors)
        
        end_predict = time.time()

        self.predict_time = end_predict - start_predict
      
    # Método responsável pelo cálculo da acurácia do SVM.
    def calculate_svm_classifier_accuracy(self):
        self.accuracy = accuracy_score(self.test_birads, self.predicted_birads) * 100

    # Método responsável pelo cálculo da especificidade do SVM a partir dos valores
    # verdadeiros negativos, falsos positivos, falsos negativos e verdadeiros positivos.
    def calculate_svm_classifier_specificity(self):
        # Média das linhas sem contar os valores da diagonal principal (verdadeiros 
        # positivos).
        false_positive = self.confusion_matrix_instance.sum(axis=0) - numpy.diag(self.confusion_matrix_instance)  

        # Média das colunas sem contar os valores da diagonal principal (verdadeiros 
        # positivos).
        false_negative = self.confusion_matrix_instance.sum(axis=1) - numpy.diag(self.confusion_matrix_instance)

        # Diagonal principal da matriz de confusão.
        true_positive = numpy.diag(self.confusion_matrix_instance)
        
        true_negative = self.confusion_matrix_instance.sum() - (false_positive + false_negative + true_positive)

        # Calcula a especificade por classe (para obter a especificidade para o modelo 
        # como um todo é preciso calcular a média).
        self.specificity = true_negative/(true_negative + false_positive)
        self.specificity = self.specificity.mean(axis=0)

    def display_classifier_metrics(self):
        self.metrics_screen = MetricsScreen(self.fit_time, self.predict_time, self.accuracy, self.specificity)
        
    # Método responsável pela geração e apresentação da matriz de confusão, 
    # calculada com base nos testes aplicados ao classificador. 
    def plot_svm_confusion_matrix(self):
        self.confusion_matrix_instance = confusion_matrix(
            self.test_birads, 
            self.predicted_birads, 
            labels=self.svm_classifier.classes_
        )
        confusion_matrix_display = ConfusionMatrixDisplay(
          confusion_matrix=self.confusion_matrix_instance,
          display_labels=self.svm_classifier.classes_
        )
        
        confusion_matrix_display.plot()
        plt.title('Matriz de Confusão')
        plt.show()
        
    # Método responsável por apresentar descritores de Haralick para a imagem 
    # selecionada pelo usuário.
    def get_selected_image_descriptors(self):
        if self.selected_image is None:
            return showwarning(
                'Nenhuma imagem selecionada',
                'É preciso que uma imagem seja selecionada para a realização dos cálculos dos descritores.'
            )
        
        start = time.time()        

        list_of_descriptors = self.get_image_descriptors(self.selected_image)

        end = time.time()
        execution_time = end - start

        self.descriptors_screen = DescriptorsScreen(list_of_descriptors, execution_time)
        
    # Método responsável pela classificação da imagem selecionada pelo usuário
    # nas BIRADS disponíveis.
    def get_selected_image_classification(self):
        if self.selected_image is None:
            return showwarning(
                'Nenhuma imagem selecionada',
                'É preciso que uma imagem seja selecionada para a realização da classificação.'
            )

        if self.trained:
            image_descriptors = self.get_image_descriptors(self.selected_image)
            image_descriptors = numpy.array(image_descriptors)
            predicted_birads = self.svm_classifier.predict(image_descriptors.reshape(1, -1))
            
            showinfo(
                'Classificação da Imagem',
                f'A imagem foi classificada como BIRADS {predicted_birads}'
            )

        else:
            return showinfo(
                'O modelo ainda não foi treinado',
                'É preciso treinar o modelo antes de classificar a imagem selecionada'
            )         
