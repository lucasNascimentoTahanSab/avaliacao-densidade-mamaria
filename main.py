from email.mime import image
from genericpath import isfile
from gettext import npgettext
from ntpath import join
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import os

from requests import head
import mahotas
from matplotlib.pyplot import axis
# import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.model_selection import train_test_split


class ImageDescriptor:
    def __init__(self, image, descriptors, birads=0):
        self.image = image
        self.descriptors = descriptors
        self.birads = birads


class Application:
    def __init__(self, master=None):
        self.root = Tk()
        self.root.title('Classificador')
        self.root.geometry('500x500')
        self.root.configure(background="#dde")

        self.image_selected = None
        self.shades_of_gray = 32  # valor padrão para a quantidade de tons de cinza das imagens

        self.images_birads_1 = list()
        self.images_birads_2 = list()
        self.images_birads_3 = list()
        self.images_birads_4 = list()

        self.main_menu()
        self.root.mainloop()

    def main_menu(self):
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.file = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='File', menu=self.file)
        self.file.add_command(label='Open File', command=self.open_file)
        self.file.add_separator()
        self.file.add_command(label='Exit', command=self.root.quit)

        self.options = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Opções', menu=self.options)
        self.options.add_command(
            label='Ler diretório de imagens de treino/teste', command=self.read_directory)
        self.options.add_command(
            label='Treinar Classificador', command=self.train_classifier)
        self.options.add_command(
            label='Calcular e exibir características para imagem visualizada', command=self.get_descriptors_from_selected_image)
        self.options.add_command(
            label='Classificar imagem', command=None)

    def open_file(self):
        file_types = (
            ('PNG Files', '*.png'),
            ('JPG Files', '*.jpg')
        )
        # retorna o nome do arquivo que cliquei para abrir
        self.file_name = filedialog.askopenfilename(title='Open an image', filetypes=file_types, initialdir=os.path.normpath("Imagens/"))

        # Retorna caso nenhuma foto tenha sido escolhida
        if self.file_name == '' : return
        
        frame = Frame(self.root, width=128, height=128)
        frame.pack()
        frame.place(anchor='center', relx=0.5, rely=0.5)


        # Criando um objeto tkinter ImageTk
        self.image_selected = Image.open(self.file_name)
        
        image = ImageTk.PhotoImage(self.image_selected)

        # Criando um Label Widget para mostrar a imagem
        label = Label(frame, image=image)
        label.image = image

        # Position image
        label.pack()

        # retorna o nome do arquivo que cliquei para abrir
        #self.file_selected = filedialog.askopenfilename(title='Open an image', initialdir=os.path.normpath("Imagens/"))
        # self.open_selected_file()

    def read_directory(self):
        path1 = 'Imagens/1/'
        path2 = 'Imagens/2/'
        path3 = 'Imagens/3/'
        path4 = 'Imagens/4/'

        showinfo(
                title='Lendo diretórios...',
                message='Aguarde enquanto a leitura do diretório de imagens de treino/teste é efetuada.'
            )

        # Criando um dataset de imagens para cada um dos BIRADS
        for file in os.listdir(path1):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.images_birads_1.append(
                    ImageDescriptor(Image.open(path1+file), [], 1))

        for file in os.listdir(path2):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.images_birads_2.append(
                    ImageDescriptor(Image.open(path2+file), [], 2))

        for file in os.listdir(path3):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.images_birads_3.append(
                    ImageDescriptor(Image.open(path3+file), [], 3))

        for file in os.listdir(path4):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.images_birads_4.append(
                    ImageDescriptor(Image.open(path4+file), [], 4))

        showinfo(
                title='Leitura de diretórios efetuada',
                message='A leitura do diretório de imagens foi efetuada com sucesso!'
            )

    def train_classifier(self):
        self.calculate_descriptors_all_images()

    def calculate_descriptors_all_images(self):
        cols = ["energy_or_uniformity", 
            "contrast",
            "correlation",
            "variance",
            "texture_homogeneity",
            "sum_average",
            "sum_variance",
            "sum_entropy",
            "entropy",
            "diff_variance",
            "diff_entropy",
            "info_measures_of_correlation",
            "info_measures_of_correlation",
            "birads"]

        d_birads_1 = self.calculate_descriptors_for_images(self.images_birads_1)
        df_birads_1 = pd.DataFrame(d_birads_1, columns=cols)

        X_birads_1 = df_birads_1.drop(column=['birads'])
        y_birads_1 = df_birads_1['birads']

        # X_birads_1_train, y_birads_1_train, X_birads_1_test, y_birads_1_test = train_test_split(X_birads_1, y_birads_1, test_size=0.25, random_state=42)
        # model = svm.SVC()
        # model.fit(X_birads_1_train, y_birads_1_train)
        # y_pred = svm.predict(X_birads_1_test)

        d_birads_2 = self.calculate_descriptors_for_images(self.images_birads_2)
        df_birads_2 = pd.DataFrame(d_birads_2, columns=cols)

        d_birads_3 = self.calculate_descriptors_for_images(self.images_birads_3)
        df_birads_3 = pd.DataFrame(d_birads_3, columns=cols)
        
        d_birads_4 = self.calculate_descriptors_for_images(self.images_birads_4)
        df_birads_4 = pd.DataFrame(d_birads_4, columns=cols)

        df_birads_1.to_csv("birads_1.csv", header=True, index=False)
        df_birads_2.to_csv("birads_2.csv", header=True, index=False)
        df_birads_3.to_csv("birads_3.csv", header=True, index=False)
        df_birads_4.to_csv("birads_4.csv", header=True, index=False)

    def calculate_descriptors_for_images(self, imagesDescriptors):
        imag_descrip = []

        for imagesDescriptor in imagesDescriptors:
            imagesDescriptor.descriptors = self.get_descriptors_from_image(imagesDescriptor.image)
            imagesDescriptor_list = imagesDescriptor.descriptors.tolist()
            imagesDescriptor_list.append(imagesDescriptor.birads)
            imag_descrip.append(imagesDescriptor_list)
        
        return imag_descrip
    

    def get_descriptors_from_selected_image(self):
        if self.image_selected is None:
            showinfo(
                title='Nenhuma imagem selecionada',
                message='É preciso que uma imagem seja selecionada para a realização dos cálculos dos descritores.'
            )
            return

        print(self.get_descriptors_from_image(self.image_selected))

    def get_descriptors_from_image(self, image):
        #Reamostrando o número de tons de cinza da imagem
        resample_ratio = int(round(255 / self.shades_of_gray))
        np_img = np.round(np.array(image) / resample_ratio).astype(np.uint8)

        raios = [1, 2, 4, 8, 16]

        num_of_haralick_descriptors = 13
        haralick_descriptors_for_all_radii: np.ndarray = np.empty(
            shape=(len(raios), num_of_haralick_descriptors))

        for i in range(len(raios)):
            haralick_descriptors: np.ndarray = np.array(
                mahotas.features.haralick(np_img, distance=raios[i]))
            haralick_descriptors = haralick_descriptors.mean(
                axis=0)  # Mean of each column
            haralick_descriptors_for_all_radii[i] = haralick_descriptors

        # Mean of each column
        return haralick_descriptors_for_all_radii.mean(axis=0)


Application()
