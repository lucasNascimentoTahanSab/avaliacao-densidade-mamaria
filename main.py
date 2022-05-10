from genericpath import isfile
from gettext import npgettext
from ntpath import join
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import os
import mahotas
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class Application:
    def __init__(self, master=None):
        self.root = Tk()
        self.root.title('Classificador')
        self.root.geometry('500x500')
        self.root.configure(background="#dde")

        self.image_selected = None
        self.shades_of_gray = 32    #valor padrão para a quantidade de tons de cinza das imagens

        self.main_menu()
        self.root.mainloop()

    def open_file(self):
        file_types = (
            ('PNG Files', '*.png'),
            ('JPG Files', '*.jpg')
        )
        #retorna o nome do arquivo que cliquei para abrir
        self.file_name=filedialog.askopenfilename(title='Open an image', filetypes=file_types, initialdir=os.path.normpath("Imagens/"))
        print(self.file_name)
        frame = Frame(self.root, width=128, height=128)
        frame.pack()
        frame.place(anchor='center', relx=0.5, rely=0.5)

        #Criando um objeto tkinter ImageTk
        self.image_selected = Image.open(self.file_name)
        image = ImageTk.PhotoImage(self.image_selected)

        #Criando um Label Widget para mostrar a imagem
        label = Label(frame, image = image)
        label.image = image

        #Position image
        label.pack()
    
        #retorna o nome do arquivo que cliquei para abrir
        #self.file_selected = filedialog.askopenfilename(title='Open an image', initialdir=os.path.normpath("Imagens/"))
        #self.open_selected_file()

    def descriptors(self):
        if self.image_selected is None:
            showinfo(
                title='Nenhuma imagem selecionada',
                message='É preciso que uma imagem seja selecionada para a realização dos cálculos dos descritores.'
            )
        
        #Reamostrar o número de tons de cinza da imagem
        resample_ratio = int(round(255 / self.shades_of_gray))
        np_img = np.round(np.array(self.image_selected) / resample_ratio).astype(np.uint8)

        raios = [1, 2, 4, 8, 16]

        #angles = [0, np.pi/2, np.pi, 5*np.pi/4, 3*np.pi/2]
        #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]

        #im1 = self.image_selected.quantize(8)

        #np_img = np.array(im1).astype(np.uint8)

        num_of_haralick_descriptors = 13
        haralick_descriptors_for_all_radii: np.ndarray = np.empty(
            shape=(len(raios), num_of_haralick_descriptors))

        for i in range(len(raios)):
            haralick_descriptors: np.ndarray = np.array(mahotas.features.haralick(
                np_img, distance=raios[i]
            ))
            haralick_descriptors = haralick_descriptors.mean(axis=0)  # Mean of each column
            haralick_descriptors_for_all_radii[i] = haralick_descriptors

        print(haralick_descriptors_for_all_radii.mean(axis=0))
        #return haralick_descriptors_for_all_radii.mean(axis=0)  # Mean of each column

    def read_directory(self):
        path1 = 'Imagens/1/'
        path2 = 'Imagens/2/'
        path3 = 'Imagens/3/'
        path4 = 'Imagens/4/'

        self.dataset1 = list()
        self.dataset2 = list()
        self.dataset3 = list()
        self.dataset4 = list()

        #Criando um dataset de imagens para cada um dos BIRADS
        for file in os.listdir(path1):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.dataset1.append(Image.open(path1+file))

        for file in os.listdir(path2):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.dataset2.append(Image.open(path2+file))

        for file in os.listdir(path3):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.dataset3.append(Image.open(path3+file))

        for file in os.listdir(path4):
            if file.endswith(".png") or file.endswith(".jpg"):
                self.dataset4.append(Image.open(path4+file))

        #files1 = [file for file in os.listdir(path1) if isfile(join(path1, file))]
        #print(dataset1[:20])
        #print(len(dataset1))

    def train_classifier(self):
        pass

    def main_menu(self):
        self.menu_bar = Menu(self.root) 
        self.root.config(menu = self.menu_bar) 
        
        self.file = Menu(self.menu_bar, tearoff = 0)
        self.menu_bar.add_cascade(label ='File', menu = self.file) 
        self.file.add_command(label ='Open File', command = self.open_file) 
        self.file.add_separator()
        self.file.add_command(label ='Exit', command = self.root.quit) 

        self.options = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Opções', menu = self.options)
        self.options.add_command(label='Ler diretório de imagens de treino/teste', command=self.read_directory)
        self.options.add_command(label='Treinar Classificador', command=self.train_classifier)
        self.options.add_command(label='Calcular e exibir características para imagem visualizada', command=self.descriptors)
        self.options.add_command(label='Classificar imagem', command=None)

Application()


