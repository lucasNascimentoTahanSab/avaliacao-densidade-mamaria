import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import os
import mahotas
import cv2
import numpy as np

class Application:
    def __init__(self, master=None):
        self.root = Tk()
        self.root.title('Classificador')
        self.root.geometry('500x500')
        self.root.configure(background="#dde")

        self.image_selected = None

        self.main_menu()
        self.root.mainloop()

    def open_file(self):
        file_types = (
            ('PNG Files', '*.png'),
            ('JPG Files', '*.jpg')
        )
        #filename = filedialog.askopenfilename(filetypes=file_types)
        self.file_name=filedialog.askopenfilename(title='Open an image', initialdir=os.path.normpath("Imagens/"))
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
        '''
        showinfo(
            title='Teste'
            message='Mensagem Teste'
        )
        '''

    def descriptors(self):
        if self.image_selected is None:
            showinfo(
                title='Nenhuma imagem selecionada',
                message='É preciso que uma imagem seja selecionada para a realização dos cálculos dos descritores.'
            )
        
        raios = [1, 2, 4, 8, 16] 
        


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
        self.options.add_command(label='Ler diretório de imagens de treino/teste', command=None)
        self.options.add_command(label='Treinar Classificador', command=None)
        self.options.add_command(label='Calcular e exibir características para imagem visualizada', command=self.descriptors)
        self.options.add_command(label='Classificar imagem', command=None)

Application()


