import tkinter as tk
from tkinter import *

class DescriptorsScreen:
    def __init__(self, list_of_descriptors=list(), execution_time=0):
        # self.descriptors_screen = tk.Toplevel(root)
        self.descriptors_screen = Tk()
        self.descriptors_screen.title('Descritores')
        self.descriptors_screen.geometry('500x250')
        self.descriptors_screen.configure(background='#dde')

        label = Label(self.descriptors_screen,
          text = f' Tempo de execução: {execution_time}\n \
                    Energia: {list_of_descriptors[0]}\n \
                    Contraste: {list_of_descriptors[1]}\n \
                    Correlação: {list_of_descriptors[2]}\n \
                    Variância: {list_of_descriptors[3]}\n \
                    Homogeneidade: {list_of_descriptors[4]}\n \
                    sum_average: {list_of_descriptors[5]}\n \
                    sum_variance: {list_of_descriptors[6]}\n \
                    sum_entropy: {list_of_descriptors[7]}\n \
                    Entropia: {list_of_descriptors[8]}\n \
                    diff_variance: {list_of_descriptors[9]}\n \
                    diff_entropy: {list_of_descriptors[10]}\n \
                    info_measures_of_correlation: {list_of_descriptors[11]}\n \
                    info_measures_of_correlation: {list_of_descriptors[12]}')

        label.place(anchor="center")
        label.pack(pady = 10)
        
        self.descriptors_screen.mainloop()