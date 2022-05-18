from tkinter import *


class DescriptorsScreen:
    def __init__(self, list_of_descriptors=list(), execution_time=0):
        self.descriptors_screen = Tk()
        self.descriptors_screen.title('Descritores')
        self.descriptors_screen.geometry('500x250')
        self.descriptors_screen.configure(background='#dde')

        label = Label(
            self.descriptors_screen,
            text=f' Tempo de execução: {"{:.2f}".format(execution_time)}ms\n \
                    Energia: {list_of_descriptors[0]}\n \
                    Contraste: {list_of_descriptors[1]}\n \
                    Correlação: {list_of_descriptors[2]}\n \
                    Variância: {list_of_descriptors[3]}\n \
                    Homogeneidade: {list_of_descriptors[4]}\n \
                    Soma da média: {list_of_descriptors[5]}\n \
                    Soma da variância: {list_of_descriptors[6]}\n \
                    Soma da entropia: {list_of_descriptors[7]}\n \
                    Entropia: {list_of_descriptors[8]}\n \
                    Diferença de variância: {list_of_descriptors[9]}\n \
                    Diferença de entropia: {list_of_descriptors[10]}\n \
                    Medidas de corerlação 1: {list_of_descriptors[11]}\n \
                    Medidas de corerlação 2: {list_of_descriptors[12]}'
        )

        label.place(anchor="center")
        label.pack(pady=10)

        self.descriptors_screen.mainloop()
