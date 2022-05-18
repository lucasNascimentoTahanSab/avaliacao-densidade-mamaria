from tkinter import *


class MetricsScreen:
    def __init__(self, fit_time, predict_time, accuracy, specificity=0):
        self.metrics_screen = Tk()
        self.metrics_screen.title('Métricas')
        self.metrics_screen.geometry('500x200')
        self.metrics_screen.configure(background='#dde')

        label = Label(
            self.metrics_screen,
            text=f'Tempo de Execução Treino: {"{:.2f}".format((fit_time * 1000))}ms\n \
            Tempo de Execução Teste: {"{:.2f}".format((predict_time * 1000))}ms\n \
            Acurácia: {accuracy}\n \
            Especificidade: {specificity}'
        )

        label.place(anchor="center")
        label.pack(pady=10)

        self.metrics_screen.mainloop()
