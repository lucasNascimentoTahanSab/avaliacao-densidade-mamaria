from tkinter import *
from application import Application


class Screen:
    # Construtor responsável pela inicialização das funcionalidades da
    # aplicação e tela do classificador.
    def __init__(self):
        self.root = Tk()
        self.root.title('Classificador')
        self.root.geometry('500x500')
        self.root.configure(background='#dde')

        self.application = Application(self.root)

        self.main_menu()
        self.root.mainloop()

    # Método responsável pela construção do menu principal da aplicação,
    # com sessões de 'Arquivo', 'Opções' e 'Reamostragem'.
    def main_menu(self):
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.build_file_manager()
        self.build_options()
        self.build_resampling()

        self.menu_bar.add_cascade(label='Arquivo', menu=self.file_manager)
        self.menu_bar.add_cascade(label='Opções', menu=self.options)
        self.menu_bar.add_cascade(label='Reamostragem', menu=self.resampling)

    # Método responsável pela construção da sessão 'Arquivo' no menu
    # principal da aplicação.
    def build_file_manager(self):
        self.file_manager = Menu(self.menu_bar, tearoff=0)

        self.file_manager.add_command(
            label='Abrir arquivo',
            command=self.application.open_file
        )

        self.file_manager.add_separator()

        self.file_manager.add_command(
            label='Sair',
            command=self.root.destroy
        )

    # Método responsável pela construção da sessão 'Opções' no menu
    # principal da aplicação.
    def build_options(self):
        self.options = Menu(self.menu_bar, tearoff=0)

        self.options.add_command(
            label='Ler diretório de imagens de treino/teste',
            command=self.application.read_directories
        )

        self.options.add_command(
            label='Treinar classificador',
            command=self.application.train_svm_classifier
        )

        self.options.add_command(
            label='Calcular e exibir características para imagem visualizada',
            command=self.application.get_selected_image_descriptors
        )

        self.options.add_command(
            label='Classificar imagem',
            command=self.application.get_selected_image_classification
        )

    # Método responsável pela construção da sessão 'Reamostragem' no
    # menu principal da aplicação.
    def build_resampling(self):
        self.resampling = Menu(self.menu_bar, tearoff=0)

        self.resampling.add_command(
            label='Reamostrar tons de cinza da imagem selecionada',
            command=self.application.resampling_shades_of_gray_interface
        )


Screen()
