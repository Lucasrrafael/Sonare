"""
Aplicação principal - ponto de entrada do sistema
"""

import tkinter as tk
from camera_screen import CameraScreen

class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Aplicação Principal")
        self.root.state('zoomed')  # Abrir em tela cheia
        self.root.configure(bg='#2c3e50')
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface inicial"""
        # Título
        title_label = tk.Label(
            self.root,
            text="Aplicação Principal",
            font=('Arial', 32, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=100)
        
        # Subtítulo
        subtitle_label = tk.Label(
            self.root,
            text="Sistema em desenvolvimento",
            font=('Arial', 16),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack(pady=20)
        
        # Botão Iniciar
        start_button = tk.Button(
            self.root,
            text="INICIAR",
            font=('Arial', 20, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=50,
            pady=20,
            command=self.start_application,
            relief='flat',
            cursor='hand2'
        )
        start_button.pack(pady=50)
    
    def start_application(self):
        """Inicia a aplicação"""
        print("Iniciando câmera...")
        # Abrir tela da câmera
        camera_screen = CameraScreen(self.root)
    
    def run(self):
        """Executa a aplicação"""
        self.root.mainloop()

def main():
    """Função principal"""
    print("Iniciando aplicação...")
    app = MainApp()
    app.run()

if __name__ == "__main__":
    main()
