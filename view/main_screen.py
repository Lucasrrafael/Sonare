"""
Aplicação principal - ponto de entrada do sistema
"""

import tkinter as tk
import platform
import json
import argparse
from camera_screen import CameraScreen

# Definir fonte padrão que suporta acentos
DEFAULT_FONT = ('DejaVu Sans', 10)
DEFAULT_FONT_BOLD = ('DejaVu Sans', 10, 'bold')
DEFAULT_FONT_LARGE = ('DejaVu Sans', 16)
DEFAULT_FONT_LARGE_BOLD = ('DejaVu Sans', 16, 'bold')
DEFAULT_FONT_TITLE = ('DejaVu Sans', 32, 'bold')

class MainApp:
    def __init__(self, debug: bool = False, conf: float = 0.85, display_seconds: float = 6.0):
        self.root = tk.Tk()
        self.root.title("Aplicação Principal")
        self.debug = debug
        self.conf = conf
        self.display_seconds = display_seconds
        
        # Maximizar janela (compatível com Linux e Windows)
        self.maximize_window()
        
        self.root.configure(bg='#2c3e50')
        
        self.setup_ui()

    def maximize_window(self):
        """Maximiza a janela de forma compatível com Linux e Windows"""
        system = platform.system()
        
        if system == "Windows":
            # Windows: usar state('zoomed')
            self.root.state('zoomed')
        elif system == "Linux":
            # Linux: usar attributes('-zoomed', True)
            self.root.attributes('-zoomed', True)
        else:
            # macOS ou outros sistemas: tentar maximizar
            try:
                self.root.attributes('-zoomed', True)
            except:
                # Fallback: definir tamanho da tela
                self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
    
    def setup_ui(self):
        """Configura a interface inicial"""
        # Título
        title_label = tk.Label(
            self.root,
            text="Aplicação Principal",
            font=DEFAULT_FONT_TITLE,
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=100)
        
        # Subtítulo
        subtitle_label = tk.Label(
            self.root,
            text="Sistema em desenvolvimento",
            font=DEFAULT_FONT_LARGE,
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack(pady=20)
        
        # Botão Iniciar
        start_button = tk.Button(
            self.root,
            text="INICIAR",
            font=('DejaVu Sans', 20, 'bold'),
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
        camera_screen = CameraScreen(self.root, debug=self.debug, conf=self.conf, display_seconds=self.display_seconds)
    
    def run(self):
        """Executa a aplicação"""
        self.root.mainloop()

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Aplicação Principal Sonare")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug (desenhar bounding boxes)")
    parser.add_argument("-c", "--conf", type=float, default=0.85, help="Threshold de confiança YOLO (0.0 a 1.0)")
    parser.add_argument("-d", "--display-seconds", type=float, default=6.0, help="Duração (s) que cada produto permanece na tela")
    args = parser.parse_args()

    print("Iniciando aplicação...")
    app = MainApp(debug=args.debug, conf=args.conf, display_seconds=args.display_seconds)
    app.run()

if __name__ == "__main__":
    main()
