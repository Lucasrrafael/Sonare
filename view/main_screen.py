"""
Aplicação principal - ponto de entrada do sistema
"""

import sys
import os
import tkinter as tk
from tkinter import Canvas
import platform
import json
import argparse
from PIL import Image, ImageTk

# Adicionar a raiz do projeto ao PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        """Configura tela cheia sem barras"""
        # Remover decorações da janela para tela cheia
        self.root.overrideredirect(True)
        
        # Configurar tela cheia
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        
        # Configurar para ficar sempre no topo
        self.root.attributes('-topmost', True)
        
        # Configurar cursor
        self.root.configure(cursor='hand2')
    
    def setup_ui(self):
        """Configura a interface inicial"""
        # Carregar imagem como background
        self.load_background_image()
        
        # Canvas para sobrepor elementos
        self.canvas = Canvas(
            self.root,
            highlightthickness=0,
            bg='white'  # Background temporário
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Adicionar imagem como background se carregada
        if hasattr(self, 'background_photo') and self.background_photo:
            self.background_item = self.canvas.create_image(0, 0, image=self.background_photo, anchor='nw')
            print("Imagem carregada como background!")
        else:
            print("Imagem não foi carregada, usando background azul")
            self.canvas.configure(bg='#2c3e50')
        
        # Relógio no canto superior esquerdo
        self.clock_label = self.canvas.create_text(
            50, 50,
            text="10:45",
            font=('DejaVu Sans', 60, 'bold'),
            fill='white',
            anchor='nw'
        )
        
        # Configurar clique em qualquer lugar para iniciar
        self.canvas.bind('<Button-1>', lambda e: self.start_application())
        
        # Configurar teclas para sair
        self.root.bind('<Escape>', lambda e: self.quit_application())
        self.root.bind('<F4>', lambda e: self.quit_application())
        self.root.bind('<F12>', lambda e: self.quit_application())
        
        # Iniciar atualização do relógio
        self.update_clock()
    
    def load_background_image(self):
        """Carrega a imagem JPG como background"""
        try:
            # Caminho para a imagem JPG
            img_path = os.path.join(project_root, 'view', 'assets', 'Frame 444_page-0001.jpg')
            print(f"Tentando carregar imagem: {img_path}")
            print(f"Arquivo existe: {os.path.exists(img_path)}")
            
            if os.path.exists(img_path):
                print("Carregando imagem JPG...")
                # Carregar imagem diretamente
                img = Image.open(img_path)
                
                print("Imagem carregada com sucesso!")
                # Redimensionar para a tela
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                print(f"Redimensionando para: {screen_width}x{screen_height}")
                
                self.background_image = img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
                self.background_photo = ImageTk.PhotoImage(self.background_image)
                print("Imagem carregada e convertida para PhotoImage!")
            else:
                print(f"Arquivo de imagem não encontrado: {img_path}")
                self.background_photo = None
        except Exception as e:
            print(f"Erro ao carregar imagem: {e}")
            import traceback
            traceback.print_exc()
            self.background_photo = None
    
    def update_clock(self):
        """Atualiza o relógio a cada segundo"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M")
        self.canvas.itemconfig(self.clock_label, text=current_time)
        # Agendar próxima atualização em 1 segundo
        self.root.after(1000, self.update_clock)
    
    def start_application(self):
        """Inicia a aplicação"""
        print("Iniciando câmera...")
        # Abrir tela da câmera
        camera_screen = CameraScreen(self.root, debug=self.debug, conf=self.conf, display_seconds=self.display_seconds)
    
    def return_to_main(self):
        """Volta para a tela principal"""
        # Limpar todos os widgets da janela
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Recriar a tela inicial
        self.setup_ui()
        
        print("Voltando para a tela principal")
    
    def quit_application(self):
        """Para a aplicação completamente"""
        print("Parando aplicação...")
        # Fechar aplicação principal
        self.root.quit()
        self.root.destroy()
    
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
