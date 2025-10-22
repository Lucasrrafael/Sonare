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
import glob

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
    def __init__(self, debug: bool = False, conf: float = 0.85, display_seconds: float = 6.0, carousel_time: int = 3):
        self.root = tk.Tk()
        self.root.title("Aplicação Principal")
        self.debug = debug
        self.conf = conf
        self.display_seconds = display_seconds
        self.carousel_time = carousel_time
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
        # Canvas para sobrepor elementos
        self.canvas = Canvas(
            self.root,
            highlightthickness=0,
            bg='#2c3e50'  # Background temporário
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Carregar e iniciar carrossel de backgrounds
        self.load_carousel_backgrounds()
        
        # Adicionar primeira imagem como background se carregada
        if hasattr(self, 'background_images') and self.background_images:
            self.background_item = self.canvas.create_image(0, 0, image=self.background_images[0], anchor='nw')
            print("Primeira imagem carregada como background!")
        else:
            print("Nenhuma imagem carregada, usando background azul")
        
        # Relógio no canto superior esquerdo (estilo da imagem)
        self.clock_label = self.canvas.create_text(
            110, 150,
            text="10:45",
            font=('Outfit', 160, 'bold'),
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
        
        # Iniciar carrossel de backgrounds
        self.start_background_carousel()
    
    def update_clock(self):
        """Atualiza o relógio a cada segundo"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M")
        self.canvas.itemconfig(self.clock_label, text=current_time)
        # Agendar próxima atualização em 1 segundo
        self.root.after(1000, self.update_clock)
    
    def load_carousel_backgrounds(self):
        """Carrega todas as imagens da pasta assets como backgrounds"""
        try:
            # Buscar todas as imagens PNG e JPG
            assets_dir = os.path.join(project_root, 'view', 'assets')
            png_files = glob.glob(os.path.join(assets_dir, '*.png'))
            jpg_files = glob.glob(os.path.join(assets_dir, '*.jpg'))
            
            # Combinar todas as imagens
            all_files = sorted(png_files + jpg_files)
            
            print(f"Encontradas {len(all_files)} imagens para o carrossel de backgrounds")
            
            self.background_images = []
            self.background_photos = []
            
            # Obter dimensões da tela
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            for img_file in all_files:
                try:
                    img = Image.open(img_file)
                    # Redimensionar para o tamanho da tela
                    img_resized = img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img_resized)
                    self.background_photos.append(photo)
                    self.background_images.append(photo)
                    print(f"Carregada como background: {os.path.basename(img_file)}")
                except Exception as e:
                    print(f"Erro ao carregar {img_file}: {e}")
            
            # Variáveis para controle do carrossel
            self.current_bg_index = 0
            self.bg_animation_offset = 0
            self.bg_animation_speed = 30  # Pixels por frame (começa rápido)
            self.bg_is_animating = False
            
        except Exception as e:
            print(f"Erro ao carregar imagens do carrossel: {e}")
            self.background_images = []
    
    def start_background_carousel(self):
        """Inicia o carrossel de backgrounds"""
        if not self.background_images:
            return
        
        # Agendar primeira troca após 3 segundos
        self.root.after(self.carousel_time * 1000, self.change_background)
    
    def change_background(self):
        """Muda para o próximo background com animação"""
        if not self.background_images:
            return
        
        # Próximo índice
        self.current_bg_index = (self.current_bg_index + 1) % len(self.background_images)
        
        # Atualizar background com fade/transição suave
        self.canvas.itemconfig(
            self.background_item,
            image=self.background_images[self.current_bg_index]
        )
        
        print(f"Background alterado para índice {self.current_bg_index}")
        
        # Agendar próxima troca após o tempo configurado
        self.root.after(self.carousel_time * 1000, self.change_background)
    
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
    parser.add_argument("-t", "--carousel-time", type=int, default=3, help="Duração (s) de cada imagem no carrossel")
    args = parser.parse_args()

    print("Iniciando aplicação...")
    app = MainApp(debug=args.debug, conf=args.conf, display_seconds=args.display_seconds, carousel_time=args.carousel_time)
    app.run()

if __name__ == "__main__":
    main()
