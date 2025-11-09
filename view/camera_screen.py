"""
Tela da câmera com OpenCV
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import pygame
from gtts import gTTS
import tempfile
import numpy as np
import soundfile as sf
import platform
import json
from product_selector import ProductSelector
from detection.detection import YOLODetector
from detection.detection import get_object_info

# Definir fonte padrão que suporta acentos
DEFAULT_FONT = ('DejaVu Sans', 10)
DEFAULT_FONT_BOLD = ('DejaVu Sans', 10, 'bold')
DEFAULT_FONT_LARGE = ('DejaVu Sans', 16)
DEFAULT_FONT_LARGE_BOLD = ('DejaVu Sans', 16, 'bold')
DEFAULT_FONT_TITLE = ('DejaVu Sans', 24, 'bold')

class CameraScreen:
    def __init__(self, parent_window, debug: bool = False, conf: float = 0.85, display_seconds: float = 6.0):
        self.parent = parent_window
        self.cap = None
        self.camera_active = False
        self.video_thread = None
        self.current_product = None
        self.product_display_timer = None
        self.product_overlay = None  # Para armazenar informações do produto
        
        # Inicializar pygame para reprodução de áudio
        pygame.mixer.init()
        self.is_playing_audio = False
        self.speech_speed = 1.3  # Velocidade da fala (1.0 = normal, 1.3 = 30% mais rápido)
        self.debug_mode = bool(debug)  # Modo debug para desenhar bounding boxes
        self.conf_threshold = float(conf)
        self.display_seconds = max(0.5, float(display_seconds))
        self._detection_locked_until = 0.0
        
        self.create_camera_window()
        
        # Calcular caminho raiz do projeto
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Inicializar YOLODetector (usa mesmo modelo do main)
        try:
            model_path = os.path.join(project_root, 'resources', 'best.pt')
            self.detector = YOLODetector(model_path=model_path, confidence_threshold=self.conf_threshold)
        except Exception as e:
            print(f"Erro ao carregar modelo YOLO: {str(e)}")
            self.detector = None
        
        # Carregar produtos para mapear class_id -> produto
        try:
            products_path = os.path.join(project_root, 'resources', 'products', 'products.json')
            with open(products_path, 'r') as f:
                self.products_data = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar products.json: {str(e)}")
            self.products_data = []
    
    def set_speech_speed(self, speed):
        """Define a velocidade da fala (1.0 = normal, 1.5 = 50% mais rápido, 0.8 = 20% mais lento)"""
        self.speech_speed = max(0.5, min(2.0, speed))  # Limitar entre 0.5x e 2.0x
        
        # Atualizar display na interface
        if hasattr(self, 'speed_display'):
            self.speed_display.config(text=f"{self.speech_speed:.1f}x")
        
        print(f"Velocidade da fala ajustada para: {self.speech_speed}x")
    
    def speak_text(self, text):
        """Reproduz o texto em áudio usando gTTS e pygame"""
        try:
            if text and text.strip() and not self.is_playing_audio:
                self.is_playing_audio = True
                print(f"Iniciando reprodução de áudio: {text}")
                
                # Executar em thread separada para não bloquear a interface
                def play_audio():
                    try:
                        # Gerar áudio usando gTTS com configuração baseada na velocidade
                        # Para velocidades mais altas, usar slow=False, para mais baixas usar slow=True
                        use_slow = self.speech_speed < 1.0
                        tts = gTTS(text=text, lang='pt', slow=use_slow)
                        
                        # Criar arquivo temporário
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                            temp_path = temp_file.name
                            tts.save(temp_path)
                        
                        # Se a velocidade for diferente de 1.0, ajustar usando numpy e soundfile
                        if self.speech_speed != 1.0:
                            try:
                                # Carregar áudio
                                audio_data, sample_rate = sf.read(temp_path)
                                
                                # Ajustar velocidade usando interpolação
                                if self.speech_speed > 1.0:
                                    # Acelerar: reduzir número de amostras
                                    new_length = int(len(audio_data) / self.speech_speed)
                                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                                else:
                                    # Desacelerar: aumentar número de amostras
                                    new_length = int(len(audio_data) / self.speech_speed)
                                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                                
                                # Salvar áudio ajustado
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                                    adjusted_path = temp_file.name
                                    sf.write(adjusted_path, audio_data, sample_rate)
                                
                                # Reproduzir áudio ajustado
                                pygame.mixer.music.load(adjusted_path)
                                
                                # Limpar arquivo temporário original
                                os.unlink(temp_path)
                                temp_path = adjusted_path
                                
                            except Exception as e:
                                print(f"Erro ao ajustar velocidade, usando áudio original: {str(e)}")
                                pygame.mixer.music.load(temp_path)
                        else:
                            pygame.mixer.music.load(temp_path)
                        
                        # Reproduzir áudio
                        pygame.mixer.music.play()
                        
                        # Configurar volume
                        pygame.mixer.music.set_volume(1.0)
                        
                        # Aguardar o áudio terminar
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        # Limpar arquivo temporário
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                        print(f"Áudio reproduzido com sucesso (velocidade {self.speech_speed}x): {text}")
                        
                    except Exception as e:
                        print(f"Erro na reprodução de áudio: {str(e)}")
                    finally:
                        self.is_playing_audio = False
                
                # Iniciar thread de áudio
                audio_thread = threading.Thread(target=play_audio, daemon=True)
                audio_thread.start()
                
            elif self.is_playing_audio:
                print("Áudio já está sendo reproduzido, aguardando...")
                
        except Exception as e:
            print(f"Erro ao reproduzir áudio: {str(e)}")
            self.is_playing_audio = False
    
    def maximize_camera_window(self):
        """Maximiza a janela da câmera em tela cheia sem barras"""
        # Remover decorações da janela para tela cheia
        self.camera_window.overrideredirect(True)
        
        # Configurar tela cheia
        self.camera_window.geometry(f"{self.camera_window.winfo_screenwidth()}x{self.camera_window.winfo_screenheight()}+0+0")
        
        # Configurar para ficar sempre no topo
        self.camera_window.attributes('-topmost', True)
    
    def create_camera_window(self):
        """Cria a janela da câmera"""
        # Criar nova janela
        self.camera_window = tk.Toplevel(self.parent)
        self.camera_window.title("Câmera Ativa")
        self.maximize_camera_window()  # Maximizar janela (compatível com Linux e Windows)
        self.camera_window.configure(bg='#2c3e50')
        
        # Título (sobreposto no canto superior)
        title_label = tk.Label(
            self.camera_window,
            text="Câmera Ativa",
            font=DEFAULT_FONT_TITLE,
            fg='white',
            bg='#2c3e50'
        )
        title_label.place(relx=0.5, rely=0.05, anchor='center')
        
        # Frame para o vídeo (ocupar toda a tela)
        self.video_frame = tk.Frame(
            self.camera_window,
            bg='black',
            relief='flat'
        )
        self.video_frame.pack(fill='both', expand=True)
        
        # Label para o vídeo (ocupar toda a área)
        self.video_label = tk.Label(
            self.video_frame,
            text="Iniciando câmera...",
            font=DEFAULT_FONT_LARGE,
            fg='white',
            bg='black'
        )
        self.video_label.pack(fill='both', expand=True)
        
        # Remover canvas - vamos usar OpenCV para desenhar sobre o vídeo
        
        # Frame para botões (sobreposto no canto inferior)
        buttons_frame = tk.Frame(self.camera_window, bg='#2c3e50')
        buttons_frame.place(relx=0.5, rely=0.95, anchor='center')
        
        # Botão Parar
        self.stop_button = tk.Button(
            buttons_frame,
            text="PARAR CÂMERA",
            font=DEFAULT_FONT_LARGE_BOLD,
            bg='#e74c3c',
            fg='white',
            padx=30,
            pady=10,
            command=self.stop_camera,
            relief='flat',
            cursor='hand2'
        )
        self.stop_button.pack(side='left', padx=10)
        
        # Botão Selecionar Produto
        self.select_product_button = tk.Button(
            buttons_frame,
            text="SELECIONAR PRODUTO",
            font=DEFAULT_FONT_LARGE_BOLD,
            bg='#3498db',
            fg='white',
            padx=30,
            pady=10,
            command=self.open_product_selector,
            relief='flat',
            cursor='hand2'
        )
        self.select_product_button.pack(side='left', padx=10)
        
        # Botão Fechar
        self.close_button = tk.Button(
            buttons_frame,
            text="FECHAR",
            font=DEFAULT_FONT_LARGE_BOLD,
            bg='#95a5a6',
            fg='white',
            padx=30,
            pady=10,
            command=self.close_camera,
            relief='flat',
            cursor='hand2'
        )
        self.close_button.pack(side='left', padx=10)
        
        # Botão Debug
        # Modo debug controlado apenas por parâmetro de linha de comando
        
        # Frame para controles de velocidade
        speed_frame = tk.Frame(self.camera_window, bg='#2c3e50')
        speed_frame.place(relx=0.02, rely=0.95, anchor='w')
        
        # Label de velocidade
        speed_label = tk.Label(
            speed_frame,
            text="Velocidade:",
            font=DEFAULT_FONT_BOLD,
            fg='white',
            bg='#2c3e50'
        )
        speed_label.pack(side='left', padx=5)
        
        # Botão mais lento
        self.slower_button = tk.Button(
            speed_frame,
            text="-",
            font=DEFAULT_FONT_LARGE_BOLD,
            bg='#e67e22',
            fg='white',
            width=3,
            height=1,
            command=lambda: self.set_speech_speed(self.speech_speed - 0.1),
            relief='flat',
            cursor='hand2'
        )
        self.slower_button.pack(side='left', padx=2)
        
        # Label da velocidade atual
        self.speed_display = tk.Label(
            speed_frame,
            text=f"{self.speech_speed:.1f}x",
            font=DEFAULT_FONT_BOLD,
            fg='#f39c12',
            bg='#2c3e50'
        )
        self.speed_display.pack(side='left', padx=5)
        
        # Botão mais rápido
        self.faster_button = tk.Button(
            speed_frame,
            text="+",
            font=DEFAULT_FONT_LARGE_BOLD,
            bg='#e67e22',
            fg='white',
            width=3,
            height=1,
            command=lambda: self.set_speech_speed(self.speech_speed + 0.1),
            relief='flat',
            cursor='hand2'
        )
        self.faster_button.pack(side='left', padx=2)
        
        # Configurar fechamento da janela
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera)
        
        # Configurar teclas para voltar à tela inicial
        self.camera_window.bind('<Escape>', lambda e: self.close_camera())
        self.camera_window.bind('<F4>', lambda e: self.close_camera())
        self.camera_window.bind('<F12>', lambda e: self.close_camera())
        
        # Iniciar câmera automaticamente
        self.start_camera()
    
    def start_camera(self):
        """Inicia a câmera (prioritariamente webcam externa)"""
        try:
            # Tentar webcam externa primeiro (índice 1)
            print("Tentando abrir webcam externa (índice 1)...")
            self.cap = cv2.VideoCapture(1)
            
            # Se não conseguir abrir ou não estiver disponível, tentar câmera integrada
            if not self.cap.isOpened():
                print("Webcam externa não encontrada. Tentando câmera integrada (índice 0)...")
                self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    self.video_label.config(text="Erro: Nenhuma câmera encontrada")
                    return
                else:
                    print("Usando câmera integrada do notebook")
            else:
                print("Usando webcam externa")
            
            self.camera_active = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.video_label.config(text=f"Erro: {str(e)}")
    
    def video_loop(self):
        """Loop do vídeo"""
        while self.camera_active:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Redimensionar frame mantendo proporção
                        height, width = frame.shape[:2]
                        
                        # Obter tamanho da janela
                        window_width = self.video_frame.winfo_width()
                        window_height = self.video_frame.winfo_height()
                        
                        # Se a janela ainda não foi renderizada, usar tamanhos padrão
                        if window_width <= 1 or window_height <= 1:
                            window_width = 1200
                            window_height = 800
                        
                        # Calcular escala mantendo proporção
                        scale = min(window_width/width, window_height/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        
                        # Redimensionar mantendo proporção
                        frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Rodar detecção e exibir produto correspondente
                        frame = self.detect_and_show_product(frame)

                        # Se debug estiver ativo, desenhar boxes/labels do YOLO
                        frame = self.draw_debug_boxes(frame)
                        
                        # Adicionar informações do produto sobre o frame
                        frame = self.add_product_overlay(frame)
                        
                        # Converter BGR para RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Converter para PIL Image
                        pil_image = Image.fromarray(rgb_frame)
                        photo = ImageTk.PhotoImage(pil_image)
                        
                        # Atualizar label (verificar se ainda existe)
                        try:
                            if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                                self.video_label.config(image=photo, text="")
                                self.video_label.image = photo  # Manter referência
                            else:
                                break  # Janela foi fechada
                        except tk.TclError:
                            break  # Widget foi destruído
                            
            except Exception as e:
                print(f"Erro no loop do vídeo: {str(e)}")
                break
            
            time.sleep(0.03)  # ~30 FPS
    
    def stop_camera(self):
        """Para a câmera"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="", text="Câmera parada")

    def draw_debug_boxes(self, frame):
        """Desenha bounding boxes e confidências se o modo debug estiver ativo."""
        if not self.debug_mode or not hasattr(self, 'detector') or self.detector is None:
            return frame
        try:
            detections = self.detector.detect_objects(frame)
            for det in detections:
                bbox = det['bbox']
                if hasattr(bbox, 'astype'):
                    x1, y1, x2, y2 = bbox.astype(int)
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                conf = float(det['confidence'])
                class_id = det['class_id']

                # Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label com nome do produto, se disponível
                product_name = None
                try:
                    prod = None
                    if isinstance(self.products_data, dict):
                        prod = self.products_data.get(class_id) or self.products_data.get(str(class_id))
                    elif isinstance(self.products_data, list) and 0 <= class_id < len(self.products_data):
                        prod = self.products_data[class_id]
                    if isinstance(prod, dict) and 'name' in prod:
                        product_name = prod['name']
                except Exception:
                    product_name = None

                label = product_name if product_name else f"ID {class_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(y1 - 10, th + 5)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 4), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        except Exception as e:
            print(f"Erro ao desenhar debug boxes: {str(e)}")
        return frame
    
    def detect_and_show_product(self, frame):
        """Executa a detecção e exibe o produto correspondente na UI atual."""
        if not hasattr(self, 'detector') or self.detector is None:
            return frame
        try:
            # Travar novas detecções até o fim do período de exibição
            if time.time() < self._detection_locked_until:
                return frame

            # Executar detecção uma vez para usar tanto no mapeamento quanto fallback
            detections = self.detector.detect_objects(frame)
            if not detections:
                return frame  # sem logs em caso de ausência de detecções

            # Tentar obter produto via utilitário (suporta dict/list)
            product_info = get_object_info(frame, self.detector, self.products_data)

            # Fallback: se não encontrar no JSON, usar class_id da maior detecção
            if not product_info:
                # maior bounding box
                largest = max(detections, key=lambda d: d.get('area', 0))
                class_id = int(largest.get('class_id', -1))
                if isinstance(self.products_data, dict):
                    product_info = self.products_data.get(class_id) or self.products_data.get(str(class_id))
                elif isinstance(self.products_data, list) and 0 <= class_id < len(self.products_data):
                    product_info = self.products_data[class_id]

            if product_info:
                # Evitar repetição se o mesmo produto já está exibido
                if not self.current_product or self.current_product.get('name') != product_info.get('name'):
                    self.display_product_info(product_info)
                    # Bloquear novas detecções pelo período configurado
                    self._detection_locked_until = time.time() + self.display_seconds
        except Exception as e:
            # Não logar casos de ausência; manter somente erros reais
            pass
        return frame
    
    def open_product_selector(self):
        """Abre o seletor de produtos"""
        ProductSelector(self.camera_window, on_product_selected=self.on_product_selected)
    
    def on_product_selected(self, product):
        """Callback quando um produto é selecionado"""
        print(f"Produto selecionado: {product['name']}")
        self.current_product = product
        self.display_product_info(product)
    
    def display_product_info(self, product):
        """Exibe informações do produto sobre o vídeo"""
        print(f"Exibindo produto: {product['name']}")
        
        # Armazenar informações do produto para desenhar sobre o vídeo
        self.product_overlay = product
        self.current_product = product
        
        # Reproduzir áudio usando chave 'tts'
        if 'tts' in product and product['tts']:
            self.speak_text(product['tts'])
        
        # Agendar para esconder após 3 segundos
        if self.product_display_timer:
            self.camera_window.after_cancel(self.product_display_timer)
        
        self.product_display_timer = self.camera_window.after(int(self.display_seconds * 1000), self.hide_product_info)
        print("Produto exibido com sucesso!")
    
    def add_product_overlay(self, frame):
        """Adiciona informações do produto diretamente sobre o frame do OpenCV"""
        if not self.product_overlay:
            return frame
        
        product = self.product_overlay
        height, width = frame.shape[:2]
        
        # Posição central do frame
        center_x = width // 2
        center_y = height // 3
        
        # Carregar e redimensionar imagem do produto
        try:
            img_path = product['img_path']
            # Usar caminho absoluto baseado na raiz do projeto
            if not os.path.isabs(img_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                img_path = os.path.join(project_root, img_path)
            
            if os.path.exists(img_path):
                # Carregar imagem
                product_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                if product_img is not None:
                    # Redimensionar imagem
                    img_size = 400
                    product_img = cv2.resize(product_img, (img_size, img_size))
                    
                    # Se a imagem tem canal alpha (transparência)
                    if product_img.shape[2] == 4:
                        # Extrair canal alpha
                        alpha = product_img[:, :, 3] / 255.0
                        
                        # Calcular posição para centralizar (bem embaixo dos textos)
                        y1 = center_y + 50 - img_size // 2
                        y2 = y1 + img_size
                        x1 = center_x - img_size // 2
                        x2 = x1 + img_size
                        
                        # Garantir que não saia dos limites
                        y1 = max(0, y1)
                        y2 = min(height, y2)
                        x1 = max(0, x1)
                        x2 = min(width, x2)
                        
                        # Ajustar tamanhos se necessário
                        actual_h = y2 - y1
                        actual_w = x2 - x1
                        
                        if actual_h > 0 and actual_w > 0:
                            # Redimensionar imagem e alpha se necessário
                            if actual_h != img_size or actual_w != img_size:
                                product_img = cv2.resize(product_img, (actual_w, actual_h))
                                alpha = cv2.resize(alpha, (actual_w, actual_h))
                            
                            # Aplicar transparência
                            for c in range(3):
                                frame[y1:y2, x1:x2, c] = (
                                    alpha * product_img[:, :, c] + 
                                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                                )
                    else:
                        # Imagem sem transparência - colocar diretamente (bem embaixo dos textos)
                        y1 = center_y + 50 - img_size // 2
                        y2 = y1 + img_size
                        x1 = center_x - img_size // 2
                        x2 = x1 + img_size
                        
                        y1 = max(0, y1)
                        y2 = min(height, y2)
                        x1 = max(0, x1)
                        x2 = min(width, x2)
                        
                        if y2 > y1 and x2 > x1:
                            frame[y1:y2, x1:x2] = product_img[:y2-y1, :x2-x1]
                            
        except Exception as e:
            print(f"Erro ao carregar imagem: {str(e)}")
        
        # Adicionar texto do nome do produto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5  # Aumentado de 1.0 para 1.5
        color = (255, 255, 255)  # Branco
        thickness = 3  # Aumentado de 2 para 3
        
        # Nome do produto (muito mais acima)
        text_size = cv2.getTextSize(product['name'], font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y - 250  # Movido muito mais para cima
        
        # Adicionar sombra preta para melhor legibilidade
        cv2.putText(frame, product['name'], (text_x + 3, text_y + 3), 
                   font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, product['name'], (text_x, text_y), 
                   font, font_scale, color, thickness)
        
        # Preço do produto (mais próximo do nome)
        price_color = (0, 165, 255)  # Laranja em BGR
        price_text = product['price']
        price_size = cv2.getTextSize(price_text, font, font_scale * 0.9, thickness)[0]
        price_x = center_x - price_size[0] // 2
        price_y = center_y - 200  # Mais próximo do nome
        
        # Adicionar sombra preta para melhor legibilidade
        cv2.putText(frame, price_text, (price_x + 3, price_y + 3), 
                   font, font_scale * 0.9, (0, 0, 0), thickness + 2)
        cv2.putText(frame, price_text, (price_x, price_y), 
                   font, font_scale * 0.9, price_color, thickness)
        
        return frame
    
    def hide_product_info(self):
        """Esconde as informações do produto"""
        self.product_overlay = None
        self.current_product = None
    
    def close_camera(self):
        """Fecha a janela da câmera"""
        self.stop_camera()
        if self.product_display_timer:
            self.camera_window.after_cancel(self.product_display_timer)
        
        # Parar qualquer áudio que esteja sendo reproduzido
        try:
            pygame.mixer.music.stop()
            self.is_playing_audio = False
        except:
            pass
        
        # Se a janela pai tem método return_to_main, chamar ele
        if hasattr(self.parent, 'return_to_main'):
            self.parent.return_to_main()
        else:
            # Fallback: apenas fechar a janela
            self.camera_window.destroy()
    
    def __del__(self):
        """Destrutor"""
        if self.cap:
            self.cap.release()
