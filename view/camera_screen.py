"""
Tela da câmera com OpenCV
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
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
from detection.detection import load_detector, get_object_info

# Definir fonte padrão que suporta acentos
DEFAULT_FONT = ('DejaVu Sans', 10)
DEFAULT_FONT_BOLD = ('DejaVu Sans', 10, 'bold')
DEFAULT_FONT_LARGE = ('DejaVu Sans', 16)
DEFAULT_FONT_LARGE_BOLD = ('DejaVu Sans', 16, 'bold')
DEFAULT_FONT_TITLE = ('DejaVu Sans', 24, 'bold')

class CameraScreen:
    def __init__(self, parent_window, debug: bool = False, conf: float = 0.85, display_seconds: float = 6.0, model_path: str = None, speech_speed: float = 1.3):
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
        self.speech_speed = float(speech_speed)  # Velocidade da fala (1.0 = normal, 1.3 = 30% mais rápido)
        self.debug_mode = bool(debug)  # Modo debug para desenhar bounding boxes
        self.conf_threshold = float(conf)
        self.display_seconds = max(0.5, float(display_seconds))
        self._detection_locked_until = 0.0
        
        # Thresholds customizados por classe (sobrescrevem o threshold básico)
        # Baseado no mapeamento do products.json:
        # ID 0: Coca-Cola, ID 1: Fanta, ID 2: Feijão Carioca, ID 3: Feijão Fradinho
        # ID 4: Feijão Preto, ID 5: Leite Condensado, ID 6: Leite Italac, ID 7: Leite Piracanjuba
        # ID 8: Macarrão, ID 9: Miojo (filtrado), ID 10: Pomarola, ID 11: Óleo
        self.class_thresholds = {
            0: 0.81,   # Coca-Cola
            1: 0.84,   # Fanta
            2: 0.65,   # Feijão Carioca
            3: 0.71,   # Feijão Fradinho
            4: 0.80,   # Feijão Preto
            5: 0.82,   # Leite Condensado
            6: 0.87,   # Leite Italac
            7: 0.72,   # Leite Piracanjuba
            8: 0.80,   # Macarrão
            10: 0.86,  # Pomarola
            11: 0.84   # Óleo
        }
        
        self.create_camera_window()
        
        # Calcular caminho raiz do projeto
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Ajustar model_path se fornecido e for relativo
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        # Se model_path for None, load_detector usará caminhos relativos à raiz do projeto
        
        # Inicializar detector (escolhe automaticamente OpenVINO/TensorRT baseado em GPU)
        try:
            self.detector = load_detector(model_path=model_path, confidence_threshold=self.conf_threshold)
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            self.detector = None
        
        # Carregar produtos para mapear class_id -> produto
        try:
            products_path = os.path.join(project_root, 'resources', 'products', 'products.json')
            with open(products_path, 'r', encoding='utf-8') as f:
                self.products_data = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar products.json: {str(e)}")
            self.products_data = []
    
    def passes_class_threshold(self, class_id, confidence):
        """Verifica se a confiança passa no threshold customizado da classe.
        
        Args:
            class_id: ID da classe detectada
            confidence: Confiança da detecção (0.0 a 1.0)
            
        Returns:
            bool: True se passa no threshold, False caso contrário
        """
        # Se a classe tem threshold customizado, usar ele
        if class_id in self.class_thresholds:
            threshold = self.class_thresholds[class_id]
            passes = confidence >= threshold
            if not passes:
                print(f"[THRESHOLD] Classe {class_id} rejeitada: conf={confidence:.3f} < threshold={threshold:.3f}")
            return passes
        
        # Caso contrário, usar threshold básico
        passes = confidence >= self.conf_threshold
        if not passes:
            print(f"[THRESHOLD] Classe {class_id} rejeitada: conf={confidence:.3f} < threshold básico={self.conf_threshold:.3f}")
        return passes
    
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
        
        # Configurar fechamento da janela
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera)
        
        # Configurar teclas
        # ESC: fecha apenas a janela da câmera e volta à tela principal
        self.camera_window.bind('<Escape>', lambda e: self.close_camera())
        
        # Q ou q: fecha a aplicação inteira
        self.camera_window.bind('<q>', lambda e: self.quit_application())
        self.camera_window.bind('<Q>', lambda e: self.quit_application())
        
        # Iniciar câmera automaticamente
        self.start_camera()
    
    def start_camera(self):
        """Inicia a câmera (prioritariamente webcam externa)"""
        try:
            # No Windows, usar DirectShow para abertura mais rápida
            use_dshow = platform.system() == "Windows"
            
            # Tentar webcam externa primeiro (índice 1)
            print("Tentando abrir webcam externa (índice 1)...")
            if use_dshow:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(1)
            
            # Verificar rapidamente se a câmera está disponível
            if self.cap.isOpened():
                # Testar se consegue ler um frame
                ret, _ = self.cap.read()
                if ret:
                    print("Usando webcam externa")
                else:
                    print("Webcam externa não funcional, tentando câmera integrada...")
                    self.cap.release()
                    if use_dshow:
                        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    else:
                        self.cap = cv2.VideoCapture(0)
            else:
                print("Webcam externa não encontrada. Tentando câmera integrada (índice 0)...")
                self.cap.release()
                if use_dshow:
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(0)
                
                if not self.cap.isOpened():
                    self.video_label.config(text="Erro: Nenhuma câmera encontrada")
                    return
                else:
                    print("Usando câmera integrada do notebook")
            
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
            height, width = frame.shape[:2]
            frame_area = height * width
            
            detections = self.detector.detect_objects(frame)
            for det in detections:
                bbox = det['bbox']
                if hasattr(bbox, 'astype'):
                    x1, y1, x2, y2 = bbox.astype(int)
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                conf = float(det['confidence'])
                class_id = det['class_id']

                # Filtrar classe 9 (Miojo)
                if class_id == 9:
                    print(f"[DEBUG] Classe 9 (Miojo) filtrada - ignorada")
                    continue
                
                # Verificar threshold customizado da classe
                if not self.passes_class_threshold(class_id, conf):
                    continue

                # Calcular área da bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                
                # Filtrar bboxes que ocupam mais de 85% da tela
                if bbox_area > 0.85 * frame_area:
                    print(f"[DEBUG] Bbox filtrada: {bbox_area / frame_area * 100:.1f}% da tela")
                    continue

                # Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label com nome do produto e confiança
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

                # Criar label com nome (ou ID) e confiança
                base_label = product_name if product_name else f"ID {class_id}"
                label = f"{base_label} {conf:.2f}"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(y1 - 10, th + 5)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 4), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        except Exception as e:
            print(f"Erro ao desenhar debug boxes: {str(e)}")
        return frame
    
    def detect_and_show_product(self, frame):
        """Executa a detecção e exibe o produto na UI."""
        if not hasattr(self, 'detector') or self.detector is None:
            return frame
        try:
            # Travar novas detecções até o fim do período de exibição
            if time.time() < self._detection_locked_until:
                return frame

            # Executar detecção
            detections = self.detector.detect_objects(frame)
            # DEBUG: quantidade de detecções por frame
            print(f"[DETECT] {len(detections)} detecções neste frame")
            if not detections:
                return frame

            # Filtrar bboxes que ocupam mais de 85% da tela e classe 9 (Miojo)
            height, width = frame.shape[:2]
            frame_area = height * width
            filtered_detections = []
            
            for det in detections:
                class_id = int(det.get('class_id', -1))
                conf = float(det.get('confidence', 0.0))
                
                # Filtrar classe 9 (Miojo)
                if class_id == 9:
                    print(f"[DETECT] Classe 9 (Miojo) filtrada - ignorada")
                    continue
                
                # Verificar threshold customizado da classe
                if not self.passes_class_threshold(class_id, conf):
                    continue
                
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    if hasattr(bbox, 'astype'):
                        x1, y1, x2, y2 = bbox.astype(int)
                    else:
                        x1, y1, x2, y2 = map(int, bbox)
                    
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    
                    # Filtrar se maior que 85% da tela
                    if bbox_area <= 0.85 * frame_area:
                        filtered_detections.append(det)
                    else:
                        print(f"[DETECT] Bbox filtrada: {bbox_area / frame_area * 100:.1f}% da tela")
            
            if not filtered_detections:
                print(f"[DETECT] Todas as {len(detections)} detecções foram filtradas")
                return frame
            
            # Pegar a maior detecção do frame atual (após filtro)
            largest = max(filtered_detections, key=lambda d: d.get('area', 0))
            class_id = int(largest.get('class_id', -1))

            # Mapear class_id para produto no JSON
            product_info = None
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
        
        # Converter frame para PIL para desenhar texto com suporte a Unicode
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # Carregar fonte com suporte a Unicode
        try:
            # Tentar carregar fonte DejaVu Sans do projeto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            font_path = os.path.join(project_root, 'resources', 'fonts', 'dejavu-sans.bold.ttf')
            font_name = ImageFont.truetype(font_path, 48)  # Reduzido de 60 para 48
            font_price = ImageFont.truetype(font_path, 42)  # Reduzido de 50 para 42
        except Exception as e:
            print(f"Erro ao carregar fonte DejaVu: {e}")
            # Fallback para fonte padrão
            try:
                font_name = ImageFont.truetype("arial.ttf", 48)
                font_price = ImageFont.truetype("arial.ttf", 42)
            except:
                font_name = ImageFont.load_default()
                font_price = ImageFont.load_default()
        
        # Nome do produto (muito mais acima)
        name_text = str(product['name'])  # Garantir que é string
        # Debug: imprimir caracteres para verificar encoding
        if 'Macarrão' in name_text or 'ã' in name_text:
            print(f"DEBUG - Nome do produto: {name_text}")
            print(f"DEBUG - Bytes: {name_text.encode('utf-8')}")
        
        name_bbox = draw.textbbox((0, 0), name_text, font=font_name)
        name_width = name_bbox[2] - name_bbox[0]
        name_x = center_x - name_width // 2
        name_y = center_y - 250
        
        # Desenhar sombra preta para o nome
        draw.text((name_x + 3, name_y + 3), name_text, font=font_name, fill=(0, 0, 0))
        # Desenhar nome em branco
        draw.text((name_x, name_y), name_text, font=font_name, fill=(255, 255, 255))
        
        # Preço do produto (mais próximo do nome)
        price_text = product['price']
        price_bbox = draw.textbbox((0, 0), price_text, font=font_price)
        price_width = price_bbox[2] - price_bbox[0]
        price_x = center_x - price_width // 2
        price_y = center_y - 180
        
        # Desenhar sombra preta para o preço
        draw.text((price_x + 3, price_y + 3), price_text, font=font_price, fill=(0, 0, 0))
        # Desenhar preço em laranja (RGB)
        draw.text((price_x, price_y), price_text, font=font_price, fill=(255, 165, 0))
        
        # Converter de volta para OpenCV
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
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
    
    def quit_application(self):
        """Fecha a aplicação inteira"""
        print("Fechando aplicação...")
        self.stop_camera()
        
        # Parar qualquer áudio que esteja sendo reproduzido
        try:
            pygame.mixer.music.stop()
            self.is_playing_audio = False
        except:
            pass
        
        # Fechar a aplicação completamente
        if hasattr(self.parent, 'quit'):
            self.parent.quit()
        if hasattr(self.parent, 'destroy'):
            self.parent.destroy()
    
    def __del__(self):
        """Destrutor"""
        if self.cap:
            self.cap.release()
