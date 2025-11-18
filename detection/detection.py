import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, Union
import json
import os
import time


def has_gpu_available() -> bool:
    """
    Verifica se há GPU disponível no sistema.
    
    Returns:
        bool: True se GPU está disponível, False caso contrário
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU CUDA detectada: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    
    print("GPU não detectada, usando CPU")
    return False


class YOLODetector:
    """
    Classe para detecção de objetos usando modelos YOLO exportados.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.85):
        """
        Inicializa o detector YOLO.
        
        Args:
            model_path (str): Caminho para o modelo YOLO exportado (.pt, .onnx, .engine, etc.)
            confidence_threshold (float): Threshold de confiança para filtrar detecções (padrão: 0.85)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Carrega o modelo YOLO a partir do caminho especificado.
        
        Raises:
            FileNotFoundError: Se o arquivo do modelo não for encontrado
            Exception: Se houver erro ao carregar o modelo
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"Modelo YOLO carregado com sucesso: {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
        except Exception as e:
            raise Exception(f"Erro ao carregar o modelo: {str(e)}")
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """
        Detecta objetos em um frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            list: Lista de detecções com informações de bounding boxes, confiança e classes
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Use _load_model() primeiro.")
        
        # Medir tempo total
        total_start = time.perf_counter()
        
        # Executa a detecção (YOLO já faz pré e pós-processamento internamente)
        inference_start = time.perf_counter()
        results = self.model(frame, conf=self.confidence_threshold)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Pós-processar resultados
        postprocess_start = time.perf_counter()
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Extrai informações da detecção
                    box = boxes.xyxy[i].cpu().numpy()  # Coordenadas do bounding box
                    confidence = boxes.conf[i].cpu().numpy()  # Confiança
                    class_id = int(boxes.cls[i].cpu().numpy())  # ID da classe
                    
                    detections.append({
                        'bbox': box,
                        'confidence': confidence,
                        'class_id': class_id,
                        'area': (box[2] - box[0]) * (box[3] - box[1])  # Área do bounding box
                    })
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        # Tempo total
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Log do tempo de inferência detalhado
        print(f"[YOLO PyTorch] Inferência: {inference_time:.1f}ms (pós: {postprocess_time:.1f}ms, "
              f"total: {total_time:.1f}ms) - {len(detections)} detecções")
        
        return detections
    
    def get_largest_object_class(self, frame: np.ndarray) -> Optional[int]:
        """
        Retorna o número da classe do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[int]: ID da classe do maior objeto detectado, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection['class_id']
    
    def get_largest_object_info(self, frame: np.ndarray) -> Optional[dict]:
        """
        Retorna informações completas do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[dict]: Dicionário com informações da maior detecção, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Define um novo threshold de confiança.
        
        Args:
            threshold (float): Novo threshold de confiança (0.0 a 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold de confiança deve estar entre 0.0 e 1.0")
        
        self.confidence_threshold = threshold
        print(f"Threshold de confiança atualizado para: {threshold}")


class OpenVINODetector:
    """
    Classe para detecção de objetos usando modelos OpenVINO (otimizado para CPU Intel).
    Compatível com a interface do YOLODetector.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.85):
        """
        Inicializa o detector OpenVINO.
        
        Args:
            model_path (str): Caminho para o diretório do modelo OpenVINO
            confidence_threshold (float): Threshold de confiança para filtrar detecções (padrão: 0.85)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.input_layer = None
        self.output_layer = None
        self.input_shape = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Carrega o modelo OpenVINO a partir do caminho especificado.
        
        Raises:
            FileNotFoundError: Se o diretório do modelo não for encontrado
            Exception: Se houver erro ao carregar o modelo
        """
        try:
            from openvino.runtime import Core
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Diretório do modelo não encontrado: {self.model_path}")
            
            # Procurar arquivos .xml e .bin
            xml_file = None
            for file in os.listdir(self.model_path):
                if file.endswith('.xml'):
                    xml_file = os.path.join(self.model_path, file)
                    break
            
            if not xml_file:
                raise FileNotFoundError(f"Arquivo .xml não encontrado em {self.model_path}")
            
            # Carregar modelo OpenVINO
            ie = Core()
            self.model = ie.compile_model(model=xml_file, device_name="CPU")
            
            # Obter detalhes de entrada e saída
            self.input_layer = self.model.input(0)
            self.output_layer = self.model.output(0)
            self.input_shape = self.input_layer.shape
            
            print(f"Modelo OpenVINO carregado com sucesso: {self.model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Executando em: CPU (OpenVINO)")
            
        except ImportError:
            raise ImportError("OpenVINO não está instalado. Instale com: pip install openvino")
        except FileNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Erro ao carregar o modelo OpenVINO: {str(e)}")
    
    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa a imagem para o formato esperado pelo modelo OpenVINO.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem BGR)
            
        Returns:
            np.ndarray: Imagem pré-processada
        """
        # Obter tamanho de entrada esperado pelo modelo
        _, _, height, width = self.input_shape
        
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        frame_resized = cv2.resize(frame_rgb, (width, height))
        
        # Normalizar para [0, 1]
        frame_resized = frame_resized.astype(np.float32) / 255.0
        
        # Transpor para formato NCHW (Batch, Channels, Height, Width)
        frame_input = np.transpose(frame_resized, (2, 0, 1))
        frame_input = np.expand_dims(frame_input, axis=0)
        
        return frame_input
    
    def _postprocess_outputs(self, outputs: np.ndarray, original_shape: tuple) -> list:
        """
        Pós-processa as saídas do modelo OpenVINO para o formato padrão.
        
        Args:
            outputs (np.ndarray): Saídas brutas do modelo
            original_shape (tuple): Forma original da imagem (height, width)
            
        Returns:
            list: Lista de detecções no formato padrão
        """
        detections = []
        orig_height, orig_width = original_shape[:2]
        
        # Formato típico do YOLO: [1, num_detections, 85] onde 85 = [x, y, w, h, conf, classes...]
        # ou [1, 84, num_detections] dependendo da exportação
        
        output = outputs[0] if len(outputs.shape) == 3 else outputs
        
        # Se formato é [1, 84/85, num_detections], transpor
        if output.shape[1] > output.shape[2]:
            output = output.transpose(0, 2, 1)
        
        output = output[0]  # Remove batch dimension
        
        for detection in output:
            if len(detection) >= 5:
                # Extrair x, y, w, h, confiança
                x_center, y_center, w, h, obj_conf = detection[:5]
                
                # Extrair classes se houver
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    confidence = obj_conf * class_conf
                else:
                    class_id = 0
                    confidence = obj_conf
                
                if confidence >= self.confidence_threshold:
                    # Converter de center format para corner format
                    _, _, input_height, input_width = self.input_shape
                    x1 = int((x_center - w/2) * orig_width / input_width)
                    y1 = int((y_center - h/2) * orig_height / input_height)
                    x2 = int((x_center + w/2) * orig_width / input_width)
                    y2 = int((y_center + h/2) * orig_height / input_height)
                    
                    # Garantir que as coordenadas estão dentro dos limites
                    x1 = max(0, min(x1, orig_width))
                    y1 = max(0, min(y1, orig_height))
                    x2 = max(0, min(x2, orig_width))
                    y2 = max(0, min(y2, orig_height))
                    
                    bbox = np.array([x1, y1, x2, y2])
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        return detections
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """
        Detecta objetos em um frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            list: Lista de detecções com informações de bounding boxes, confiança e classes
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Use _load_model() primeiro.")
        
        # Pré-processar imagem
        preprocess_start = time.perf_counter()
        input_data = self._preprocess_image(frame)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Executar inferência
        inference_start = time.perf_counter()
        outputs = self.model([input_data])[self.output_layer]
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Pós-processar saídas
        postprocess_start = time.perf_counter()
        detections = self._postprocess_outputs(outputs, frame.shape)
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        # Tempo total
        total_time = preprocess_time + inference_time + postprocess_time
        
        # Log do tempo de inferência
        print(f"[OpenVINO CPU] Inferência: {inference_time:.1f}ms (pré: {preprocess_time:.1f}ms, "
              f"pós: {postprocess_time:.1f}ms, total: {total_time:.1f}ms) - {len(detections)} detecções")
        
        return detections
    
    def get_largest_object_class(self, frame: np.ndarray) -> Optional[int]:
        """
        Retorna o número da classe do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[int]: ID da classe do maior objeto detectado, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection['class_id']
    
    def get_largest_object_info(self, frame: np.ndarray) -> Optional[dict]:
        """
        Retorna informações completas do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[dict]: Dicionário com informações da maior detecção, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Define um novo threshold de confiança.
        
        Args:
            threshold (float): Novo threshold de confiança (0.0 a 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold de confiança deve estar entre 0.0 e 1.0")
        
        self.confidence_threshold = threshold
        print(f"Threshold de confiança atualizado para: {threshold}")


class TensorRTDetector:
    """
    Classe para detecção de objetos usando modelos TensorRT (otimizado para GPU NVIDIA).
    Compatível com a interface do YOLODetector.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.85):
        """
        Inicializa o detector TensorRT.
        
        Args:
            model_path (str): Caminho para o modelo TensorRT (.engine)
            confidence_threshold (float): Threshold de confiança para filtrar detecções (padrão: 0.85)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        # TensorRT será carregado via YOLO Ultralytics que tem suporte nativo
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Carrega o modelo TensorRT a partir do caminho especificado.
        
        Raises:
            FileNotFoundError: Se o arquivo do modelo não for encontrado
            Exception: Se houver erro ao carregar o modelo
        """
        try:
            # YOLO Ultralytics suporta TensorRT nativamente
            self.model = YOLO(self.model_path)
            print(f"Modelo TensorRT carregado com sucesso: {self.model_path}")
            print(f"Executando em: GPU (TensorRT)")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
        except Exception as e:
            raise Exception(f"Erro ao carregar o modelo TensorRT: {str(e)}")
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """
        Detecta objetos em um frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            list: Lista de detecções com informações de bounding boxes, confiança e classes
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Use _load_model() primeiro.")
        
        # Medir tempo total
        total_start = time.perf_counter()
        
        # Executa a detecção (TensorRT já faz pré e pós-processamento internamente)
        inference_start = time.perf_counter()
        results = self.model(frame, conf=self.confidence_threshold)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Pós-processar resultados
        postprocess_start = time.perf_counter()
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Extrai informações da detecção
                    box = boxes.xyxy[i].cpu().numpy()  # Coordenadas do bounding box
                    confidence = boxes.conf[i].cpu().numpy()  # Confiança
                    class_id = int(boxes.cls[i].cpu().numpy())  # ID da classe
                    
                    detections.append({
                        'bbox': box,
                        'confidence': confidence,
                        'class_id': class_id,
                        'area': (box[2] - box[0]) * (box[3] - box[1])  # Área do bounding box
                    })
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        # Tempo total
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Log do tempo de inferência detalhado
        print(f"[TensorRT GPU] Inferência: {inference_time:.1f}ms (pós: {postprocess_time:.1f}ms, "
              f"total: {total_time:.1f}ms) - {len(detections)} detecções")
        
        return detections
    
    def get_largest_object_class(self, frame: np.ndarray) -> Optional[int]:
        """
        Retorna o número da classe do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[int]: ID da classe do maior objeto detectado, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection['class_id']
    
    def get_largest_object_info(self, frame: np.ndarray) -> Optional[dict]:
        """
        Retorna informações completas do maior bounding box detectado no frame.
        
        Args:
            frame (np.ndarray): Frame de entrada (imagem)
            
        Returns:
            Optional[dict]: Dicionário com informações da maior detecção, ou None se nenhum objeto for detectado
        """
        detections = self.detect_objects(frame)
        
        if not detections:
            return None
        
        # Encontra a detecção com maior área
        largest_detection = max(detections, key=lambda x: x['area'])
        
        return largest_detection
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Define um novo threshold de confiança.
        
        Args:
            threshold (float): Novo threshold de confiança (0.0 a 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold de confiança deve estar entre 0.0 e 1.0")
        
        self.confidence_threshold = threshold
        print(f"Threshold de confiança atualizado para: {threshold}")


def load_detector(model_path: str = None, confidence_threshold: float = 0.85) -> Union[YOLODetector, OpenVINODetector, TensorRTDetector]:
    """
    Função factory para carregar o detector apropriado automaticamente.
    Detecta GPU e escolhe entre OpenVINO (CPU) ou TensorRT (GPU) de forma transparente.
    
    Args:
        model_path (str): Caminho para o modelo (opcional, detecta automaticamente se None)
        confidence_threshold (float): Threshold de confiança (padrão: 0.85)
        
    Returns:
        Union[YOLODetector, OpenVINODetector, TensorRTDetector]: Instância do detector apropriado
        
    Raises:
        FileNotFoundError: Se o modelo não for encontrado
        ValueError: Se o tipo de modelo não for suportado
    """
    # Detecção automática de GPU
    has_gpu = has_gpu_available()
    
    # Se não foi especificado um modelo, escolher automaticamente
    if model_path is None:
        if has_gpu:
            # Tentar TensorRT primeiro, depois PyTorch GPU
            tensorrt_path = "resources/best.engine"
            pytorch_path = "resources/best.pt"
            
            if os.path.exists(tensorrt_path):
                print("✓ GPU detectada - usando TensorRT para máxima performance")
                return TensorRTDetector(tensorrt_path, confidence_threshold)
            elif os.path.exists(pytorch_path):
                print("✓ GPU detectada - usando YOLO PyTorch com CUDA")
                return YOLODetector(pytorch_path, confidence_threshold)
            else:
                print("⚠ GPU detectada mas modelos GPU não encontrados, usando OpenVINO CPU")
                model_path = "resources/best_openvino_model"
        else:
            # Sem GPU: usar OpenVINO (otimizado para CPU)
            model_path = "resources/best_openvino_model"
            if not os.path.exists(model_path):
                # Fallback para PyTorch CPU
                model_path = "resources/best.pt"
                print("⚠ OpenVINO não encontrado, usando YOLO PyTorch em CPU")
            else:
                print("✓ CPU detectada - usando OpenVINO para máxima performance em CPU")
    
    # Se foi especificado um modelo, detectar o tipo pela extensão ou estrutura
    if os.path.isdir(model_path):
        # É um diretório, provavelmente OpenVINO
        print(f"Detectado modelo OpenVINO: {model_path}")
        return OpenVINODetector(model_path, confidence_threshold)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Detectar tipo de modelo pela extensão
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    
    if ext == '.engine':
        print(f"Detectado modelo TensorRT: {model_path}")
        return TensorRTDetector(model_path, confidence_threshold)
    elif ext in ['.pt', '.onnx']:
        print(f"Detectado modelo YOLO (PyTorch/ONNX): {model_path}")
        return YOLODetector(model_path, confidence_threshold)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {ext}. Use .pt, .onnx, .engine ou diretório OpenVINO")


def load_yolo_model(model_path: str, confidence_threshold: float = 0.85) -> YOLODetector:
    """
    Função utilitária para carregar um modelo YOLO.
    
    Args:
        model_path (str): Caminho para o modelo YOLO exportado
        confidence_threshold (float): Threshold de confiança (padrão: 0.85)
        
    Returns:
        YOLODetector: Instância do detector carregado
    """
    return YOLODetector(model_path, confidence_threshold)


def load_json_file(file_path: str) -> dict:
    """
    Carrega um arquivo JSON.
    
    Args:
        file_path (str): Caminho para o arquivo JSON
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def get_object_info(frame: np.ndarray, detector, json_file) -> Optional[dict]:
    """
    Obtém informações do objeto detectado usando a maior detecção (maior área).
    Funciona tanto para mapeamentos (dict) quanto para listas, onde o índice
    corresponde ao class_id do modelo.
    
    Args:
        frame (np.ndarray): Frame de entrada
        detector: Instância do detector (YOLODetector, OpenVINODetector ou TensorRTDetector)
        json_file: Estrutura com informações dos objetos (dict ou list)
    """
    class_id = detector.get_largest_object_class(frame)
    if class_id is None:
        return None
    
    # Suporta dict {class_id: obj} e list [obj_0, obj_1, ...]
    if isinstance(json_file, dict):
        # Suporta chaves inteiras ou strings ("0", "1", ...)
        if class_id in json_file:
            return json_file.get(class_id)
        return json_file.get(str(class_id))
    if isinstance(json_file, list):
        return json_file[class_id] if 0 <= class_id < len(json_file) else None
    
    return None
