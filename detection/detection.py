import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple
import json


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
        
        # Executa a detecção
        results = self.model(frame, conf=self.confidence_threshold)
        
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

def get_object_info(frame: np.ndarray, detector: YOLODetector, json_file) -> Optional[dict]:
    """
    Obtém informações do objeto detectado usando a maior detecção (maior área).
    Funciona tanto para mapeamentos (dict) quanto para listas, onde o índice
    corresponde ao class_id do modelo.
    
    Args:
        frame (np.ndarray): Frame de entrada
        detector (YOLODetector): Instância do detector YOLO
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
