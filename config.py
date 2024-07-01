import sys, re
import os
from typing import Union
from cv2 import FONT_HERSHEY_DUPLEX, getTextSize
from pathlib import Path

class Dir:
    PROJECT = Path.cwd()
    
    MODELS = PROJECT / 'models'
    SRC = PROJECT / 'src'

class Model:
    _NAME = Path('model')
    TFLITE = _NAME.with_suffix('.tflite')
    ONNX = _NAME.with_suffix('.onnx')
    ONNX_TRAINED = _NAME.with_name(_NAME.name + '_trained.onnx')
    
    YOLOv8n = Dir.MODELS / 'yolov8n'
    YOLOv5s = Dir.MODELS / 'yolov5s'
    DRONE = Dir.MODELS / 'drone'
    RESNET = Dir.MODELS / 'resnet'
    RESNET_QDQ = Dir.MODELS / 'resnetQDQ'
    
    DRONE_TFLITE = DRONE / TFLITE
    DRONE_ONNX = DRONE / ONNX
    DRONE_ONNX_TRAINED = DRONE / ONNX_TRAINED
    
    YOLOv8n_TFLITE = YOLOv8n / TFLITE
    YOLOv8n_ONNX = YOLOv8n / ONNX
    YOLOv8n_ONNX_TRAINED = YOLOv8n / ONNX_TRAINED
    
    RESNET_ONNX = RESNET / ONNX
    RESNET_QDQ_ONNX = RESNET_QDQ / ONNX
    
    @staticmethod 
    def get_tflite_path(path: Path):
        return path.with_suffix('.tflite')
        
    @staticmethod 
    def get_onnx_path(path: Path):
        return path.with_suffix('.onnx')
    
    @staticmethod 
    def get_torch_path(path: Path):
        return path.with_suffix('.pt')
    
    @staticmethod 
    def get_onnx_trained_path(path: Path):
        return path.with_name(path.with_suffix('').name + '_trained.onnx')
    
    @staticmethod
    def gen_model_paths_from_folder(folder: Path, name_wout_suffix: str):
        base_path = folder / name_wout_suffix
        tflite_path = Model.get_tflite_path(base_path)
        onnx_path = Model.get_onnx_path(base_path)
        torch_path = Model.get_torch_path(base_path)
        return folder, tflite_path, onnx_path, torch_path
        
    @staticmethod
    def gen_model_paths(model_path: Union[Path | str]):
        if type(model_path) == str:
            model_path = Path(model_path)
        folder = model_path.parent
        name = model_path.with_suffix('').name
        return Model.gen_model_paths_from_folder(folder, name)
        

class Text:
    FACE = FONT_HERSHEY_DUPLEX
    THICKNESS = 1
    FRACTION = 0.02
    COLOR = (255, 255, 255)
    
    @staticmethod
    def get_scale(shape):
        (w, h), baseline = getTextSize('text', Text.FACE, 1.0, Text.THICKNESS)
        pixelsPerScale = h+baseline
        pixels = shape[1]*Text.FRACTION 
        
        return pixels / pixelsPerScale
    
    
        