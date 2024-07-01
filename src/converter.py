from pathlib import Path
import sys
sys.path.extend([str(Path(__file__).parents[1])])

from pathlib import Path
import tensorflow as tf

import torch.nn as nn
from onnx import ModelProto

from config import Model

from src.base_converter import BaseConverter
from src.converter_impl import TFLiteConverter, ONNXConverter, TorchConverter
from src.utils import OnnxIO

class Converter:
    def __init__(self, model=None, model_path=None, load_intermediate_models_from_disk=True, input_shape=(1, 320, 320, 3)):
        '''
        `model`: [ tf.lite.Interpreter | onnx.ModelProto | torch.nn.Module ]
        
        `model_path`: [ str | Path ]. Path to model to load or to save.
        
        `load_intermediate_models_from_disk`: bool. Flag to use models from disk with 
        corresponding names with `model_path`, if they exists.
        
        `input_shape`: tuple. Used only for torch models to convert them into onnx.
        '''
        if model is not None:
            self._base = self._init_from_model(model, model_path, input_shape, load_intermediate_models_from_disk)
        elif model_path is not None:
            self._base = self._init_from_path(Path(model_path), input_shape, load_intermediate_models_from_disk)
        else:
            raise RuntimeError('At least one of parameters `model` or `model_path` must be provided')
            
            
    def _init_from_path(self, model_path: Path, input_shape, load_intermediate_models_from_disk):
        match model_path.suffix:
            case '.tflite':
                return TFLiteConverter.from_model_path(model_path, load_intermediate_models_from_disk)
            case '.onnx':
                return ONNXConverter.from_model_path(model_path, load_intermediate_models_from_disk)
            case '.pt':
                return TorchConverter.from_model_path(model_path, input_shape, load_intermediate_models_from_disk)
            case _:
                raise NotImplemented(f'Converter from {model_path.suffix} is not implemented')
            
            
    def _init_from_model(self, model, model_path, input_shape, load_intermediate_models_from_disk):
        if isinstance(model, tf.lite.Interpreter):
            return TFLiteConverter.from_model(model, model_path, load_intermediate_models_from_disk)
        if isinstance(model, ModelProto):
            return ONNXConverter.from_model(model, model_path, load_intermediate_models_from_disk)
        if isinstance(model, nn.Module):
            return TorchConverter.from_model(model, model_path, input_shape, load_intermediate_models_from_disk)
        
        
    def to_tflite(self):
        return self._base.to_tflite()
    
    def to_onnx(self):
        return self._base.to_onnx()
    
    def to_torch(self):
        return self._base.to_torch()
    
    @property
    def onnx_path(self):
        return self._base.onnx_path
    
    @property
    def torch_path(self):
        return self._base.torch_path
    
    @property
    def tflite_path(self):
        return self._base.tflite_path
    
    
if __name__ == '__main__':
    converter = Converter(
        model_path=Model.DRONE_TFLITE,
        load_intermediate_models_from_disk=False,
    )
    torch_model = converter.to_torch()
    input_shape = OnnxIO(converter.to_onnx()).input_shape
    
    converter = Converter(
        model=torch_model,
        input_shape=input_shape,
        load_intermediate_models_from_disk=False,
    )
    converter.to_tflite()