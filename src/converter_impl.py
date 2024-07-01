import onnx

import os
from typing import Union
from pathlib import Path

import onnx
import torch
import tensorflow as tf
# import torch.onnx.verification

import tf2onnx

import cv2
import numpy as np
import torch.nn as nn
from onnx import ModelProto

from config import Model

from abc import ABC, abstractmethod

from src.base_converter import BaseConverter
from src.utils import simplify_onnx, OnnxIO

class TFLiteConverter(BaseConverter):
    # def __init__(self, model_path: Union[Path | str]=None, load_intermediate_models_from_disk=True):
    #     super().__init__(model_path, load_intermediate_models_from_disk)
            
        
    @classmethod
    def from_model(cls, model: tf.lite.Interpreter, model_path: Union[Path | str]=None, load_intermediate_models_from_disk=True):
        raise NotImplementedError('Creating a converter from tf.lite.Interpreter is inconsistent -- use TFLiteConverter.from_path() instead')
    
    
    def _to_onnx(self) -> ModelProto:
        self.to_tflite()
        tf2onnx.convert.from_tflite(
            tflite_path=str(self._tflite_path),
            opset=15,
            output_path=str(self._onnx_path)
        )
        return simplify_onnx(self._onnx_path)
    
    
    def _to_tflite(self) -> tf.lite.Interpreter:
        return tf.lite.Interpreter(str(self._tflite_path))
    
    

class ONNXConverter(BaseConverter):
    # def __init__(self, model_path: Union[Path | str]=None, load_intermediate_models_from_disk=True):
    #     super().__init__(model_path, load_intermediate_models_from_disk)
        
        
    @classmethod
    def from_model(cls, model: ModelProto, model_path: Union[Path | str]=None, load_intermediate_models_from_disk=True):
        self = cls(model_path, load_intermediate_models_from_disk)
        self._onnx_model = model
        onnx.save_model(model, model_path)
        return self
    
    
    def _to_onnx(self) -> ModelProto:
        return onnx.load(self._onnx_path)
    
    
class TorchConverter(BaseConverter):
    def __init__(self, model_path: Union[Path | str]=None, input_shape=(1, 3, 640, 640), load_intermediate_models_from_disk=True):
        super().__init__(model_path, load_intermediate_models_from_disk)
        self.input_shape = input_shape
        
    
    @classmethod
    def from_path(cls, model_path: Union[Path | str]=None, input_shape=(1, 3, 640, 640), load_intermediate_models_from_disk=True):
        return cls(model_path, input_shape, load_intermediate_models_from_disk)
        
    
    @classmethod
    def from_model(cls, model: nn.Module, model_path: Union[Path | str]=None, input_shape=(1, 3, 640, 640), load_intermediate_models_from_disk=True):
        self = cls(model_path, input_shape, load_intermediate_models_from_disk)
        self._torch_model = model
        self.to_torch()
        return self
    
    
    def _to_onnx(self) -> ModelProto:
        self.to_torch()
        dummy_input = torch.randn(self.input_shape)

        # torch to onnx
        torch.onnx.export(
            self._torch_model,
            dummy_input,
            self._onnx_path,
            verbose=False,
            opset_version=15, 
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        )
        return simplify_onnx(self._onnx_path)


    def _to_torch(self) -> nn.Module:
        return torch.jit.load(self._torch_path)
        

if __name__ == '__main__':
    converter_tfl = TFLiteConverter.from_model_path(Model.DRONE_TFLITE, load_intermediate_models_from_disk=True)
    onnxio = OnnxIO(converter_tfl.to_onnx())
    converter_torch = TorchConverter.from_model(
        converter_tfl.to_torch(),
        input_shape=onnxio.input_shape,
        load_intermediate_models_from_disk=False,
    )
    converter_torch.to_tflite()
        
    
        