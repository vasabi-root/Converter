import os
from typing import Union
from pathlib import Path

import onnx
import torch
import tensorflow as tf
# import torch.onnx.verification

import tf2onnx
import onnx2tf
import onnx2torch

import cv2
import numpy as np
import torch.nn as nn
from onnx import ModelProto

from config import Model

from abc import ABC, abstractmethod
from src.utils import simplify_onnx

import logging
logger = logging.getLogger(__name__)

class BaseConverter(ABC):  
    DEFAULT_MODEL_NAME = Path('converter_results/model.smth')
    def __init__(self, model_path: Union[Path | str]=None, load_intermediate_models_from_disk=True):
        '''
        `model_path`: [ Path | str ]. If not provided, all results will be saved in `converter_results` dir
        '''
        super().__init__()
        if model_path == None:
            model_path = BaseConverter.DEFAULT_MODEL_NAME
        model_path = Path(model_path)
        os.makedirs(model_path.parent, exist_ok=True)
        
        self.model_folder, self._tflite_path, self._onnx_path, self._torch_path = \
            Model.gen_model_paths(model_path)
            
        self._load_intermediate_models_from_disk = load_intermediate_models_from_disk
        
        self._tflite_model = None    
        self._onnx_model = None
        self._torch_model = None
        
        
    @classmethod
    def from_model_path(cls, model_path: Union[Path | str], load_intermediate_models_from_disk=True):
        self = cls(model_path, load_intermediate_models_from_disk)
        return self
    
    
    @classmethod
    @abstractmethod
    def from_model(cls, model, model_path=None, load_intermediate_models_from_disk=True):
        '''
        Implementation must contain initialization of corresponding _model property
        '''
        pass
    
    
    @abstractmethod
    def _to_onnx(self) -> ModelProto:
        '''
        Use origin model to convert to onnx
        '''
        pass
    

    def _to_tflite(self) -> tf.lite.Interpreter:
        '''
        Overload this method if the originate model is tflite
        '''
        self.to_onnx()
        onnx2tf.convert(
            input_onnx_file_path=self._onnx_path,
            output_folder_path=self.model_folder,
            # non_verbose=True,
        )
        old_name = str(self._tflite_path.with_suffix('')) + '_float32.tflite'
        os.replace(old_name, self._tflite_path)
        return tf.lite.Interpreter(str(self._tflite_path))
    
        
    def _to_torch(self) -> nn.Module:
        '''
        Overload this method if the originate model is torch
        '''
        self.to_onnx()
        torch_model = onnx2torch.convert(
            onnx_model_or_path=self._onnx_path,
        )
        return torch_model
    
    
    def _reduce_computations(self, model, model_path, model_load_func, model_convert_func):
        if model is not None:
            return model
        if self._load_intermediate_models_from_disk and os.path.exists(model_path):
            model = model_load_func(str(model_path))
            return model
        
        logger.info(f'convertion to "{model_path}" started')
        model = model_convert_func()
        logger.info(f'convertion to "{model_path}" ended')
        return model
    
    
    def to_onnx(self) -> ModelProto:
        self._onnx_model = self._reduce_computations(
            self._onnx_model,
            self._onnx_path,
            onnx.load,
            self._to_onnx,
        )
        return self._onnx_model
    
    
    def to_tflite(self) -> tf.lite.Interpreter:
        self._tflite_model = self._reduce_computations(
            self._tflite_model,
            self._tflite_path,
            tf.lite.Interpreter,
            self._to_tflite,
        )
        return self._tflite_model
    
    
    def to_torch(self) -> nn.Module:
        self._torch_model = self._reduce_computations(
            self._torch_model,
            self._torch_path,
            torch.jit.load,
            self._to_torch,
        )
        
        try:
            scripted = torch.jit.script(self._torch_model)
            torch.jit.save(scripted, self._torch_path)
        except RuntimeError:
            logger.warning('can not script torch model')
            pass
         
        return self._torch_model
    
    @property
    def onnx_path(self):
        self.to_onnx()
        return self._onnx_path
    
    @property
    def torch_path(self):
        self.to_torch()
        return self._torch_path
    
    @property
    def tflite_path(self):
        self.to_tflite()
        return self._tflite_path
    
    

    
