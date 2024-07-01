from pathlib import Path

import onnx
# import torch.onnx.verification

import onnxsim
from onnx import ModelProto

import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/converter.log', 
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',)

def simplify_onnx(onnx_path: Path) -> ModelProto:
    onnx_model = onnx.load(onnx_path)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, onnx_path)
    return onnx_model


class OnnxIO():
    def __init__(self, onnx_model: ModelProto):
        self.onnx_model = onnx_model
        
        self.inputs = None
        self.img_sizes = None
        self.outputs = None
        
        self.input_shape = None
        self.img_sz = None
        
        self._init_io_shape()
        
        
    def _init_io_shape(self):
        graph = self.onnx_model.graph
        assert len(graph.input) == 1

        self.inputs = {}
        self.img_sizes = {}
        for inp in graph.input:
            shape = str(inp.type.tensor_type.shape.dim)
            self.inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
            
            self.img_sizes[inp.name] = self.inputs[inp.name].copy()
            # без батча и каналов
            self.img_sizes[inp.name].remove(1) 
            self.img_sizes[inp.name].remove(3)
            self.img_sizes[inp.name] = self.img_sizes[inp.name][::-1]

        self.outputs = {}
        for out in graph.output:
            shape = str(out.type.tensor_type.shape.dim)
            self.outputs[out.name] = [int(s) for s in shape.split() if s.isdigit()]

        self.input_shape = list(self.inputs.values())[0]
        self.img_sz = list(self.img_sizes.values())[0]