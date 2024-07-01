from pathlib import Path
import sys
sys.path.extend([str(Path(__file__).parents[1])])

import numpy as np
import onnx
import time
import onnxruntime as ort
import cv2
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn .model_selection import train_test_split

from src.converter import Converter
from src.trainer import Trainer
from src.utils import simplify_onnx, OnnxIO
from config import Model
import os

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# class Converter:
#     pass

def prepare_float_img(img: cv2.Mat, onnxio: OnnxIO):
    compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(onnxio.img_sz),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = compose(img).to(torch.float32).unsqueeze(0) # [H, W] -> [1, H, W]
    tensor = tensor.reshape(onnxio.input_shape)
    return tensor

def prepare_uint8_img(img: cv2.Mat, onnxio: OnnxIO):
    tensor = torch.tensor(img.transpose([2, 0, 1]))
    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(onnxio.img_sz),
    ])
    tensor = compose(tensor).to(torch.uint8).unsqueeze(0) # [H, W] -> [1, H, W]
    tensor = tensor.reshape(onnxio.input_shape)
    return tensor

def onnx_eval(onnx_path: Path, img: cv2.Mat):
    onnx.checker.check_model(onnx_path)
    ort_sess = ort.InferenceSession(onnx_path)
    onnxio = OnnxIO(onnx.load(onnx_path))
    return ort_sess.run(None, {list(onnxio.inputs.keys())[0]: img})


def torch_eval(torch_model: torch.nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        tensor = tensor.to(torch.float32)
        out = torch_model(tensor)
    
    if type(out) == list:
        result = []
        for item in out:
            result.append(item.cpu().detach().numpy())
        return result
        
    return [out.cpu().detach().numpy()]

def gen_resnet():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, Model.RESNET_ONNX.with_suffix('.pt'))
    torch.onnx.export(
        model, 
        torch.randn(1, 3, 224, 224),
        Model.RESNET_ONNX,
        input_names=['input'],
        output_names=['output']
    )
    return model

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 48

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    
    locs = np.random.randint(len(trainset), size=round(len(trainset)*0.2))
    validset = [trainset[loc] for loc in locs]
    validloader = DataLoader(validset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    labels = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, validloader, testloader, labels

def calc_out_errors(orig_outs, onnx_outs, torch_outs):
    for orig_out, onnx_out, torch_out in zip(orig_outs, onnx_outs, torch_outs):  
        orig_out, onnx_out, torch_out = map(np.squeeze, (orig_out, onnx_out, torch_out))
        print('')
        print(' argmax(onnx)        = ', np.argmax(orig_out))
        print(' argmax(onnx_quant)  = ', np.argmax(onnx_out))
        print(' argmax(torch)       = ', np.argmax(torch_out))
        print('')
        if len(orig_out.shape) != 0:
            print(' R2(onnx, onnx_quant)  = ', r2_score(orig_out, onnx_out))
            print(' R2(onnx_quant, torch) = ', r2_score(onnx_out, torch_out))
            print(' R2(onnx, torch)       = ', r2_score(orig_out, torch_out))
            print('')
    

def test_difference_qdq(model_float: str, model_qdq: str, test_img: str, uint_flag: bool):
    # converter_f = Converter(model_path=Model.RESNET_ONNX)
    # converter_q = Converter(model_path=Model.RESNET_QDQ_ONNX)
    converter_f = Converter(model_path=model_float)
    converter_q = Converter(model_path=model_qdq)

    converter = converter_q
    onnxio = OnnxIO(converter.to_onnx())

    img = cv2.imread(test_img)
    prepare_tensor = prepare_uint8_img if uint_flag else prepare_float_img
    tensor = prepare_tensor(img, onnxio)
    img = tensor.cpu().detach().numpy()
    
    # orig_out = torch_eval(model, img)
    orig_outs = onnx_eval(converter_f.onnx_path, img)
    onnx_outs = onnx_eval(converter.onnx_path, img)
    torch_outs = torch_eval(converter.to_torch(), tensor)

    calc_out_errors(orig_outs, onnx_outs, torch_outs)


def test_cls_training(test_qdq_model=True):
    train_loader, valid_loader, test_loader, labels = get_loaders()

    if test_qdq_model:
        converter = Converter(model_path=Model.RESNET_QDQ_ONNX)
        model = converter.to_torch()
        model.__setattr__('head/fc/Conv', torch.nn.Conv2d(2048, len(labels), (3, 3), padding=1))
    else:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))

    trainer = Trainer(
        model,
        train_loader, valid_loader, test_loader,
        'resnet_cifar10',
        labels
    )
    trainer.train(3, 0.001, load_from_disk=False)
    confmat = trainer.test()
    confmat.plot()

    # converter.to_tflite()


def main():
    test_difference_qdq(
        Model.RESNET_ONNX, 
        Model.RESNET_QDQ_ONNX, 
        'test/test_images/banana.jpg', 
        uint_flag=False
    )
    # test_difference_qdq(
    #     Model.DRONE_ONNX, 
    #     Model.DRONE_ONNX, 
    #     'test_images/drone.png', 
    #     uint_flag=True
    # )
    
    # test_cls_training(test_qdq_model=True)


if __name__ == '__main__':
    main()