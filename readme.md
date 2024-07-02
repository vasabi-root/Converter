# Machine Learning Model Converter
Устанавливаем [`CMake`](https://cmake.org/download/). Клоним репо. Устанавливаем `requirements.txt`:
```sh
git clone https://github.com/vasabi-root/Converter.git
cd Converter
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```
В этом файле уже указаны мои форки с фиксами некоторых либ. 

Далее рекомендую ознакомиться с [доками](docs/notes.md).

Если кому-то потребуется коммитить, то вот инфа о том, какие либы я менял и как их собрать:

- [`onnx2torch`](https://github.com/vasabi-root/onnx2torch/commit/a3362df4f11c1d0c8236387d7d3f89a7e250b595)  
  Качаем и устанавливаем:
  ```sh
  git clone https://github.com/vasabi-root/onnx2torch.git
  cd onnx2torch
  python -m pip install --upgrade build
  python -m build
  cd dist
  pip install onnx2torch-.....whl
  ```
- `onnx2tf`. Не коммитил, ибо решение некорректное (подробнее в доках). Но модифицированные файлы лежат в папке `changes/onnx2tf`
