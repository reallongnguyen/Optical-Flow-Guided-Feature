# Nhận dạng hành động trong video - Nguyễn Phúc Long
Paper: [A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)

## Getting Started

### Prerequisites

Cài python3 và pip3. Nếu máy bạn đã có thì bỏ qua bước này.

Cài [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)
```
$ [sudo] pip3 install virtualenv
$ sudo apt install python3-venv
```

Tải dataset và pre-trained model:
- [UCF11](http://crcv.ucf.edu/data/UCF_YouTube_Action.php)
- [Inception-v2 pre-trained model](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)

### Installing

Tạo môi trường ảo bằng virtualenv
```
$ cd off
$ python3 -m venv .env
```

Chạy môi trường ảo và cài các package cần thiết
```
$ source .env/bin/activate
$ pip3 install -r requirement.txt
```

Thoát môi trường ảo
```
$ deactivate
```

### Run project

```
$ source .env/bin/activate
$ python3 train_video_classifier_1.py
```

## Built With

* [python3](https://www.python.org/)
* [tensorflow](https://tensorflow.org/)
* [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)

## Versioning

0.1

## Authors

**Nguyễn Phúc Long**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
