# HƯỚNG DẪN SỬ DỤNG

[![Demo](http://img.youtube.com/vi/kBw-xGEIYhY/0.jpg)](http://www.youtube.com/watch?v=kBw-xGEIYhY)

## Requirements

Bước 1:cài packages dùng pip
```
* opencv
* tensorflow
* matplotlib
* numpy
```
Bước 2: Train data

source gốc data hơn 1gb nhưng t xóa bớt train cho nhanh( nhưng độ chính xác giảm)
```
python train.py --bottleneck_dir=logs/bottlenecks --how_many_training_steps=2000 --model_dir=inception --summaries_dir=logs/training_summaries/basic --output_graph=logs/trained_graph.pb --output_labels=logs/trained_labels.txt --image_dir=./dataset
```
If you're using the provided dataset, it may take up to three hours.
data gốc ở đây: https://github.com/loicmarie/sign-language-alphabet-recognizer.git
  

## Using webcam (demo)

test demo xài lệnh này
```
python classify_webcam.py
```

