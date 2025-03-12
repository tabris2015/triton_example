#!/bin/bash

pip install -U tf2onnx
python -m tf2onnx.convert --input assets/frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output assets/detection.onnx
