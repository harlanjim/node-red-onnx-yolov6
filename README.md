# node-red-contrib-onnx-yolov6
[![platform](https://img.shields.io/badge/platform-Node--RED-red)](https://nodered.org)

## Node-red module that uses Microsoft Onnxruntime and Yolo version 6 object detection model.

Onnxruntime will work with Mac M1 and is the main reason for it's development. 

This code was based on this [work](https://github.com/ibaiGorordo/ONNX-YOLOv6-Object-Detection) in Python. There is a Colab there that will allow you to generate a model for the image size you want. The image size of the model in this module is 640x480.

I also used code from [node-red-contrib-tfjs-coco-ssd](https://github.com/dceejay/tfjs-coco-ssd) which uses TensorFlowjs that I couldn't get worked on Mac M1.

### Build
```
npm install
```

This node runs the object detector on a ***jpeg image***, delivered via an ```msg.payload``` in one of the following formats:
+ As a string, that represents a file path to a jpg file.
+ As a buffer of a jpg.
+ As an https url that returns a jpg.
+ As an html data:image/jpeg;base64, string



