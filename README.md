## Description
This sample contains code that performs TensorRT inference on Jetson.
1. Download ONNX Ultra-Fast-Lane-Detection Model from PINTO_model_zoo.
2. Convert ONNX Model to Serialize engine and inference on Jetson.


## Reference
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
    - [140_Ultra-Fast-Lane-Detection](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection)
- [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
- [ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference)

## Howto

### Install dependency
```
pip3 install -U scipy
```

### Download ONNX Model

Clone PINTO_model_zoo repository and download Ultra-Fast-Lane-Detection
 model.
```
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/140_Ultra-Fast-Lane-Detection/
./download.sh
```

Check `trtexec`
```
/usr/src/tensorrt/bin/trtexec --onnx=./saved_model_culane/ultra_falst_lane_detection_culane_288x800.onnx

or

/usr/src/tensorrt/bin/trtexec --onnx=./saved_model_tusimple/ultra_falst_lane_detection_tusimple_288x800.onnx
```

### Convert ONNX Model to TensorRT Serialize engine file.
Copy `ultra_falst_lane_detection_culane_288x800.onnx` or `ultra_falst_lane_detection_tusimple_288x800.onnx` to `tensorrt-examples/models`.  
In the following, `culane` is taken as an example.
```

### Convert to Serialize engine file.

Finally you can run the demo.
```
python3 main.py \
--model ../../models/ultra_falst_lane_detection_culane_288x800.trt 
--model_config culane 
--videopath output_video3.mp4 
--output /home/shirel/gst.mp4
