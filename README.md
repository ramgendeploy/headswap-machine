## Installation and required libs
pip3 install opencv-python \
sudo apt-get install libsm6 libxrender1 libfontconfig1 \
pip install opencv-contrib-python
pip3 install -r requirements.txt \
## Usage
```
result = swaplib.doTheSwap(
  './crop_source.jpg', 
  './rf_template.png', 
  './hl_template.png')
  
  cv2.imwrite("./test_result.jpg", result)
```