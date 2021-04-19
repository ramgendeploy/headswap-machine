## Installation and required libs
pip3 install opencv-python \
sudo apt-get install libsm6 libxrender1 libfontconfig1 \
pip install opencv-contrib-python
pip3 install -r requirements.txt \
## Usage
```
result = headTransfer.Swap(
source_path='./tests/source_1.jpg', 
ref_path='./tests/target_img.jpg', 
headless_path='./tests/headless_template.jpg',
)
cv2.imwrite("./tests/test_result.jpg", result)
```