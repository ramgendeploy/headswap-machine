import swaplib
import cv2

def main():
  result = swaplib.doTheSwap('./crop_source.jpg', 
  './rf_template.png', 
  './hl_template.png')
  # result.save("")
  cv2.imwrite("./test_result.jpg", result)
if __name__ == "__main__":
  main()