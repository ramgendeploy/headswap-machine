from swaplib.matrixTransfer import headTransfer
import cv2

def main():
  result = headTransfer.Swap(
  './tests/crop_source.jpg', 
  './tests/rf.jpg', 
  './tests/rf.jpg',
  )
  cv2.imwrite("./test_result.jpg", result)
if __name__ == "__main__":
  main()