from swaplib.matrixTransfer import headTransfer
import cv2

def main():
  result = headTransfer.Swap(
  source_path='./tests/source_1.jpg', 
  ref_path='./tests/target_img.jpg', 
  headless_path='./tests/headless_template.jpg',
  )
  cv2.imwrite("./tests/test_result.jpg", result)
if __name__ == "__main__":
  main()