import yaml
# from swaplib.matrixTransfer.dotheswap import doTheSwap
import swaplib

with open('./settings.yaml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

SwapDone = swaplib.doTheSwap(cropface_path, cropfacestyleCT_path, refTemp_path, hless_path, False)

cv2.imwrite(final_image_path, SwapDone)



if __name__ == "__main__":
    import sys
    print(int(sys.argv[1]))
