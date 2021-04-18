# Function that executes the code (this is one import all the rest)
import cv2
from .facelib import *
from ..seglib.faceSegmentation import *

def makeFaceBox(source_path, cropface_pathsave='./crop_source.jpg', save=True):
  """
    Execution of the face box localization
    accepts an cv2 numpy image and returns or saves the cropped image
  """
  # source_im = cv2.imread(source_path)
  # parsing = face2parsing_maps(source_path)
  
  
  source_im = cv2.imread(source_path)
  
  masked, seg_mask = parsing2mask(source_im)
  cropped_face = crop_em(source_im, seg_mask)
  
  if save:
    # plt.imsave(cropface_pathsave, cropped_face[...,::-1])
    cv2.imwrite(cropface_pathsave, cropped_face)

  return cropped_face[...,::-1]  
  