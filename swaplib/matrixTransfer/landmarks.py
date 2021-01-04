import cv2
import os
import numpy as np
import urllib.request as urlreq
import gdown

def download_models_():
  """
    Downloads the landmarks models
  """
  haarcascade_path = "haarcascade_frontalface_alt2.xml"
  LBFmodel_path = "lbfmodel.yaml"
  if (not (haarcascade_path in os.listdir(os.curdir))):
    gdown.download('https://drive.google.com/file/d/1NXCKbuafQZekUstRytk1VO9fpjwujvJa', 
      output='haarcascade_frontalface_alt2.xml')
  if (not (LBFmodel_path in os.listdir(os.curdir))):
    gdown.download('https://drive.google.com/file/d/147T_RQSD7_EoT5TROMox8qrd-mFt6UqZ', 
      output='lbfmodel.yaml')
  return haarcascade_path, LBFmodel_path
  
def landmarks(s_paths):
  haarcascade_path = "/home/rama/legendfacesPipeline/models/haarcascade_frontalface_alt2.xml"
  LBFmodel_path = "/home/rama/legendfacesPipeline/models/lbfmodel.yaml"

  landmarks_arr = []
  detector = cv2.CascadeClassifier(haarcascade_path)

  landmark_detector = cv2.face.createFacemarkLBF()
  landmark_detector.loadModel(LBFmodel_path)

  for s_path in s_paths:
    image = cv2.imread(s_path)[...,::-1]
    image_landmarks = image.copy()
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = detector.detectMultiScale(image_gray)
    
    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmarks_arr.append(landmarks)
  
  # if show_lm:
  #   print(len(landmarks_arr[i]))

  return landmarks_arr

def landmarks_(loaded_imgs):
  """
    loaded_imgs is an array of np.array images
  """
  haarcascade_path, LBFmodel_path = download_models_()

  landmarks_arr = []

  detector = cv2.CascadeClassifier(haarcascade_path)
  landmark_detector = cv2.face.createFacemarkLBF()
  landmark_detector.loadModel(LBFmodel_path)

  for ld_img in loaded_imgs:
    image = ld_img
    image_landmarks = image.copy()
    image_draw = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(image_gray)
    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmarks_arr.append(landmarks)

  return landmarks_arr