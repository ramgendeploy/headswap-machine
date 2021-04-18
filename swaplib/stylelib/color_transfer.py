# Here is all the Color transfer code

#@title CS yuv LIB { form-width: "33%" , display-mode: "form" }
#@markdown > rgb2luv \
#@markdown luv2rgb \
#@markdown doColorTransfer \
#@markdown doColorTransfer_ {doesnt work} \

# match_color
import cv2
import skimage 
import numpy as np
import imageio
from skimage import io,transform,img_as_float
from skimage.io import imread,imsave
from PIL import Image
from numpy import eye 

def match_color(target_img, source_img, eps=1e-5):

    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape(3,-1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])

    #PCA mode
    eva_t, eve_t = np.linalg.eigh(Ct)
    Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
    eva_s, eve_s = np.linalg.eigh(Cs)
    Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
    ts = Qs.dot(np.linalg.inv(Qt)).dot(t)


    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    
    return matched_img	

def rgb2luv(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114],[-.147, -.288, .436],[.615, -.515, -.1]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)
def luv2rgb(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    rgb = np.array([[1, 0, 1.139],[1, -.395, -.580],[1, 2.03, 0]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return rgb.transpose(1,2,0)

def doColorTransfer(org_content, output, raw_data, with_color_match = False):
  """
    org_content path or 0-1 np array
    output path or 0-1 np array
    raw_data boolean | toggles input
  """
  if not raw_data:
    org_content = imageio.imread(org_content, pilmode="RGB").astype(float)/256
    output = imageio.imread(output, pilmode="RGB").astype(float)/256

  org_content = skimage.transform.resize(org_content, output.shape)

  if with_color_match:
    output = match_color(output, org_content)

  org_content = rgb2luv(org_content)
  org_content[:,:,0] = output.mean(2)
  output = luv2rgb(org_content)
  output[output<0] = 0
  output[output>1]=1

  return output

def doColorTransfer_(org_content, output, output_f = "/content/"):

  org_content = org_content[...,::-1].astype(float)/256
  output = output[...,::-1].astype(float)/256

  org_content = cv2.resize(org_content, output.shape[:2])

  org_content = rgb2luv(org_content)
  org_content[:,:,0] = output.mean(2)
  output = luv2rgb(org_content)
  output[output<0] = 0
  output[output>1]=1

  return output