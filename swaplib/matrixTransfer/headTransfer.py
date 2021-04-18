from ..faceboxloc.facebox import makeFaceBox
from ..stylelib.doStyle import doStyle
from ..stylelib.color_transfer import match_color
from ..seglib.faceSegmentation import face2parsing_maps, parsing2mask 
from cv2.ximgproc import guidedFilter

from PIL import Image

from .landmarks import *
from .swapLib import *

def Swap(source_path, 
              # style_path, 
              ref_path, 
              headless_path, 
              hue_value=260, 
              alpha_blend=40, 
              alpha_blend_color=40,
              color_variation=1e-5, 
              hue_change=False, 
              morph_closing=False,
              linear_color=False):

  """
    source_path Path to the source face
    style_path deprecated
    ref_path reference template path
    headless_path headless template path
  """
  # Makefacebox
  face_box = makeFaceBox(source_path, save=False)
  # Style transfer and colortransfer

  face_stylized = doStyle(Image.fromarray(face_box))

  # Face Matrix Transference
  eps = 5e-6
  eps *= 255*255
  COLOUR_CORRECT_BLUR_FRAC = 0.6
  LEFT_EYE_POINTS = list(range(42, 48))
  RIGHT_EYE_POINTS = list(range(36, 42))
  FACE_POINTS = list(range(17, 68))
  LEFT_EYE_POINTS = list(range(42, 48))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_BROW_POINTS = list(range(22, 27))
  RIGHT_BROW_POINTS = list(range(17, 22))
  NOSE_POINTS = list(range(27, 35))
  MOUTH_POINTS = list(range(48, 61))
  JAW_POINTS_ = list(range(4, 13))
  ALIGN_POINTS = (JAW_POINTS_)

  OVERLAY_POINTS = [
      LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS+
      NOSE_POINTS + MOUTH_POINTS+JAW_POINTS_
  ]

  FEATHER_AMOUNT = 15 

  # Landmarks
  lm = landmarks([source_path, ref_path])
  source_lm = lm[0][0][0]
  target_lm = lm[1][0][0]

  # Getting the transformation matrix
  tr = transformation_from_points(np.matrix(target_lm), np.matrix(source_lm))

  # Reading inputs and showing { form-width: "33%" }
  # source_im = cv2.imread(source_path, cv2.IMREAD_COLOR)
  # style_im = cv2.imread(style_path, cv2.IMREAD_COLOR)

  source_im = np.asarray(face_box)[...,::-1]
  style_im = np.asarray(face_stylized)[...,::-1]

  cv2.imwrite("./temp/source_im.jpg", source_im)
  cv2.imwrite("./temp/style_im.jpg", style_im)

  target_im = cv2.imread(ref_path, cv2.IMREAD_COLOR)
  head_less = cv2.imread(headless_path, cv2.IMREAD_COLOR)

  st_im = style_im[...,::-1]/255
  hl_tm = head_less[...,::-1]/255
  style_linear_ct = match_color(st_im, hl_tm, eps = color_variation)
  cv2.imwrite("./temp/style_linear_ct.jpg", style_linear_ct[...,::-1]*255)

  if linear_color:
    style_im = cv2.addWeighted(style_im, alpha_blend_color/100, (style_linear_ct*255).astype(np.uint8), (1-alpha_blend_color/100),0, dtype=cv2.CV_8U)
    style_im = style_im[...,::-1]


  # Face segmentantion & mask of source { form-width: "33%" }
  # [] fix this to do it inplace

  parsing = face2parsing_maps(face_box)
  # masked_face, mask_face, points_face = parsing2mask(source_im, parsing, include=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]) 
  masked_face, mask_face = parsing2mask(source_im, parsing, include = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]) 
  
  mask_face = guidedFilter(source_im.astype(np.float32), mask_face.astype(np.float32), 40, eps)
  mask_face3d = np.array([mask_face,mask_face,mask_face]).transpose(1,2,0)


  
  # mask_face3d = get_masked_blur(mask_face, 15, (15,15)) #<- blurs and makes 3d mask
  # masked_hair, mask_hair, points_hair = parsing2mask(source_im, parsing, include=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0])  
  masked_hair, mask_hair = parsing2mask(source_im, parsing, include = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0])  

  

  # Hair mask process
  mask_hair_erode = cv2.erode(mask_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 3)
  better_hair = guidedFilter(source_im.astype(np.float32), mask_hair_erode.astype(np.float32), 40, eps)
  better_hair3d = np.array([better_hair,better_hair,better_hair]).transpose(1,2,0)

  # mask_blur = get_masked_blur(mask, 15, (15,15)) 


  compose_mask =  better_hair3d + mask_face3d

  # Mask morph operations
  if morph_closing:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
    compose_mask = cv2.morphologyEx(compose_mask, cv2.MORPH_CLOSE, kernel)
    compose_mask = cv2.morphologyEx(compose_mask, cv2.MORPH_OPEN, kernel)

  s_lm = np.matrix(source_lm[ALIGN_POINTS])
  t_lm = np.matrix(target_lm[ALIGN_POINTS])
  M = transformation_from_points(t_lm, s_lm) 

  
  warped_mask = warp_im(compose_mask, M, target_im.shape)
  
  # Warped style/source image
  if hue_change:
    # hue filter
    hsv_img = rgb2hsv(style_im/255)
    rgb_img = hsv2rgb(hsv_img)

    hue_mean = np.mean(hsv_img[:,:,0])*360

    picked_hue = 260
    if hue_mean < picked_hue:
      hue_shifter = (picked_hue-hue_mean)/360
    else:
      hue_shifter = (hue_mean-picked_hue)/360
      hue_shifter = -hue_shifter
    
    new_hue = np.array([hsv_img[:,:,0]+hue_shifter, hsv_img[:,:,1], hsv_img[:,:,2]]).transpose(1,2,0)
    
    new_img = hsv2rgb(new_hue)
    
    blend_hue = cv2.addWeighted(new_img, alpha_blend/100,rgb_img, (1-alpha_blend/100),0)

    warped_source_im = warp_im(blend_hue*255, M, target_im.shape)
  else:
    warped_source_im = warp_im(style_im, M, target_im.shape)



  final = head_less*(1-warped_mask)+(warped_source_im*warped_mask)
 
  
  
  print("Swap Done")
  return final