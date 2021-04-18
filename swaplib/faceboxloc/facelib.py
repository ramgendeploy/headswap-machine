import cv2
import numpy as np
import matplotlib.pyplot as plt

# draw_facebox
# cropface
# image_resize
def draw_facebox(filename, dim, fill):
    x,y,w,h = dim
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    x, y, width, height = dim
    rect = plt.Rectangle((x-fill/2, y-fill/2,), width+fill, height+fill, fill=False, color='orange')
    ax.add_patch(rect)
    plt.show()

def cropface(image, box, fill=50):
  shape = image.shape
  if len(shape) > 2 :
    h,w,c = shape
  else:
    h,w = shape

    print(mask)

  x,y,w,h = box

  y_fill = y-(fill//2) if y-(fill//2) > 0 else 0
  x_fill = x-(fill//2) if x-(fill//2) > 0 else 0

  h_fill = y+h+fill 
  w_fill = x+w+fill

  return image[y_fill:(h_fill),(x_fill):(w_fill)]

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def crop_em(source_im, seg_mask):
  """
    Crops the image accordingly to the binary mask given.
    source_im np.array cv2.imread
    seg_mask binary mask  
  """
  # parsing = face2parsing_maps(source_path)
  # masked, mask, points = parsing2mask(source_im, parsing)

  box = cv2.boundingRect(seg_mask.astype(np.uint8))
  cropface_img = cropface(source_im, box, fill=50)

  if cropface_img.shape[1] > 1000:
    crop_r = image_resize(cropface_img, width = 1000)
  else:
    crop_r = cropface_img
  
  return crop_r


  # (make the check inline)

  # crop_im = cv2.imread(cropface_path)

  # parsing_c = face2parsing_maps(cropface_path)
  # masked_c, mask_c, points = parsing2mask(crop_im, parsing_c)

  # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_c.astype(np.uint8), connectivity=8)
  # sizes = stats[1:, -1]; nb_components = nb_components - 1

  # if nb_components > 2:
  #   min_size = 150  
  #   print("There is more components than expected")


