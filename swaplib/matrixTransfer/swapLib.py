import numpy as np
import cv2

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# Blurring the mask and converting it to 3 channel binary mask
def get_masked_blur(mask, FM, kernel_s, iteration=1):
  kernel = np.ones(kernel_s,np.uint8)

  mask_im = np.array([mask,mask,mask]).transpose((1,2,0))
  mask_im = cv2.erode(mask_im, kernel, iteration)  
  

  mask_im = (cv2.GaussianBlur(mask_im, (FM,FM), 0) > 0) * 1.0
  mask_im = cv2.GaussianBlur(mask_im, (FM,FM), 0)
  return mask_im  

def correct_colours(im1, im2, landmarks1, COLOUR_CORRECT_BLUR_FRAC,points):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[points[0]], axis=0) -
                              np.mean(landmarks1[points[1]], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += 128 * (im2_blur <= 1.0).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))