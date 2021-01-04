import cv2
import os
import torch
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .bisenet import BiSeNet

pretrained_W = "/home/rama/legendfacesPipeline/models/bisenet_pretrained.pth"

def face2parsing_maps(img_source):
  """
    img_source nd.array RGB Image
  """
  n_classes = 19
  net = BiSeNet(n_classes=n_classes)
  net.cpu()
  net.load_state_dict(torch.load(pretrained_W, map_location = torch.device('cpu')))
  net.eval()

  to_tensor = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  with torch.no_grad():
      # img = Image.open(img_path)
      
      img = Image.fromarray(img_source)

      image = img.resize((512, 512), Image.BILINEAR)

      img = to_tensor(image)
      img = torch.unsqueeze(img, 0)
      img = img.cpu()
      out = net(img)[0]
      parsing = out.squeeze(0).cpu().numpy().argmax(0)

  return parsing

def parsing2mask(source_image, parsing_anno=[], include=[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0]):
    """
      im is a cv2 image
      parsing_anno is the parsing returned by face2parsing_maps
    """
    if len(parsing_anno) == 0:
      parsing_anno = face2parsing_maps(source_image[...,::-1])
      
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    num_of_class = np.max(vis_parsing_anno)
    mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 1))
    
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        mask[index[0], index[1], :] = include[pi]


    # source_im = cv2.imread(source_path)
    source_im = source_image
    mask_resized = cv2.resize(mask,(source_im.shape[1], source_im.shape[0]), interpolation=cv2.INTER_NEAREST)

    bitws = cv2.bitwise_and(source_im, source_im, mask=np.uint8(mask_resized))

    return bitws, mask_resized