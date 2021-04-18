import re
import torch
import numpy as np
from torchvision import transforms
from PIL import ImageFilter
from PIL import Image, ImageEnhance
from .color_transfer import *
from .fast_style_transfer import TransformerNet, load_image, process_image

def doStyle(cropface, 
        ct_sat_brigh = [0,0,0],
        model_path = f"./models/styles/st_style_3.pth"):
    """
        cropface : PIL image

        returns PIL image
    """
    # Refactor this to do it inplace

    device = torch.device("cpu")

    content_image = cropface

    # content_image = load_image(cropface)

    content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        output = style_model(content_image).cpu()
    
    source_st, source_st_arr = process_image(output[0])

    source_st = source_st.filter(ImageFilter.DETAIL)
    source_st = source_st.filter(ImageFilter.SMOOTH_MORE)

    # Saving the file
    # source_st.save(cropfacestyle_path)

    # --- Color transfer and filter
    # cropface_styleCT = doColorTransfer(cropface.astype(float)/256, cropfacestyle_path, raw_data)
    cropface_styleCT = doColorTransfer(np.asarray(cropface).astype(float)/256, np.asarray(source_st).astype(float)/256, True)

    # Filters
    cropf_ct = Image.fromarray((cropface_styleCT * 255).astype(np.uint8))

    contrast_enh = ImageEnhance.Contrast(cropf_ct)
    contrast_filter = contrast_enh.enhance(1.0 + ct_sat_brigh[0])

    color_enh = ImageEnhance.Color(contrast_filter)
    color_filter = color_enh.enhance(1.0+ ct_sat_brigh[1])

    br_enh = ImageEnhance.Brightness(color_filter)
    br_filter = br_enh.enhance(1.0 + ct_sat_brigh[2])

    # br_filter.save(final_image_path)

    return br_filter