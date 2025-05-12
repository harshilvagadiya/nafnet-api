import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Basicsr imports (adjust PYTHONPATH or pip install -e . if needed)
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse

def imread(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # convert BGR → RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.title('Input'); plt.axis('off'); plt.imshow(img1)
    plt.subplot(1, 2, 2); plt.title('NAFNet Output'); plt.axis('off'); plt.imshow(img2)
    plt.show()

def single_image_inference(model, img_tensor, save_path):
    model.feed_data(data={'lq': img_tensor.unsqueeze(0)})
    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()
    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

def main():
    # 1) Paths
    opt_path    = 'options/test/REDS/NAFNet-width64.yml'
    input_path  = '/home/admin2/Workspace/NAFNet/demo/blurry.jpg'
    output_dir  = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))

    # 2) Load model
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    # Force CPU mode by setting num_gpu to 0
    opt['num_gpu'] = 0
    model = create_model(opt)

    # 3) Read, infer, display
    img_in = imread(input_path)
    tensor = img2tensor(img_in)
    single_image_inference(model, tensor, output_path)
    img_out = imread(output_path)

    # 4) Show results
    display(img_in, img_out)

if __name__ == '__main__':
    main()
