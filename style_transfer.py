from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from image_utils import display_content_style_images, im_convert, load_image
from model import get_features, load_model
from transfer_utils import gram_matrix, transfer_style

parser = ArgumentParser(description="Neural Style Transfer")

parser.add_argument("-c", "--content",
                    help="filepath for content image", required=True)
parser.add_argument("-s", "--style",
                    help="filepath for style image", required=True)
parser.add_argument("-g", "--gpu", default="False",
                    help="whether to use the GPU for style transfer")
parser.add_argument("-cw", "--content-weight", default=1,
                    help="weight for content")
parser.add_argument("-sw", "--style-weight", default=1e4,
                    help="weight for style")
parser.add_argument("-t", "--target",
                    help="filepath for target image", required=True)
parser.add_argument("-x", "--steps", default=10000,
                    help="how many iterations to update your image", required=True)
args = parser.parse_args()

# determine device
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

# load in content and style image
content = load_image(args.content).to(device)

# resize style to match content, makes code easier
style = load_image(args.style, shape=content.shape[-2:]).to(device)

assert content.size() == style.size()

# display_content_style_images(content, style)

vgg = load_model(device)

print(vgg)

# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start of with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.6,
                 'conv4_1': 0.4,
                 'conv5_1': 0.2}

# you may choose to leave these as is
content_weight = args.content_weight  # alpha
style_weight = args.style_weight  # beta

target = transfer_style(args.steps, target, vgg, content_features, style_features, style_weights, style_grams, content_weight, style_weight)

# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))


# save the target image
target_image = Image.fromarray((im_convert(target) * 255).astype(np.uint8))
target_image.save(args.target)
