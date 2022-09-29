import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# helper function to load image given the image filepath
def load_image(img_path, max_size=800, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 800 pixels in the x-y dims.'''

    image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image

# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def display_content_style_images(content, style):
    # display the images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # content and style ims side-by-side
    ax1.set_title("Content Image",fontsize = 14)
    ax1.imshow(im_convert(content))
    ax2.set_title("Style Image",fontsize = 14)
    ax2.imshow(im_convert(style))
    plt.show()
