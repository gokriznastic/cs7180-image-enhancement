import numpy as np
import torch
import torch.optim as optim

from model import get_features
from image_utils import im_convert

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    ## get the batch_size, depth, height, and width of the Tensor
    batch_size, c, h, w = tensor.size()

    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.reshape(c, h * w)

    ## calculate the gram matrix
    gram = torch.mm(tensor, torch.transpose(tensor, 0, 1))

    return gram

def transfer_style(steps, target, model, content_features, style_features, style_weights, style_grams, content_weight, style_weight):
    # for displaying the target image, intermittently
    show_every = 300

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    # steps = 10000  # decide how many iterations to update your image

    for ii in range(1, steps+1):
        ## get the features from your target image
        ## Then calculate the content loss
        print('Epoch: ', ii)
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            ## Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)

            ## get the "style" style representation
            style_gram = style_grams[layer]

            ## Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)


        ## calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
        #     plt.imshow(im_convert(target))
        #     plt.show()

    return target
