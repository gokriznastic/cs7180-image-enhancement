# cs7180-image-enhancement
Image enhancement based on Image Style Transfer using CNN by Gatys et al.

Link to the wiki-report : https://wiki.khoury.northeastern.edu/x/aR6iBw

### Name:
Gopal Krishna

### OS used: 
MACOS, Google Colab

### Usage instructions
The python file `style_transfer.py` contains the main function which require the following arguments to be run:

- `content` : The filepath to the content image to be used for style transfer.
- `style` : The filepath to the style image to be used for style transfer.
- `gpu` : If GPU, if available, is to be used during the iterations.
- `content-weight` : The amount of weightage to be given to the content image in the final target.
- `style-weight` : The amount of weightage to given to the style image in the final target.
- `target` : The filepath where the output style transferred image is to be saved.
- `steps` : The no. of iterations for which style transfer is to be done.

The file can be run as follows:
``` python3 style_transfer.py -c images/content.jpeg -s images/style.jpg -t images/target.png -x 10000 -cw 1 -sw 1000 ```

### Time travel days
None
