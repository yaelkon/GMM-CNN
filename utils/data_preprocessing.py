from os.path import join as pjoin
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from utils.file import makedir


def pure_pil_alpha_to_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def add_watermark(image, text='T', fontsize=8, color=(255, 0, 0, 100), offX=None, offY=None, verbose=0,
                  fonttype='Ubuntu-R.ttf'):
    img_width, img_height, _ = image.shape

    pil_image = Image.fromarray(np.uint8(image))
    pil_image = pil_image.convert('RGBA')
    draw = ImageDraw.Draw(pil_image)

    font = ImageFont.truetype(fonttype, fontsize)

    # watermark font
    # watermark offset
    offX = offX if offX else random.random()
    offY = offY if offY else random.random()
    # add watermark
    # If the image size is smaller than the watermark image, no watermark is added
    if img_width >= fontsize and img_height >= fontsize:

        x = int((img_width - fontsize)*offX)
        y = int((img_height - fontsize)*offY)
        draw.text((x, y), text, fill=color, font=font)

        # Keep transparent data
        # rgb_img = pil_image.convert('RGB')
        rgb_img = pure_pil_alpha_to_color(pil_image, color=color[:-1])
        if verbose:
            plt.subplots()
            plt.subplot(1, 2, 1)
            plt.title("RGBA")
            plt.imshow(pil_image)
            plt.subplot(1, 2, 2)
            plt.title("RGB")
            plt.imshow(rgb_img)

        return np.array(rgb_img)


def add_watermark_by_class(class_watermark_dict, X, Y, labels_str, train0validation1=0,
                           save_dataset=False, saveing_dir='/tmp/', fonttype='Ubuntu-R.ttf'):
    """
    :param save_dataset:
    :param class_watermark_dict: {class_name: wm_symbol} -> {str: str}
    :param X: data
    :param Y: labels
    :param labels_str: list of labels str
    :return: X_watermark
    """
    X_watermark = np.empty_like(X)
    n_data = X.shape[0]
    n_digits = '0' + str(len(str(n_data)))

    for i, (x, y) in enumerate(zip(X, Y)):
        y = y[0]
        class_name = labels_str[y]
        if class_name in class_watermark_dict:
            x = add_watermark(x,
                              text=class_watermark_dict[class_name],
                              fonttype=fonttype
                              )
        X_watermark[i] = x

        if save_dataset:
            if i == 0:
                # Create dataset root
                saveing_dir = pjoin(saveing_dir, 'Watermark_Data', 'validation' if train0validation1 else 'train')
                makedir(saveing_dir)
            # Save image
            pil_img = Image.fromarray(x)
            full_dir = pjoin(saveing_dir, f'{i:{n_digits}}' + '.png')
            pil_img.save(full_dir)

    return X_watermark
