import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def add_watermark(image, text='T', fontsize=10, color=(0, 0, 0), offX=None, offY=None):
    img_width, img_height, _ = image.shape

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("Ubuntu-R.ttf", fontsize)

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
        plt.title("black text")
        plt.imshow(image)

    return np.array(image)


def add_watermark_by_class(class_watermark_dict, X, Y, labels_str, save_dataset=True, saveing_dir='/tmp/'):
    """
    :param class_watermark_dict: {class_name: wm_symbol} -> {str: str}
    :param X: data
    :param Y: labels
    :param labels_str: list of labels str
    :return: x_watermark
    """
    X_watermark = np.empty_like(X)
    n_data = X.shape[0]

    for i, (x, y) in enumerate(zip(X, Y)):
        y = y[0]
        class_name = labels_str[y]
        if class_name in class_watermark_dict:
            x = add_watermark(x,
                              text=class_watermark_dict[class_name]
                              )
        X_watermark[i] = x

        # if save_dataset:

    return X_watermark
