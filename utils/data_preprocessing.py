from os.path import join as pjoin
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import os
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


def add_watermark(image, text='T', fontsize=8, color=(128,0,128, 100), offX=None, offY=None, verbose=0,
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
    # wm_indices = {}
    # for key in class_watermark_dict.keys():
    #     kv = {key: []}
    #     wm_indices.update(kv)

    for i, (x, y) in enumerate(zip(X, Y)):
        y = y[0]
        c = labels_str[y]
        if c in class_watermark_dict:
            x = add_watermark(x,
                              text=class_watermark_dict[c],
                              fonttype=fonttype
                              )

            if save_dataset:
                class_saving_dir = pjoin(*[saveing_dir, 'val' if train0validation1 else 'train', c])
                if not (os.path.isfile(class_saving_dir) and os.access(class_saving_dir, os.R_OK)):
                    # Create dataset root
                    makedir(class_saving_dir)
                # Save image
                pil_img = Image.fromarray(x)
                full_dir = pjoin(class_saving_dir, f'{i:{n_digits}}' + '.png')
                pil_img.save(full_dir)

        X_watermark[i] = x
    return X_watermark

def prepare_watermark_dataset(x, cls1, cls2, data_path):
    cls1_dir = pjoin(data_path, cls1)
    cls2_dir = pjoin(data_path, cls2)

    cls1_data, cls1_inds = load_watermark_dataset(cls1_dir)
    cls2_data, cls2_inds = load_watermark_dataset(cls2_dir)

    x_watermark = np.empty_like(x)
    n_data = x.shape[0]

    for i in range(n_data):
        counter = 0
        if i in cls1_inds:
            img_ind = cls1_inds.index(i)
            x_watermark[i] = cls1_data[img_ind]
            counter += 1

        if i in cls2_inds:
            img_ind = cls2_inds.index(i)
            x_watermark[i] = cls2_data[img_ind]
            counter += 1

        if counter > 1:
            raise IndexError (f'Image {i} was found in both {cls1} and {cls2} datasets')

        elif counter == 0:
            x_watermark[i] = x[i]

    return x_watermark


def load_watermark_dataset(data_path):
    images = []
    indices = []
    for img_name in os.listdir(data_path):
        try:
            img = mpimg.imread(os.path.join(data_path, img_name))
            img_index = int(img_name.split('.')[0])
            if img is not None:
                images.append(img)
                indices.append(img_index)
        except:
            print('Cant import ' + img_name)
    images = np.asarray(images)
    return images, indices