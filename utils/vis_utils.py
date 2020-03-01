
def crop_image(image, row_pixel, column_pixel, height, width):
    croped_img = image[row_pixel:row_pixel+height, column_pixel:column_pixel+width]
    return croped_img


def get_cifar10_labels():
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']