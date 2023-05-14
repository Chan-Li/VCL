import cupy as np
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img
def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab
def load_mnist(path, kind='train',normalize=True,one_hot=True):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        label = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        image= np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(label), 784)
    if normalize:
        images= __normalize_image(image)
    if one_hot:
        labels = __one_hot_label(label)

    return images, labels

