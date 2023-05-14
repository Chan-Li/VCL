#来自https://blog.csdn.net/weixin_40522523/article/details/82823812
import cupy as np
from struct import unpack
import os

TRAIN_IMAGES = str(os.getcwd())+'/mnist/train-images.idx3-ubyte'  #训练集图像的文件名
TRAIN_LABELS = str(os.getcwd())+'/mnist/train-labels.idx1-ubyte'  #训练集label的文件名
TEST_IMAGES = str(os.getcwd())+'/mnist/t10k-images.idx3-ubyte'    #测试集图像的文件名
TEST_LABELS = str(os.getcwd())+'/mnist/t10k-labels.idx1-ubyte'    #测试集label的文件名


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab
def load_mnist(train_image_path=TRAIN_IMAGES, train_label_path=TRAIN_LABELS, test_image_path=TEST_IMAGES, test_label_path=TEST_LABELS, normalize=True, one_hot=False):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])

