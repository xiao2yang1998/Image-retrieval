import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist


class ImageBlock:
    def __init__(self, path):
        self.image_path = path

    def show_corners(self,corners, image):
        fig = plt.figure()
        plt.gray()
        plt.imshow(image)
        y_corner, x_corner = zip(*corners)
        plt.plot(x_corner, y_corner, 'or')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
        plt.show()

    def get_corners(self):
        im = io.imread(self.image_path)
        im = equalize_hist(rgb2gray(im))
        corners = corner_peaks(corner_harris(im), min_distance=2)
        #self.show_corners(corners, im)
        return len(corners)

