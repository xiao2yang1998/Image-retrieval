import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist
from PIL import Image,ImageDraw
import func_1 as f1
import csv

def getColorFeature(img):
    re = img.split()
    if len(re) == 1:
        r = re[0];
        g = re[0];
        b = re[0]
    else:
        r = re[0];
        g = re[1];
        b = re[2]
    r = np.array(r).flatten()
    g = np.array(g).flatten()
    b = np.array(b).flatten()
    h = []
    s = []
    v = []
    for i in range(r.shape[0]):
        h_t, s_t, v_t = f1.rgb22hsv(r[i], g[i], b[i])
        h.append(h_t)
        s.append(s_t)
        v.append(v_t)
    h = np.array(h).flatten()
    s = np.array(s).flatten()
    v = np.array(v).flatten()
    calc = [0 for i in range(0, 80)]
    re = []
    for i in range(r.shape[0]):
        temp = f1.quantilize(h[i], s[i], v[i])
        re.append(temp)
        calc[temp] += 1;
    return np.array(calc)

class ImageBlock:
    def __init__(self, path,image_name):
        self.image_path = path
        self.image_name = image_name

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

    def get_corners(self,im):
        im = np.array(im)
        corners = corner_peaks(corner_harris(im), min_distance=2)
        re = len(corners)
        # if(len(corners)!=0):
        #     self.show_corners(corners,im)
        # # 背景颜色反转
        # if re<15:
        #     im = 255-im
        #     corners = corner_peaks(corner_harris(im), min_distance=2)
        #     if re<len(corners):
        #         re = len(corners)
        return re



    @property
    def get_results(self):
        im = Image.open(self.image_path+'/'+self.image_name)
        im_color = im
        im = im.convert('L')
        w, h = im.size
        w_p = int(w / 3)
        h_p = int(h / 3)
        w_array = [0, w_p, 2 * w_p, w]
        h_array = [0, h_p, 2 * h_p, h]
        # 处理图片，直方图均值化
        im = (equalize_hist(np.array(im)))
        # 背景颜色反转
        if len(corner_peaks(corner_harris(im), min_distance=2))<10:
            im = Image.open(self.image_path+'/'+self.image_name).convert('L')
            im = equalize_hist(255 - np.array(im))
        im = im*255
        im = Image.fromarray(im.astype('uint8'))
        interest_num = []
        total_interest = 0
        total_color_feature = []
        for j in range(3):
            for i in range(3):
                box = (w_array[i], h_array[j], w_array[i + 1], h_array[j + 1])
                itemp = im.crop(box)
                re = self.get_corners(itemp)
                total_interest += re
                interest_num.append(re)
                # get the color feature of a single block
                itemp_color = im_color.crop(box)
                re = getColorFeature(itemp_color)
                re = [int(i) for i in re]
                total_color_feature.append(re)
        print(interest_num)
        print(total_interest)
        if total_interest == 0:
            w = [ '1' for i in interest_num]
        else:
            w = ['%.02f' % (i / total_interest) for i in interest_num]
        w = np.array(w,dtype=float)
        w = w.reshape(1,9)
        total_color_feature = np.array(total_color_feature)
        re = w.dot(total_color_feature)
        re = re[0]
        total_occur = (sum(re))
        main_color_occur = 0.05*total_occur
        main_color = {}
        for i in range(len(total_color_feature[0])):
            if re[i]> main_color_occur:
                main_color[i]=re[i]

        outfile = "fuc2.csv"
        main_color_store = [self.image_name]
        for i in main_color:
            main_color_store.append(i)
            main_color_store.append(main_color[i])
        print(main_color_store)
        with open(outfile, 'a', newline='') as out:
            csv_writer = csv.writer(out, dialect='excel')
            csv_writer.writerow(main_color_store)

        return 0




