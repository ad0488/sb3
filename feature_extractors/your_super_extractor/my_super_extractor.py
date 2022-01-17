import cv2, sys
from skimage import feature
import numpy as np
from skimage import filters


class LBP:
    def __init__(self, resize=100):
        self.resize = resize
        self.radius = 3
        self.n_points = 8 * self.radius

    def lbp(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im = feature.local_binary_pattern(im, self.n_points, self.radius)

        im = im.ravel()

        return im

    def lbp_hist(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im = feature.local_binary_pattern(im, self.n_points, self.radius)

        global_feature_vector = []

        for w in range(0, 100, 20):
            for h in range(0, 100, 20):
                global_feature_vector += np.histogram(im[w:w + 20, h:h + 20], bins=8)[0].tolist()

        global_feature_vector_norm = np.array(global_feature_vector) / np.linalg.norm(global_feature_vector)

        return global_feature_vector_norm.tolist()

    def prewitt(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im = filters.prewitt(im)

        im = im.ravel()

        return im

    def sobel(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im = filters.sobel(im)

        im = im.ravel()

        return im

    def daisy(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im = feature.daisy(im)

        im = im.ravel()

        return im

    def hog(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.resize, self.resize))
        im_hog = feature.hog(im)

        return im_hog

if __name__ == '__main__':
    fname = sys.argv[1]
    img = cv2.imread(fname)
    extractor = Extractor()
    features = extractor.extract(img)
    print(features)