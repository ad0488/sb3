import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        lbp_features_arr = []
        lbp_features_arr_histogram = []
        p2p_features_arr = []
        prewitt_features_arr = []
        hog_features_arr = []
        sobel_features_arr = []
        daisy_features_arr = []
        y = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)

        # Change the following extractors, modify and add your own

        import feature_extractors.your_super_extractor.my_super_extractor as super_ext
        my_lbp = super_ext.LBP()

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)

            # y.append(cla_d['/'.join(im_name.split('/')[-2:])])
            y.append(cla_d[im_name.split('\\')[0].split('/')[-1] + '/' + im_name.split('\\')[1]])

            # Apply some preprocessing here

            # intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
            # img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

            # Run the feature extractors

            p2p_features = pix2pix.extract(img)
            p2p_features_arr.append(p2p_features)

            lbp_features = my_lbp.lbp(img)
            lbp_features_arr.append(lbp_features)

            lbp_features_hist = my_lbp.lbp_hist(img)
            lbp_features_arr_histogram.append(lbp_features_hist)

            prewitt_features = my_lbp.prewitt(img)
            prewitt_features_arr.append(prewitt_features)

            hog_features = my_lbp.hog(img)
            hog_features_arr.append(hog_features)

            sobel_features = my_lbp.sobel(img)
            sobel_features_arr.append(sobel_features)

            daisy_features = my_lbp.daisy(img)
            daisy_features_arr.append(daisy_features)

        Y_p2p = cdist(p2p_features_arr, p2p_features_arr, 'jensenshannon')
        Y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        Y_lbp_hist = cdist(lbp_features_arr_histogram, lbp_features_arr_histogram, 'jensenshannon')
        Y_prewitt = cdist(prewitt_features_arr, prewitt_features_arr, 'jensenshannon')
        Y_sobel = cdist(sobel_features_arr, sobel_features_arr, 'jensenshannon')
        Y_daisy = cdist(daisy_features_arr, daisy_features_arr, 'jensenshannon')
        Y_hog = cdist(hog_features_arr, hog_features_arr, 'jensenshannon')

        r1 = eval.compute_rank1(Y_p2p, y)
        r5 = eval.compute_rank5(Y_p2p, y)

        r1lbp = eval.compute_rank1(Y_lbp, y)
        r5lbp = eval.compute_rank5(Y_lbp, y)

        r1hog = eval.compute_rank1(Y_hog, y)
        r5hog = eval.compute_rank5(Y_hog, y)

        r1lbp_hist = eval.compute_rank1(Y_lbp_hist, y)
        r5lbp_hist = eval.compute_rank5(Y_lbp_hist, y)
        
        r1prewitt = eval.compute_rank1(Y_prewitt, y)
        r5prewitt = eval.compute_rank5(Y_prewitt, y)

        r1sobel = eval.compute_rank1(Y_sobel, y)
        r5sobel = eval.compute_rank5(Y_sobel, y)

        r1daisy = eval.compute_rank1(Y_daisy, y)
        r5daisy = eval.compute_rank5(Y_daisy, y)

        print('Pix2Pix Rank 1:', r1)
        print('Pix2Pix Rank 5:', r5)

        print('LBP Rank 1:', r1lbp)
        print('LBP Rank 5:', r5lbp)

        print('LBP Histogram Rank 1:', r1lbp_hist)
        print('LBP Histogram Rank 5:', r5lbp_hist)

        print('HOG Rank 1:', r1hog)
        print('HOG Rank 5:', r5hog)

        print('Prewitt Rank 1:', r1prewitt)
        print('Prewitt Rank 5:', r5prewitt)

        print('Sobel Rank 1:', r1sobel)
        print('Sobel Rank 5:', r5sobel)

        print('Daisy Rank 1:', r1daisy)
        print('Daisy Rank 5:', r5daisy)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()