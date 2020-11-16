import color_analyzer.ntc
import numpy as np
from scipy.cluster.vq import *
import image_to_numpy
import cv2

NUM_CLUSTERS = 5


def analyze_color(image_path, candidates=2):
    # Reading Image file
    img = image_to_numpy.load_image_file(image_path)

    img = cv2.resize(img, (160, 160))
    ar = np.asarray(img)

    # Collapse image to rgb array
    ar = np.reshape(ar, newshape=(-1, ar.shape[-1])).astype(float)

    # print('finding clusters')
    codes, dist = kmeans2(ar, NUM_CLUSTERS)

    # print('cluster centres:\n', codes)

    vecs, dist = vq(ar, codes)  # assign codes

    counts, bins = np.histogram(vecs, len(codes))  # count occurrences

    sorted_idx = np.argsort(counts)

    res = []
    for i in range(candidates):
        peak = codes[sorted_idx[i]]
        res += [ntc.name(peak)]

    return res


if __name__ == '__main__':
    res = analyze_color('C:/Users/Sopiro/Desktop/20200825/uchan.jpg')
