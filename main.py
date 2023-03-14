from __future__ import absolute_import
from Defense_Tracker import Defense_Tracker

import os
import os
import glob
import six
import numpy as np
import io

def run(tracker, img_files, box,  visualize=False):
    # tracking loop

    boxes, times = tracker.track(
        img_files, box, visualize=visualize)



if __name__ == '__main__':

    net_path_DCF = 'pretrained/param.pth'  #DCFNET weight
    net_path_VAE = 'pretrained/VAE_Param.pth'  #Variational auto encoder weight

    data_root = 'Data/army-181108_0_0002'  #test image root1
    # data_root = 'Data/fog'  #test image root2

    ground_truth = os.path.join(data_root, 'groundtruth.txt')  #first frame box information

    with open(ground_truth, 'r') as f:
        box = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

    img_files = sorted(glob.glob(os.path.join(data_root, '*.jpg')))

    tracker = Defense_Tracker(net_path1=net_path_DCF, net_path2=net_path_VAE)  #tracker network initial

    run(tracker, img_files, box, visualize=True)    #tracker running



