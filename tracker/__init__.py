from __future__ import absolute_import

import numpy as np
import time
from PIL import Image

from util import ops


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

from .identity_tracker import IdentityTracker
