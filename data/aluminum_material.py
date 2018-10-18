# encoding=utf-8

import json
import os
import pprint

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.common import lsdir
from utils.plot_bbox import plot_bbox
from .bbox_dataset import DetectionDataset


class AluminumDet(DetectionDataset):
    def __init__(self, root=u"/data1/zyx/yks/dataset/guangdong_round2_train_20181011", is_train=True):
        super(AluminumDet, self).__init__()
        self.classes = [u"不导电", u"擦花", u"角位漏底", u"桔皮", u"漏底", u"喷流", u"漆泡", u'起坑', u'杂色', u'脏点']

        anno_list = list(lsdir(root, suffix=u".json"))
        self.objs = {}
        for ann_file in anno_list:
            anno = json.load(open(ann_file, "rb"))
            name = anno["imagePath"]
            bboxes = []
            for bbox in anno["shapes"]:
                points = bbox['points']
                x0, y0, x1, y1 = points[0][0], points[0][1], points[1][0], points[3][1]
                cls = self.classes.index(bbox['label'])
                bboxes.append([x0, y0, x1, y1, cls])
            filepath = ann_file[:-5] + u".jpg"
            assert os.path.exists(filepath), pprint.pprint(filepath)
            self.objs[filepath] = bboxes
        self.names = list(self.objs.keys())
        self.names.sort()
        train_names, val_names = train_test_split(self.names, test_size=.1, random_state=42)
        if is_train:
            self.names = train_names
        else:
            self.names = val_names

    def at_with_image_path(self, idx):
        filepath = self.names[idx]
        return filepath, np.array(self.objs[filepath])

    def __len__(self):
        return len(self.names)

    def viz(self, indexes=None, font=None):
        import matplotlib.pyplot as plt
        if indexes is None:
            indexes = range(len(self))
        for index in indexes:
            x = self.at_with_image_path(index)

            plot_bbox(np.array(Image.open(x[0])), x[1][:, :4], labels=x[1][:, 4], class_names=self.classes, font=font)
            plt.show()


