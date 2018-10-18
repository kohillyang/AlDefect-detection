"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import mxnet as mx
import pycocotools
from gluoncv.utils.bbox import bbox_clip_xyxy,bbox_xywh_to_xyxy
from .bbox_dataset import DetectionDataset
from pycocotools.coco import COCO
__all__ = ['COCODetection']
import numpy as np

class COCODetection(DetectionDataset):
    """MS COCO detection dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.

    """

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('instances_val2017',),
                 images_root = "images",
                 transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True, classes = None):
        super(COCODetection, self).__init__(root)
        self.images_root = images_root

        self._root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._skip_empty = skip_empty
        self._use_crowd = use_crowd
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        # to avoid trouble, we always use contiguous IDs except dealing with cocoapi
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []
        self._items, self._labels = self._load_jsons()
        self.classes = classes
    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        elif len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    def __len__(self):
        return len(self._items)
    def at_with_image_path(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        return img_path,np.array(label)

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []


        for split in self._splits:
            anno = os.path.join(self._root, 'annotations', split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                filename = entry["file_name"]
                dirname = filename.split("_")[1]
                abs_path = os.path.join(self._root, os.path.join(self.images_root,dirname), filename)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                if not label:
                    continue
                items.append(abs_path)
                labels.append(label)
        return items, labels

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj['area'] < self._min_object_area:
                continue
            if obj.get('ignore', 0) == 1:
                continue
            if not self._use_crowd and obj.get('iscrowd', 0):
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] > 0 and xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])
        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1])
        return valid_objs
if __name__ == '__main__':
    da = COCODetection(root="/data1/zyx/yks/dataset/coco",splits=('instances_train2014',"instances_valminusminival2014")).to_roidb("cache/coco_train35k.roidb")