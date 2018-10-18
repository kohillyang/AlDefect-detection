#encoding=utf-8
from data.aluminum_material import AluminumDet
if __name__ == '__main__':
    data = AluminumDet(root=u"/data1/zyx/yks/dataset/guangdong_round2_train_20181011", is_train=False)
    data.to_coco("../annotations/instances_aluminumval2018.json")