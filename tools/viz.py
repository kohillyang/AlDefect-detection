#encoding=utf-8
from data.aluminum_material import AluminumDet
from utils.plot_bbox import get_chinese_font
if __name__ == '__main__':
    font = get_chinese_font()
    data = AluminumDet(root=u"/data1/zyx/yks/dataset/guangdong_round2_train_20181011")
    data.viz(font=font)