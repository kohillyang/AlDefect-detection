#encoding=utf-8
from __future__ import print_function
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse
from pprint import pprint
def parse_args():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-l', dest = "label", default=u"../annotations/instances_aluminumval2018.json")
    parser.add_argument('-p', dest = "predict", default=u"../annotations/format_submit_validation_example.json")
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    cocoGt = COCO(args.label)
    #create filename to imgid
    catIds = cocoGt.getCatIds()
    catid2catbane = {entry["id"]:entry["name"] for entry in cocoGt.cats.values()}

    imgIds = cocoGt.getImgIds()
    imgs = cocoGt.loadImgs(imgIds)
    filename2imgid = {entry["file_name"]:entry["id"] for entry in imgs}
    submit_validataion = json.load(open(args.predict,"rt"),encoding="utf-8")["results"]
    coco_results = []
    for onefile in submit_validataion:
        #{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
        filename = onefile["filename"]
        for rect in onefile["rects"]:
            coco_results.append({"image_id":filename2imgid[filename],
                                 "category_id":int(rect["label"][-1]) + 1,
                                 "bbox":[rect["xmin"],rect["ymin"],rect["xmax"]-rect["xmin"] + 1,rect["ymax"]-rect["ymin"] + 1],
                                 "score":rect["confidence"]
                                 })
    json.dump(coco_results,open("tmp.json","wt"))
    cocoEval = COCOeval(cocoGt,cocoGt.loadRes("tmp.json"),"bbox")
    cocoEval.params.imgIds  = imgIds
    mAP_eachclasses = {}
    for catId in catIds:
        print( u"Evaluate %s"%(catid2catbane[catId]))
        cocoEval.params.catIds = [catId]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP_eachclasses[catid2catbane[catId]] = cocoEval.stats[1]
    print( u"Evaluate all classes.")
    cocoEval.params.catIds = catIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP_eachclasses[u"mAP"] = cocoEval.stats[1]
    print("************summary***************")
    for k in mAP_eachclasses.keys():
        print (k,mAP_eachclasses[k])