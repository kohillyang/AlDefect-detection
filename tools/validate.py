#encoding=utf-8
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pprint import pprint
if __name__ == '__main__':
    cocoGt = COCO(u"../annotations/instances_aluminumval2018.json")
    #create filename to imgid
    catIds = cocoGt.getCatIds()
    catid2catbane = {entry["id"]:entry["name"] for entry in cocoGt.cats.values()}

    imgIds = cocoGt.getImgIds()
    imgs = cocoGt.loadImgs(imgIds)
    filename2imgid = {entry["file_name"]:entry["id"] for entry in imgs}
    submit_validataion = json.load(open(u"../annotations/format_submit_validation_example.json","rt"),encoding="utf-8")["results"]
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
    # for i in range(len())
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    for catId in catIds:
        print( u"Evaluate %s"%(catid2catbane[catId]))
        cocoEval.params.catIds = [catId]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
