import csv
from pycocotools.coco import COCO
import requests

coco = COCO('cocoapi/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]

catIds = coco.getCatIds(catNms=['car'])
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

for im in images:
    #print("im: ", im)
    img_data = requests.get(im['coco_url']).content
    with open('test_xx_downloaded_images/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)


with open('test_xx_annotations_download_' + 'car_bus' + '.csv', mode='w', newline='') as annot:
    for im in images:
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for i in range(len(anns)):
            annot_writer = csv.writer(annot)
            annot_writer.writerow(['downloaded_images/' + im['file_name'], int(round(anns[i]['bbox'][0])), int(round(anns[i]['bbox'][1])),
                        int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2])), int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3])), 'car'])
