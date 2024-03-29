import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image


REQUIRE_MASK = False
labels = {'aircraft': 1, 'oiltank': 2}




class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./new.json'):
        '''
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.require_mask = REQUIRE_MASK
        self.save_json()


    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            if not json_file == self.save_json_path:
                with open(json_file, 'r') as fp:
                    data = json.load(fp)
                    self.images.append(self.image(data, num))
                    for shapes in data['shapes']:
                        print("label is ")
                        print(shapes['label'])
                        label = shapes['label']
                        #                    if label[1] not in self.label:
                        if label not in self.label:
                            print("find new category: ")
                            self.categories.append(self.categorie(label))
                            print(self.categories)
                            # self.label.append(label[1])
                            self.label.append(label)
                        points = shapes['points']
                        self.annotations.append(self.annotation(points, label, num))
                        self.annID += 1


    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]
        self.height = height
        self.width = width
        return image


    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = label
        #        categorie['supercategory'] = label
        categorie['id'] = labels[label]  # 0 默认为背景
        categorie['name'] = label
        return categorie


    def annotation(self, points, label, num):
        annotation = {}
        print(points)
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]
        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])  # points = [[x1, y1], [x2, y2]] for rectangle
        contour = contour.astype(int)
        area = cv2.contourArea(contour)
        print("contour is ", contour, " area = ", area)
        annotation['segmentation'] = [list(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).flatten())]
        # [list(np.asarray(contour).flatten())]
        annotation['iscrowd'] = 0
        annotation['area'] = area
        annotation['image_id'] = num + 1


        if self.require_mask:
            annotation['bbox'] = list(map(float, self.getbbox(points)))
        else:
            x1 = points[0][0]
            y1 = points[0][1]
            width = points[1][0] - x1
            height = points[1][1] - y1
            annotation['bbox'] = list(np.asarray([x1, y1, width, height]).flatten())


        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation


    def getcatid(self, label):
        for categorie in self.categories:
            #            if label[1]==categorie['name']:
            if label == categorie['name']:
                return categorie['id']
        return -1


    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)


    def mask2box(self, mask):


        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]


        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        return [left_top_c, left_top_r, right_bottom_c - left_top_c, right_bottom_r - left_top_r]


    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask


    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco


    def save_json(self):
        print("in save_json")
        self.data_transfer()
        self.data_coco = self.data2coco()


        print(self.save_json_path)
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)




labelme_json = glob.glob('LabelmeData/*.json')
from sklearn.model_selection import train_test_split


trainval_files, test_files = train_test_split(labelme_json, test_size=0.2, random_state=55)
import os


if not os.path.exists("../projects/SparseRCNN/datasets/coco/annotations"):
    os.makedirs("../projects/SparseRCNN/datasets/coco/annotations/")
if not os.path.exists("../projects/SparseRCNN/datasets/coco/train2017"):
    os.makedirs("../projects/SparseRCNN/datasets/coco/train2017")
if not os.path.exists("../projects/SparseRCNN/datasets/coco/val2017"):
    os.makedirs("../projects/SparseRCNN/datasets/coco/val2017")
labelme2coco(trainval_files, '../projects/SparseRCNN/datasets/coco/annotations/instances_train2017.json')
labelme2coco(test_files, '../projects/SparseRCNN/datasets/coco/annotations/instances_val2017.json')
import shutil


for file in trainval_files:
    shutil.copy(os.path.splitext(file)[0] + ".jpg", "../projects/SparseRCNN/datasets/coco/train2017/")
for file in test_files:
    shutil.copy(os.path.splitext(file)[0] + ".jpg", "../projects/SparseRCNN/datasets/coco/val2017/")