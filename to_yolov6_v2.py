import sys
import os
import json
import cv2
import random
import shutil
from pathlib import Path
from to_coco import COCOCreater

class YoloV6Creater:

    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.src_label_file = os.path.join(self.src_dir, 'label.txt')
        for file in os.listdir(dst_dir):
            if str(file) != '.' and str(file) != '..':
                print('error: the dst_dir', dst_dir, ' should be empty')
                exit(0)
                
    def read_ori_labels(self, shuffle=True, train_ratio=0.8):
        print("trans to coco: start read ori labels")
        labels = []
        with open(self.src_label_file, 'r') as f:
            for line in f.readlines():
                labels.append(line.strip('\r\n'))
        if shuffle:
            random.shuffle(labels)
        
        train_count = int(len(labels) * train_ratio)
        val_count = len(labels) - train_count
        if train_count == 0:
            self.ori_train_labels = []
            self.ori_val_labels = labels
        elif val_count == 0:
            self.ori_train_labels = labels
            self.ori_val_labels = []
        else:
            self.ori_train_labels = labels[0: train_count]
            self.ori_val_labels = labels[train_count:]
        return self.ori_train_labels, self.ori_val_labels
       
    def create(self, shuffle=True, train_ratio=0.8):
        # 创建images和labels文件夹
        images_path = os.path.join(self.dst_dir, 'images')
        images_train_path = os.path.join(images_path, 'train2017')
        images_val_path = os.path.join(images_path, 'val2017')
        labels_path = os.path.join(self.dst_dir, 'labels')
        labels_train_path = os.path.join(labels_path, 'train2017')
        labels_val_path = os.path.join(labels_path, 'val2017')
        if not os.path.exists(images_path):
            os.mkdir(images_path)
        if not os.path.exists(images_train_path):
            os.mkdir(images_train_path)
        if not os.path.exists(images_val_path):
            os.mkdir(images_val_path)
        if not os.path.exists(labels_path):
            os.mkdir(labels_path)
        if not os.path.exists(labels_train_path):
            os.mkdir(labels_train_path)
        if not os.path.exists(labels_val_path):
            os.mkdir(labels_val_path)
            
        for i, line in enumerate(self.ori_train_labels ):
            fileds = line.split(' ')
            img_name = fileds[0]
            ori_img_path = os.path.join(self.src_dir,  img_name)
            new_img_path = os.path.join(images_train_path, img_name)
            shutil.copy(ori_img_path, new_img_path)
            label_name = Path(img_name).stem + '.txt'
            label_path = os.path.join(labels_train_path, label_name)
            with open(label_path, 'w') as f:
                if len(fileds) > 2:
                    box_infos = fileds[1:]
                    assert(len(box_infos) % 5 == 0)
                    box_count = int(len(box_infos) / 5)
                    img = cv2.imread(ori_img_path)
                    h, w, c = img.shape
                    for i in range(box_count):
                        box_info = box_infos[i*5:i*5+5]
                        x0, y0, x1, y1, cls = box_info
                        xc = (float(x0) + float(x1)) / 2.0 / w
                        yc = (float(y0) + float(y1)) / 2.0 / h
                        w_n = (float(x1) - float(x0)) / w
                        h_n = (float(y1) - float(y0)) / h
                        cls = int(float(cls))
                        f.writelines('%d %.6f %.6f %.6f %.6f\n'%(cls, xc, yc, w_n, h_n))
        
        for i, line in enumerate(self.ori_val_labels):
            fileds = line.split(' ')
            img_name = fileds[0]
            ori_img_path = os.path.join(self.src_dir, img_name)
            new_img_path = os.path.join(images_val_path, img_name)
            shutil.copy(ori_img_path, new_img_path)
            label_name = Path(img_name).stem + '.txt'
            label_path = os.path.join(labels_val_path, label_name)
            with open(label_path, 'w') as f:
                if len(fileds) > 2:
                    box_infos = fileds[1:]
                    assert(len(box_infos) % 5 == 0)
                    box_count = int(len(box_infos) / 5)
                    img = cv2.imread(ori_img_path)
                    h, w, c = img.shape
                    for i in range(box_count):
                        box_info = box_infos[i*5:i*5+5]
                        x0, y0, x1, y1, cls = box_info
                        xc = (float(x0) + float(x1)) / 2.0 / w
                        yc = (float(y0) + float(y1)) / 2.0 / h
                        w_n = (float(x1) - float(x0)) / w
                        h_n = (float(y1) - float(y0)) / h
                        cls = int(float(cls))
                        f.writelines('%d %.6f %.6f %.6f %.6f\n'%(cls, xc, yc, w_n, h_n))
        
        
if __name__ == '__main__':
    trans = YoloV6Creater('../gongchengche-std', '../gongchengche-yolov6')
    trans.read_ori_labels(shuffle=True, train_ratio=0.8)
    trans.create(shuffle=True, train_ratio=0.8)




