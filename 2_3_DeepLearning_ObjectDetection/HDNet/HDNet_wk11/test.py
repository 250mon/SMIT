## -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import cv2, os
import numpy as np
from pycocotools.coco import COCO
import coco_annotation as coano


def main():
    
    #annFile = './annotations//person_keypoints_train2017.json'
    annFile = './annotations//instances_train2017.json'
    imgDir = './train2017'
    ann_color = np.array([[[128, 0, 0]]]) # mid blue
    img_type = True
    
    coco = COCO(annFile)
    
    #catIds = coco.getCatIds() # all category ids
    #catIds = coco.getCatIds(catNms=['person', 'dog']) # specified category ids
    catIds = coco.getCatIds(catNms=['person']) # specified category ids
    imgIds = coco.getImgIds(catIds = catIds)
    
    '''
    cats = coco.loadCats(catIds)
    print('lengh of categories - {:d}'.format(len(cats)))
    
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    
    nms = [cat['supercategory'] for cat in cats]
    print('COCO supercategories: \n{}\n'.format(' '.join(nms)))
    '''            
    cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    
    print('Number of Images Loaded - ', len(imgIds))
    
    for idx in range(len(imgIds)):
                
        img = coco.loadImgs(imgIds[idx])[0]
                
        annIds = coco.getAnnIds(imgIds = imgIds[idx], catIds = catIds, iscrowd=None)
        ann = coco.loadAnns(annIds)
        print ('Length of Annotations - ', len(ann))
        
        image = cv2.imread(os.path.join(imgDir, img['file_name']), cv2.IMREAD_UNCHANGED)
        if len(np.shape(image)) == 2:
            image = np.stack((image, image, image), axis=2)
            
        mask = coano.annotation2binarymask(img['height'], img['width'], ann)
        
        mask = np.expand_dims(mask, axis=2) * ann_color
        
        if img_type:
            image = (np.clip((image + mask), 0, 255)).astype(np.uint8)
        
        cv2.imshow('Images', image)
        
        key = cv2.waitKey(0)
        
        if key == ord('q'):
           break
        elif key == ord('c'):
           print('Image Showing Type is Changed from {} to {}'.format(str(img_type), str(not(img_type))))
           img_type = not(img_type)
    
                    
    
    
if __name__ == '__main__':
    main()