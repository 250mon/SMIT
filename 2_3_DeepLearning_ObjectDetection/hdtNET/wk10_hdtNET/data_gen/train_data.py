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
    
    annFile = './annotations/person_keypoints_train2017.json'
    #annFile = './annotations//instances_train2017.json'
    imgDir = './train2017'
    
    trainDir = './Train'
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    
    ann_color = np.array([[[128, 0, 0]]]) # mid blue
    img_type = True
    
    coco = COCO(annFile)
    
    catIds = coco.getCatIds(catNms=['person']) # specified category ids
    '''
    cats = coco.loadCats(catIds)
    print('Number of Categories Loaded - ', len(cats))
    for id2 in range(len(cats)):
        print(cats[id2]['keypoints'])
    '''            
    imgIds = coco.getImgIds(catIds = catIds)
        
    #cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    
    print('Number of Images Loaded - ', len(imgIds))
    
    for idx in range(len(imgIds)):
                
        img = coco.loadImgs(imgIds[idx])[0]
                        
        annIds = coco.getAnnIds(imgIds = imgIds[idx], catIds = catIds, iscrowd=None)
        ann = coco.loadAnns(annIds)
        #print ('Length of Annotations - ', len(ann))
        
        image = cv2.imread(os.path.join(imgDir, img['file_name']), cv2.IMREAD_UNCHANGED)
        new_name = img['file_name'].split('.')[0] + "_mask.png"
                
        if len(np.shape(image)) == 2:
            image = np.stack((image, image, image), axis=2)
            
        mask = coano.annotation2binarymask2(img['height'], img['width'], ann, 1/35)
        
        if np.sum(mask) == 0:
            #print('{} is skipped'.format(img['file_name']))
            continue
        
        #mask = np.expand_dims(mask, axis=2) * ann_color
        mask = np.clip(np.expand_dims(mask, axis=2)*255, 0, 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(trainDir, new_name), np.concatenate((image, mask), axis=2))
        print('Writing - ', new_name , ' at ', idx)
        '''
        if img_type:
            image = (np.clip((image + mask), 0, 255)).astype(np.uint8)
        
        cv2.imshow('Images', image)
        
        key = cv2.waitKey(0)
        
        if key == ord('q'):
           break
        elif key == ord('c'):
           print('Image Showing Type is Changed from {} to {}'.format(str(img_type), str(not(img_type))))
           img_type = not(img_type)
           
        '''
    
                    
    
    
if __name__ == '__main__':
    main()