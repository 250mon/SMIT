import numpy as np
import cv2, os
import PIL.Image as Image
import coco_annotation as coanno

from pycocotools.coco import COCO

#dataDir='C:/Project/CloudStation/deeplearning/data/COCO'
dataDir='.'
dataType='train2017'
annFile='{}\\annotations2017\\instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)
#annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
#coco_kps=COCO(annFile)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )

# print('--imgIds -------------------------------')
# print(imgIds)

# imgIds = coco.getImgIds(imgIds = imgIds[0])
# print('--imgIds -------------------------------')
# print(imgIds)

#cv2.namedWindow('mask_window', cv2.WINDOW_AUTOSIZE)
print("Number of Images : ", len(imgIds))
if not os.path.exists('.\\coco'):
    os.makedirs('.\\coco')
if not os.path.exists('.\\coco\\tall'):
    os.makedirs('.\\coco\\tall')
if not os.path.exists('.\\coco\\wide'):
    os.makedirs('.\\coco\\wide')
if not os.path.exists('.\\coco\\tall\\small'):
    os.makedirs('.\\coco\\tall\\small')
if not os.path.exists('.\\coco\\tall\\normal'):
    os.makedirs('.\\coco\\tall\\normal')
if not os.path.exists('.\\coco\\wide\\small'):
    os.makedirs('.\\coco\\wide\\small')
if not os.path.exists('.\\coco\\wide\\normal'):
    os.makedirs('.\\coco\\wide\\normal')

dir1 = '.'
dir2 = 'coco'

for i in range(len(imgIds)): 
    img = coco.loadImgs(imgIds[i])[0]
    
    orig = Image.open('.\\train2017\\'+img['file_name'])
    #orig = Image.open('.\\train2014\\'+img['file_name'])
    if len(np.shape(orig)) is not 3:
        print("Not RGB, shape was -", np.shape(orig))
        orig = np.reshape(orig, np.shape(orig)+(1,))
        orig = np.concatenate((orig, orig, orig), axis=2)
        orig = Image.fromarray(orig.astype(np.uint8))
    
    fileName = img['file_name'][:-1].split('.')[0] + "_mask.png"
    height = img['height']
    width = img['width']
    #print('--imgIds {}, file_name={},{},{}'.format(imgIds[i],fileName,height,width))
    #print(img)

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = coanno.annotation2binarymask(height, width, anns)
    mask = Image.fromarray(mask.astype(np.uint8))
    #print("Mask Shape", np.shape(mask))
    #print("Image Shape", np.shape(orig))
    #sh_img = orig*mask
    
    
    #orig.putalpha(Image.fromarray((mask).astype(np.uint8)))
    #print("Image Shape", np.shape(orig))
    
    #cv2.imshow('mask_window', orig)
    #image = Image.fromarray(orig.astype(np.uint8))
    # Resize as appropriately (considering the mask size - remove too small mask)
    
    if height > width: #portrait
        factor = max(480./float(width), 640./float(height))
        size = (np.array(orig.size) * factor + (0.5, 0.5)).astype(np.int32)
        orig = orig.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.NEAREST)
        
        left = (size[0] - 480)//2
        right = left + 480
        upper = (size[1] - 640)//2
        lower = upper + 640
        
        image = np.array(orig.crop((left, upper, right, lower)))
        mask = mask.crop((left, upper, right, lower))
        mask = np.reshape(np.array(mask), np.shape(mask)+(1,))
        
        dir3 = 'tall'
    else:
        factor = max(480./float(height), 640./float(width))
        size = (np.array(orig.size) * factor + (0.5, 0.5)).astype(np.int32)
        orig = orig.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.NEAREST)
        
        left = (size[0] - 640)//2
        right = left + 640
        upper = (size[1] - 480)//2
        lower = upper + 480
        
        image = np.array(orig.crop((left, upper, right, lower)))
        mask = mask.crop((left, upper, right, lower))
        mask = np.reshape(np.array(mask), np.shape(mask)+(1,))
    
        dir3 = 'wide'
            
    mask_size = np.sum(mask)
    if mask_size < 640*480/40:
        dir4 = 'small'
    else:
        dir4 = 'normal'
    
    
    #s_image = Image.fromarray((mask*image).astype(np.uint8))
    s_image = np.concatenate((image, mask*255), axis=2)
    s_image = Image.fromarray(s_image.astype(np.uint8))
    fileName = os.path.join(dir1, dir2, dir3, dir4, fileName)
    s_image.save(fileName)
    print('step {:d}: Processed - {:s}'.format(i, fileName))
    
    #orig = np.concatenate((orig, mask), axis=2)
    #image.show()
    #orig.show()
    #image.save('.\\'+fileName)
    #q = cv2.waitKey(0)
    #if q == ord('q'):
    #    break
 
    

#coco.showAnns(anns)

