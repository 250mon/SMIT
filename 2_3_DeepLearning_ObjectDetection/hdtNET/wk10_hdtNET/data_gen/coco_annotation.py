import numpy, cv2


def decodeSeg(mask, segmentations):
    """
    Draw segmentation
    """
    pts = [
        numpy
            .array(anno)
            .reshape(-1, 2)
            .round()
            .astype(int)
        for anno in segmentations
    ]
    mask = cv2.fillPoly(mask, pts, 1)

    return mask

def decodeRl(mask, rle):
    """
    Run-length encoded object decode
    """
    mask = mask.reshape(-1, order='F')

    last = 0
    val = True
    for count in rle['counts']:
        val = not val
        mask[last:(last+count)] |= val
        last += count

    mask = mask.reshape(rle['size'], order='F')
    return mask

def annotation2binarymask(h, w, annotations):
    mask = numpy.zeros((h, w), numpy.uint8)
    for annotation in annotations:
        segmentations = annotation['segmentation']
        if isinstance(segmentations, list): # segmentation
            mask = decodeSeg(mask, segmentations)
        else:                               # run-length
            mask = decodeRl(mask, segmentations)
    return mask

def annotation2binarymask2(h, w, annotations, thres):
    
    mask = numpy.zeros((h, w), numpy.uint8)
    zero = numpy.zeros((h, w), numpy.uint8)
    thres *= (h*w)
    
    if max(h,w) < 512:
        #print('Too Small Image, so Skipped')
        return zero
    
    for annotation in annotations:
        segmentations = annotation['segmentation']
        
        if annotation['keypoints'][2] == 0:
            #print('No Nose, so Skipped')
            return zero
        
        maskT = numpy.zeros((h, w), numpy.uint8)
        
        if isinstance(segmentations, list): # segmentation
            maskT = decodeSeg(maskT, segmentations)
        else:                               # run-length
            maskT = decodeRl(maskT, segmentations)
                    
        if numpy.sum(maskT) < thres:
            #print('Threshold, so Skipped')
            return zero
        else:
            mask += maskT
        
    return mask

def test(h,w):
    mask = numpy.zeros((h, w), numpy.uint8)
    return mask