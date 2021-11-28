import numpy
import cv2


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

def test(h,w):
    mask = numpy.zeros((h, w), numpy.uint8)
    return mask