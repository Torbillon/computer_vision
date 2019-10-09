import math
import sys

import cv2
import numpy as np
import matplotlib as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plts


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that that takes an image and a
       transform, and computes the bounding box of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8


    #TODO-BLOCK-BEGIN
    y,x,rgb = img.shape
    # lower left
    pt_matrix = np.zeros((3,4))
    pt_matrix[2] = 1
    # upper left
    pt_matrix[0][1] = 0
    pt_matrix[1][1] = y - 1
    # lower right
    pt_matrix[0][2] = x - 1
    pt_matrix[1][2] = 0
    # upper right
    pt_matrix[0][3] = x - 1
    pt_matrix[1][3] = y - 1


    # apply transformation
    tran_pts = M.dot(pt_matrix)
    tran_pts = np.divide(tran_pts[:2,:],tran_pts[2:3,:])
    # get mins/maxes
    minX = np.amin(tran_pts[0])
    minY = np.amin(tran_pts[1])
    maxX = np.amax(tran_pts[0])
    maxY = np.amax(tran_pts[1])

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """

    img = img.astype(np.float32) / 255.0
    #plts.imshow(img,interpolation='nearest')
    #plts.show()
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    # --------------- Do edge feathering ----------------------
    # DONE
    row = img.shape[0]
    col = img.shape[1]
    # fill in edge slices
    hat = np.zeros((row,col))
    slope = 1 / float(blendWidth + 1)
    for i in range(blendWidth):
        hat[:, i:(col-1) - i] += slope
    # fill in inner portion
    hat[:, blendWidth:col - blendWidth] = np.ones((row, col - (2 * blendWidth)))

    # mask
    sum_axis = np.sum(img[:,:,:3],axis=2).reshape(row,col)
    mask = sum_axis != 0
    #mask2 = sum_axis == 0
    hat = hat * mask
    #hat = hat + (mask2 * 1)


    # make hat function 3 dimensional for RGB channels
    hat_fun = np.expand_dims(hat,axis=2)
    hat_fun = np.concatenate((hat_fun,hat_fun,hat_fun), axis=2)
    img = img * hat_fun
    # add alpha channel to img
    hat = np.expand_dims(hat, axis=2)
    img = np.concatenate((img, hat), axis=2)
    
    #plts.imshow(img[:,:,:3],interpolation='nearest')
    #plts.show()

    # -------------- Create maps for remap() ------------------
    acc_h, acc_w = acc.shape[:2]
    # Make array containing index values
    y_indices, x_indices = np.indices((acc_h,acc_w), dtype=np.float32)
    indices_arr = np.array([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel()])
    # Do transformation
    tran_ind = np.linalg.inv(M).dot(indices_arr)
    # Get first two dimensions, divide by third to ensure of the form [x,y,1]
    x_map, y_map = tran_ind[:-1]/tran_ind[-1]
    # Reshape to match acc dims
    x_map = x_map.reshape(acc_h, acc_w).astype(np.float32)
    y_map = y_map.reshape(acc_h, acc_w).astype(np.float32)

    # -------------- Remap -----------------------------------
    final_img = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR)
    #print("",final_img.shape)
    #plts.imshow(final_img[:,:,:3],interpolation='nearest')
    #plts.show()
    #print('acc shape', acc.shape)
    acc += final_img

    #TODO-BLOCK-END
    # END TODO

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    # get alpha matrix
    alpha_channel = np.reshape(np.array(acc[:,:,3]), (acc.shape[0],acc.shape[1],1))
    # extend alpha matrix to depths of 3
    alpha_3 = np.concatenate((alpha_channel, alpha_channel, alpha_channel),axis=2)
    # normalize
    mask = alpha_3 == 0

    # set all 0's to 1's
    alpha_nozeros = mask * 1 + alpha_3
    img = np.divide(acc[:,:,:3], alpha_nozeros)
    # set alpha channel to oplique
    opique_mtx = np.ones(img[:,:,2].shape)
    img = np.concatenate((img, np.reshape(opique_mtx, (opique_mtx.shape[0],opique_mtx.shape[1],1))), axis=2)
    #TODO-BLOCK-END
    # END TODO
    return (img * 255).astype(np.uint8)


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and returns useful information about the
       accumulated image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and
             transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all
             tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all
             tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = -minX
    maxY = -minY
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        #TODO-BLOCK-BEGIN
        t_minX, t_minY, t_maxX, t_maxY = imageBoundingBox(img,M)
        if t_minX < minX:
            minX = t_minX
        if t_minY < minY:
            minY = t_minY
        if t_maxX > maxX:
            maxX = t_maxX
        if t_maxY > maxY:
            maxY = t_maxY
        
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation



def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        temp_acc = np.copy(acc)
        accumulateBlend(img, acc, M_trans, blendWidth)
        

    return acc


def getDriftParams(ipv, translation, width):
    """ Computes parameters for drift correction.
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         translation: transformation matrix so that top-left corner of accumulator image is origin
         width: Width of each image(assumption: all input images have same width)
       OUTPUT:
         x_init, y_init: coordinates in acc of the top left corner of the
            panorama with half the left image cropped out to match the right side
         x_final, y_final: coordinates in acc of the top right corner of the
            panorama with half the right image cropped out to match the left side
    """
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    if is360:
        # BEGIN TODO 12
        # 497P: you aren't required to do this. 360 mode won't work.
        # 597P: fill in appropriate entries in A to trim the left edge and
        # to take out the vertical drift. Shift the image left by half the
        # image width, and add a shear to horizontally align the init and final
        # coordinates.
        # Note: warpPerspective does forward mapping which means A is an affine
        # transform that maps accumulator coordinates to final panorama coordinates
        #TODO-BLOCK-BEGIN
        trans = np.array([[1, 0, -(width / 2.0)],[0, 1, 0],[0,0,1]])
        # FIXME
        # what is a?
        a = (float(y_final) - float(y_init)) / (float(x_final) - float(x_init))
        a *= -1.0
        shear = np.array([[1, 0, 0],[a,1,0],[0,0,1]])
        A = np.dot(shear,trans)
        # raise Exception("TODO in blend.py not implemented")
        #TODO-BLOCK-END
        # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

