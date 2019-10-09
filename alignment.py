import math
import random

import cv2
import numpy as np

# added libs
import itertools
from random import shuffle
# added libs

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    #BEGIN TODO 2
    #Fill in the matrix A with the appropriate entries baed on
    #the contents of f1, f2, and matches. Recall that cv2.KeyPoint
    #objects store a pt field with the (x,y) pixel coordinates of
    #the feature.
    #TODO-BLOCK-BEGIN
   
    # Loop through matches list, filling in A with coord vals from f1 and f2
    A_index = 0
    for match in matches:
        x1, y1 = f1[match.queryIdx].pt
        x2, y2 = f2[match.trainIdx].pt
        A[A_index][0] = x1
        A[A_index][1] = y1
        A[A_index][2] = 1
        A[A_index][6] = -x2*x1
        A[A_index][7] = -x2*y1
        A[A_index][8] = -x2
        A_index += 1
        A[A_index][3] = x1
        A[A_index][4] = y1
        A[A_index][5] = 1
        A[A_index][6] = -y2*x1
        A[A_index][7] = -y2*y1
        A[A_index][8] = -y2
        A_index += 1

    #TODO-BLOCK-END

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    # Getting the row of Vt that corresponds to smallest singular val
    H = np.reshape(Vt[-1], (3,3), order='C')
    return H

    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''
    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN


    match_len = len(matches)
    max_inliers = -1
    
    if m == eHomography:
        for i in range(nRANSAC):
            four_match = np.random.choice(matches, 4)

            # fit a model to samples
            M = computeHomography(f1,f2,four_match)

            # count number of inliers
            inlier_indices = getInliers(f1,f2,matches,M,RANSACthresh)
            n_inliers = len(inlier_indices)

            # update max inlier if no longer max
            if max_inliers < n_inliers:
                max_inliers = n_inliers
                max_inlier_ind = inlier_indices

    elif m == eTranslate:
        for i in range(nRANSAC):
            # get randomly selected samples
            sing_match = np.random.choice(matches,1)
            delta_x = f2[sing_match[0].trainIdx].pt[0] - f1[sing_match[0].queryIdx].pt[0]
            delta_y = f2[sing_match[0].trainIdx].pt[1] - f1[sing_match[0].queryIdx].pt[1]

            # fit a model to samples
            M = np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])

            # count number of inliers
            inlier_indices = getInliers(f1,f2,matches,M,RANSACthresh)
            n_inliers = len(inlier_indices)

            # update max inlier if no longer max
            if max_inliers < n_inliers:
                max_inliers = n_inliers
                max_inlier_ind = inlier_indices

    if len(max_inlier_ind) > 0:
        M = leastSquaresFit(f1,f2,matches,m,max_inlier_ind)
    else:
        M = leastSquaresFit(f1,f2,matches,m,[i for i in range(match_len)])
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []
    point = np.ones((3,1))
    for i in range(len(matches)):
        #BEGIN TODO 5
        # Determine if the ith matched feature, when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        #TODO-BLOCK-BEGIN

        #Transform the matched features in f1 by M.
        point[0] = f1[matches[i].queryIdx].pt[0]
        point[1] = f1[matches[i].queryIdx].pt[1]
        pt = np.array ([f1[matches[i].queryIdx].pt])
        #tran_pt = np.matmul (M, np.append (pt, 1)) [0:2]
        tran_pt = np.matmul (M, np.append (pt, 1)) [0:3]
        tran_pt = (tran_pt / tran_pt[2])[0:2]

        new_pt = np.array ([f2[matches[i].trainIdx].pt])
        new_dist = np.linalg.norm( new_pt - tran_pt)
        #new_dist = np.sqrt((tran_pt[0] - f2[matches[i].trainIdx].pt[0])**2 + (tran_pt[1] - f2[matches[i].trainIdx].pt[1])**2)

        # Store if transformed distance is less than RANSAC threshold
        if new_dist <= RANSACthresh:
            inlier_indices.append(i)

        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0
         
        #BEGIN TODO 6
        #Compute the average translation vector over all inliers.
        #Fill in the appropriate entries of M to represent the
        #average translation transformation.
        #TODO-BLOCK-BEGIN

        num_inliers = len(inlier_indices)
        for index in inlier_indices:
            u += f2[matches[index].trainIdx].pt[0] - f1[matches[index].queryIdx].pt[0]
            v += f2[matches[index].trainIdx].pt[1] - f1[matches[index].queryIdx].pt[1]
        M[0][2] = u / num_inliers
        M[1][2] = v / num_inliers 
        return M
        #TODO-BLOCK-END
        #END TODO

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        
        filtered_matches = []
        for index in inlier_indices:
            filtered_matches.append(matches[index])
        return computeHomography(f1, f2, filtered_matches)

        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

