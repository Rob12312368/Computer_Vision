from PIL import Image, ImageDraw
import numpy as np
from typing import Union, Tuple, List


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    height = src_pts_nx2.shape[0]
    A = []
    for i in range(len(src_pts_nx2)):
        xs, ys = src_pts_nx2[i] # may be wrong
        xd, yd = dest_pts_nx2[i]
        yd = height - yd
        ys = height - ys
        #print(xs, ys)
        #print(xd, yd)
        #print()
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd*ys, -xd])
        A.append([0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd])
    A = np.array(A)
    egval, egvect = np.linalg.eig(A.T @ A)
    #print('egvect')
    #print(back)
    #print(egval, egvect)
    return np.reshape(egvect[:,np.argmin(np.abs(egval))], (3,3))
    #raise NotImplementedError


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    #print(src_pts_nx2)
    height = src_pts_nx2.shape[0]
    ans = []
    addCol = np.full((src_pts_nx2.shape[0],1), 1)
    homovector = np.hstack((src_pts_nx2, addCol))
    homovector[:,1] = height - homovector[:,1]
    #print('my home vector:')
    #print(homovector)
    for h in homovector:
        tmp = H_3x3 @ h
        #print(tmp)
        tmp = (tmp/tmp[-1])[:-1]
        tmp[1] = height - tmp[1]
        ans.append(tmp)
    return np.array(ans) 

    raise NotImplementedError


def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''

    height = img1.size[0]
    width = img1.size[1] + img2.size[1]
    concatImg = Image.new('RGB', (width, height), (255,255,255))
    concatImg.paste(img1, (0,0))
    concatImg.paste(img2, (img1.size[1] ,0))
    #concatImg.show()
    draw = ImageDraw.Draw(concatImg)
    for i in range(len(pts1_nx2)):
        #print('each line')
        #print(pts1_nx2[i], pts2_nx2[i])
        draw.line([pts1_nx2[i][0], pts1_nx2[i][1], pts2_nx2[i][0]+width//2, pts2_nx2[i][1]], fill='red', width=3)
    #concatImg.show()
    return np.array(concatImg)
    raise NotImplementedError

# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    
    
    
    
    
    height, width = canvas_shape
    warped = np.zeros((height, width, 3), dtype=np.uint8)
    #coordinates[indexes] = src_img[indexes]
    #height = height + 200
    for i in range(height):
        for j in range(width):
            tmp = np.append(np.array([j, i]), 1)
            srcpoint = destToSrc_H @ tmp
            x,y = (srcpoint / srcpoint[-1])[:-1]
            #print(x,y)
            #print(applyHomography(destToSrc_H, tmp))
            #print('test y x')
            #print(y,x)
            x = np.round(x).astype('int')
            y = np.round(y).astype('int')
            if x >= 0 and x < src_img.shape[1]:
                if y >=0 and y < src_img.shape[0]:
                    #print(src_img[y][x] * 255)
                    #y = src_img.shape[0] - y
                    warped[i][j] = src_img[y][x] * 255
    pil_image = Image.fromarray(warped)

    # Save or display the PIL image
    #pil_image.save("numpy_to_pil.png")
    pil_image.show()

    print(canvas_shape)
    raise NotImplementedError


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image, mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''
    raise NotImplementedError

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    raise NotImplementedError

def stitchImg(*args: Image.Image) -> Image.Image:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    raise NotImplementedError
