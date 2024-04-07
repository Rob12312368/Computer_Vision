from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import skimage.measure
def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
    # Find the center and radius of the sphere
    # Input:
    #   img - the image of the sphere
    # Output:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    threshold = skimage.filters.threshold_otsu(img)
    binary_image = (img > threshold).astype('int')
    #test = Image.fromarray(binary_image)
    #test.show()
    props = skimage.measure.regionprops(binary_image)
    radius = np.sqrt((props[0].area) / np.pi)
    return (props[0].centroid, radius)
    raise NotImplementedError

def computeLightDirections(center: np.ndarray, radius: float, images: List[np.ndarray]) -> np.ndarray:
    # Compute the light source directions
    # Input:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    #   images - list of N images
    # Output:
    #   light_dirs_5x3 - 5x3 matrix of light source directions
    
    matrix = np.zeros((5,3))
    for i, img in enumerate(images):
        max_index = np.unravel_index(np.argmax(img), img.shape)
        #print(max_index, center)
        y, x = max_index
        z = np.sqrt(radius ** 2 - (y-center[0]) ** 2 - (x-center[1]) ** 2)
        vector = np.array([x-center[1],y-center[0],z])
        # change coordinate
        vector[1] = vector[1]
        vector[2] = vector[2]
        # scale up or down to the magnitude of the light source
        # make it a vector with magnitude 1
        vector = vector / (np.sqrt(np.sum(np.square(vector))))
        vector = vector * img[y,x]
        #print(img[y,x], (np.sqrt(np.sum(np.square(vector)))))
        #print(vector)
        matrix[i] = vector
    return matrix
    raise NotImplementedError

def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask
    images = np.array(images)
    mask = np.sum(images,axis=0) > 0
    mask = mask.astype('int')
    return mask
    raise NotImplementedError

def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Compute the surface normals and albedo of the object
    # Input:
    #   light_dirs - Nx3 matrix of light directions
    #   images - list of N images
    #   mask - binary mask
    # Output:
    #   normals - HxWx3 matrix of surface normals
    #   albedo_img - HxW matrix of albedo values
    print(light_dirs)
    height = images[0].shape[0]
    width = images[0].shape[1]
    normals = np.zeros((height, width, 3))
    albedos = np.zeros((height, width))
    J_mag = np.sqrt(np.sum(np.square(light_dirs), axis = 1))
    J = np.diag(J_mag)
    pseudo = np.linalg.inv(np.transpose(light_dirs) @ light_dirs) @ np.transpose(light_dirs) @ np.linalg.inv(J)
    for i in range(height):
        for j in range(width):
            if not mask[i,j]:
                continue
            I = [images[0][i,j], images[1][i,j], images[2][i,j], images[3][i,j], images[4][i,j]]
            N = pseudo @ I
            N_mag = np.sqrt(np.sum(np.square(N)))
            n = N / N_mag
            normals[i,j] = n
            albedos[i,j] = N_mag
    print(normals)
    print(albedos)
    print(normals.shape)
    normals[:,:,2] = np.abs(normals[:,:,2])
    return [normals, albedos/np.max(albedos)]
    raise NotImplementedError

