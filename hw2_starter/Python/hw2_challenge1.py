import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from skimage.measure import label
from PIL import Image
import numpy as np
from skimage.morphology import dilation, erosion

def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    binary_img = gray_img > threshold
    binary_img = smooth_edge(binary_img, 3)
    labeled_img = label(binary_img)
    #print(labeled_img)
    #print(np.max(labeled_img))
    return labeled_img
def draw_orientation_line(degree, x, y, canva):
    #print(np.degrees(radian))
    #print(np.cos(radian))
    #radian = radian
    radian = np.radians(degree)
    hypo_len = 50
    xlist = [x]
    ylist = [y]
    xlist.append(x + hypo_len * np.cos(radian))
    #xlist.append(x - hypo_len * np.cos(radian))
    ylist.append(y + hypo_len * np.sin(radian))
    #ylist.append(y - hypo_len * np.sin(radian))
    canva.plot(xlist, ylist, linestyle='--', color='red')
    canva.plot(x, y, marker='o', color='blue')
def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''
    # centroid
    value_max = np.max(labeled_img)
    labels = [i for i in range(value_max+1)]
    area = np.array([0] * (value_max+1)) # index 0 is not used
    center_x = np.array([0] * (value_max+1))
    center_y = np.array([0] * (value_max+1))
    for i in range(labeled_img.shape[0]):
        for j in range(labeled_img.shape[1]):
            if labeled_img[i][j] == 0:
                continue
            area[labeled_img[i][j]] += 1
            center_x[labeled_img[i][j]] += i
            center_y[labeled_img[i][j]] += j
    center_x = center_x / area
    center_y = center_y / area
    #plt.figure()
    #plt.imshow(orig_img, cmap='gray')
    #plt.scatter(center_y, center_x, c='red')
    #plt.show()

    # orientation
    a,b,c = np.array([0] * (value_max+1)), np.array([0] * (value_max+1)), np.array([0] * (value_max+1))
    for i in range(labeled_img.shape[0]):
        for j in range(labeled_img.shape[1]):
            if labeled_img[i][j] == 0:
                continue
            a[labeled_img[i][j]] += np.square(i - center_x[labeled_img[i][j]])
            b[labeled_img[i][j]] += 2 * (i - center_x[labeled_img[i][j]]) * (j - center_y[labeled_img[i][j]])
            c[labeled_img[i][j]] += np.square(j - center_y[labeled_img[i][j]])
    orientation = np.arctan2(b, a-c) / 2

    # minimum inertia
    minimum_inertia = a * np.sin(orientation) * np.sin(orientation) - b * np.sin(orientation) * np.cos(orientation) + c * np.cos(orientation) * np.cos(orientation)
    # roundness
    orientation_diag = orientation + np.pi / 2
    maximum_inertia =  a * np.sin(orientation_diag) * np.sin(orientation_diag) - b * np.sin(orientation_diag) * np.cos(orientation_diag) + c * np.cos(orientation_diag) * np.cos(orientation_diag)
    roundness = minimum_inertia / maximum_inertia

    # change orientation from radians to degrees
    orientation = np.degrees(orientation)
    orientation = 90 - orientation

    orientation[orientation>90] = orientation[orientation>90] - 180
    #print('Orientation',orientation)
    result = np.column_stack((labels, center_y, center_x, minimum_inertia, orientation, roundness))
    

    return result[1:]


def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''
    #print(obj_db)
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    #plt.show()
    orig = compute2DProperties(labeled_img, labeled_img)
    #print(orig)
    for i in orig:
        for j in obj_db:
            print(abs(i[-1]-j[-1]))
            if abs(i[-1]-j[-1]) <= 0.09:
                #print('Orientation', i[-2])
                if min(i[3],j[3]) / max(i[3],j[3]) >= 0.8:
                    draw_orientation_line(i[-2], i[1], i[2], ax)
                #if min(i[3],j[3]) / max(i[3],j[3]) >= 0.8:
            
    #plt.show()
    plt.savefig(output_fn)
    #plt.show()
    return

def smooth_edge(arr: np.array, size: int):
    selem = np.ones((size, size))
    dilate = dilation(arr, footprint=selem)
    erode = erosion(dilate, footprint=selem)
    return erode
def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [0.5, 0.5, 0.5]   # You need to find the right thresholds

    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    for row in obj_db:
        #print(row[2], row[1], row[-2])
        draw_orientation_line(row[-2], row[1], row[2], ax)
    # for i in range(obj_db.shape[0]):
        # plot the position
        # plot the orientation
    plt.savefig('outputs/two_objects_properties.png')
    #plt.show()

def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')

#img = Image.open(f'data/two_objects.png')
#labeled = generateLabeledImage(np.array(img.convert('L')),125)
#database = compute2DProperties(img,labeled)

#img = Image.open(f'data/many_objects_2.png')
#labeled = generateLabeledImage(np.array(img.convert('L')),125)
#recognizeObjects(img,labeled,database,'result.png')
