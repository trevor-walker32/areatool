import sys
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids



def main(clargs: [str]):
    im_path = "./" + clargs[0]
    my_img = cv2.imread(im_path, 0)

    top_left_corner = cv2.imread('./top_left.png', 0)
    top_right_corner = cv2.imread('./top_right.png', 0)
    bottom_left_corner = cv2.imread('./bottom_left.png', 0)
    bottom_right_corner = cv2.imread('./bottom_right.png', 0)

    corners = []
    corners.append(find_location(my_img, bottom_left_corner, 'bottom_left'))
    corners.append(find_location(my_img, bottom_right_corner, 'bottom_right'))
    corners.append(find_location(my_img, top_right_corner, 'top_right'))
    corners.append(find_location(my_img, top_left_corner, 'top_left'))

    plot_and_save(my_img, corners)



def find_location(image: np.array, template: np.array, location_type: str):
    locs = template_match(image, template, location_type)
    meds = find_kmedoids(locs)

    return locs[0]



def find_kmedoids(locations, clusters=1, random_state=None):
    kmedoids = KMedoids(n_clusters=clusters, random_state=random_state).fit(locations)
    return kmedoids.cluster_centers_



def get_offset(template: np.array, location_type: str):
    w,h = template.shape[::-1]

    if location_type == 'bottom_right':
        new_w, new_h = (w, h)
    elif location_type == 'bottom_left':
        new_w, new_h = (0, h)
    elif location_type == 'top_left':
        new_w, new_h = (0, 0)
    elif location_type == 'top_right':
        new_w, new_h = (w, 0)
    else:
        print("unknown location, exiting...")
        sys.exit(0)

    return new_w, new_h



def template_match(image: np.array, template: np.array, location_type: str):
    methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    locs = []
    for name in methods:

        imgmx = image.copy()
        method = eval(name)

        # Apply template Matching
        res = cv2.matchTemplate(imgmx, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # get all the matches:
        result2 = np.reshape(res, res.shape[0] * res.shape[1])
        sort = np.argsort(result2)
        k = 5

        for i in range(k):
            w,h = get_offset(template, location_type)
            tup = np.unravel_index(sort[i], res.shape)[::-1]
            loc = [tup[0]+w, tup[1]+h]
            locs.append(loc)

    return locs



def plot_and_save(image_matrix: np.array, returned_locations: list):
    axes = plt.gca()

    for tup in returned_locations:
        x,y = tup 
        axes.plot(x, y, 'ro')

    plt.set_cmap("gray")
    axes.imshow(image_matrix)

    plt.savefig('../output.png')
    plt.imsave('../array.png', image_matrix)



if __name__ == "__main__":
    if (len(sys.argv) > 1):
        clargs = sys.argv[1:]
        main(clargs)
    else:
        print("not enough arguments!")