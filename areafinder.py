import sys
import numpy as np 
import cv2
import matplotlib.pyplot as plt
# from sklearn_extra.cluster import KMedoids
from scipy import spatial
# np.set_printoptions(threshold=sys.maxsize)



def main(clargs: [str]):
    im_path = "./" + clargs[0]
    img_mx = cv2.imread(im_path, 0)

    allvertices, vertices_by_type = find_squares(img_mx)

    shapes = build_shapes(vertices_by_type)

    #### points must pass vertical line test if plotted sequentially (no three intersections)
    area = 0

    for i, shape in enumerate(shapes):
        area += PolyArea(shape)
        
        axes = plt.gca()

        for point in shape:
            x,y = point
            axes.plot(x, y, 'ro')

        plt.set_cmap("gray")

        plt.savefig(f'../output_{i}.png')


    #TODO transform area from pixels to actual footage, use the key in the plans
    plot_and_save(img_mx, allvertices)

    return area



def build_shapes(vertices_by_type: []):

    shapes = []
    trees = []

    shape_type = ('square', 4) if len(vertices_by_type) == 4 else ('triangle', 3)

    for vertex_num in range(shape_type[1]):
        trees.append(spatial.KDTree(vertices_by_type[vertex_num]))

    if shape_type[0] == 'square':
        for vertex in vertices_by_type[0]:
            corner2 = vertices_by_type[1][trees[1].query(vertex)[1]]
            corner3 = vertices_by_type[2][trees[2].query(corner2)[1]]
            corner4 = vertices_by_type[3][trees[3].query(corner3)[1]]
            shapes.append([vertex, corner2, corner3, corner4])
    elif shape_type[0] == 'triangle':
        for vertex in vertices_by_type[0]:
            corner2 = vertices_by_type[1][trees[1].query(vertex)[1]]
            corner3 = vertices_by_type[2][trees[2].query(corner2)[1]]
            shapes.append([vertex, corner2, corner3])

    return shapes



def find_squares(image: np.array):
    vertices = []
    tl_vertices = []
    bl_vertices = []
    br_vertices = []
    tr_vertices = []

    top_left_corner = cv2.imread('./shapes/top_left.png', 0)
    bottom_left_corner = cv2.imread('./shapes/bottom_left.png', 0)
    bottom_right_corner = cv2.imread('./shapes/bottom_right.png', 0)
    top_right_corner = cv2.imread('./shapes/top_right.png', 0)

    for tl_vertex in find_location(image, top_left_corner, 'sq_top_left'):
        tl_vertices.append(tl_vertex)
        vertices.append(tl_vertex)
    for bl_vertex in find_location(image, bottom_left_corner, 'sq_bot_left'):
        bl_vertices.append(bl_vertex)
        vertices.append(bl_vertex)
    for br_vertex in find_location(image, bottom_right_corner, 'sq_bot_right'):
        br_vertices.append(br_vertex)
        vertices.append(br_vertex)
    for tr_vertex in find_location(image, top_right_corner, 'sq_top_right'):
        tr_vertices.append(tr_vertex)
        vertices.append(tr_vertex)

    return vertices, [tl_vertices, bl_vertices, br_vertices, tr_vertices]


def find_triangles(image: np.array):
    vertices = []
    t_vertices = []
    l_vertices = []
    r_vertices = []

    top_corner = cv2.imread('./shapes/triangle_top.png', 0)
    left_corner = cv2.imread('./shapes/triangle_left.png', 0)
    right_corner = cv2.imread('./shapes/triangle_right.png', 0)

    for t_vertex in find_location(image, top_corner, 'tri_top'):
        t_vertices.append(t_vertex)
        vertices.append(t_vertex)
    for l_vertex in find_location(image, left_corner, 'tri_left'):
        l_vertices.append(l_vertex)
        vertices.append(l_vertex)
    for r_vertex in find_location(image, right_corner, 'tri_right'):
        r_vertices.append(r_vertex)
        vertices.append(r_vertex)

    return vertices, [t_vertices, l_vertices, r_vertices]


def find_location(image: np.array, template: np.array, location_type: str):
    locs = template_match(image, template, location_type)
    # medoid = find_kmedoids(locs)[0]

    return locs


# def find_kmedoids(locations, clusters=1, random_state=None):
#     kmedoids = KMedoids(n_clusters=clusters, random_state=random_state).fit(locations)
#     return kmedoids.cluster_centers_



def template_match(image: np.array, template: np.array, location_type: str):
    # methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF_NORMED']
    locs = []

    for name in methods:

        imgmx = image.copy()
        method = eval(name)

        # Apply template Matching
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            res = cv2.matchTemplate(imgmx, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            result2 = np.reshape(res, res.shape[0] * res.shape[1])
            sort = np.argsort(result2)
            k = 10
            w,h = get_offset(template, location_type)

            for i in range(k):
                tup = np.unravel_index(sort[i], res.shape)[::-1]
                loc = [tup[0]+w, tup[1]+h]
                locs.append(loc)

        elif method in [cv2.TM_CCOEFF_NORMED]:

            mask = np.zeros(imgmx.shape, np.uint8)
            corner_count = 0

            res = cv2.matchTemplate(imgmx, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = max_val * .95
            results = np.where( res >= threshold)
            w,h = get_offset(template, location_type)

            height, width = template.shape[:2]

            for pt in zip(*results[::-1]):

                if mask[pt[1] + int(np.round(height/2)), pt[0] + int(np.round(width/2))] != 255:
                    mask[pt[1]:pt[1]+height, pt[0]:pt[0]+width] = 255
                    corner_count += 1
                    npt = [pt[0]+w, pt[1]+h]
                    locs.append(npt)

    return locs



def get_offset(template: np.array, location_type: str):
    w,h = template.shape[::-1]

    if location_type == 'sq_bot_right':
        new_w, new_h = (w, h)
    elif location_type == 'sq_bot_left':
        new_w, new_h = (0, h)
    elif location_type == 'sq_top_left':
        new_w, new_h = (0, 0)
    elif location_type == 'sq_top_right':
        new_w, new_h = (w, 0)
    elif location_type == 'tri_top':
        new_w, new_h = (w/2, 0)
    elif location_type == 'tri_left':
        new_w, new_h = (0, h)
    elif location_type == 'tri_right':
        new_w, new_h = (w, h)
    else:
        print("unknown location, exiting...")
        sys.exit(0)

    return new_w, new_h




def plot_and_save(image_matrix: np.array, returned_locations: list):
    axes = plt.gca()

    for point in returned_locations:
        x,y = point
        axes.plot(x, y, 'ro')

    plt.set_cmap("gray")
    axes.imshow(image_matrix)

    plt.savefig('../output.png')
    plt.imsave('../array.png', image_matrix)



def PolyArea(corners):
    uz = list(zip(*corners))
    xs = list(uz[0])
    ys = list(uz[1])
    return 0.5*np.abs(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1)))



if __name__ == "__main__":
    if (len(sys.argv) > 1):
        clargs = sys.argv[1:]
        area = main(clargs)
        print(f"area of plans are {area} square ft")
    else:
        print("not enough arguments!")
