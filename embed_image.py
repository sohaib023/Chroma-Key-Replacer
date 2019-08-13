import cv2
import numpy as np
import math
import time
import platform
import argparse

debug = False
kernel = np.ones((3, 3), np.uint8)

def extract_green_areas(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    step = 20
    for i in range(70, 256, step):
        lower_blue = np.array([0, i, 0])
        j = i + step
        upper_blue = np.array([5*j//6, j, 5*j//6])
        mask = mask + cv2.inRange(image_copy, lower_blue, upper_blue)

    mask = cv2.erode(mask, kernel, iterations=9)
    mask = cv2.dilate(mask, kernel, iterations=9)
    return mask

def extract_largest_cc(mask):
    num_labels, mask, stats, centroids = cv2.connectedComponentsWithStats(mask)
    bins = np.bincount(mask.flat)
    if len(bins) > 1:
        flag = True
        largestCC = mask == np.argmax(np.bincount(mask.flat)[1:])+1
    else:
        flag = False
        largestCC = mask == 3
    mask[largestCC] = 255
    mask[~largestCC] = 0
    mask = np.uint8(mask)
    return flag, mask

def extract_vertices(mask_largestCC):
    major = cv2.__version__.split('.')[0]
    if int(major) >= 3:
        ret, contours, hierarchy = cv2.findContours(mask_largestCC, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(mask_largestCC, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.array([[0,0],[0,0],[0,0],[0,0]])
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # define main island contour approx. and hull
    epsilon = 1
    dst_vertices = None
    i = 0
    while dst_vertices is None or len(dst_vertices) != 4:
        # print(epsilon)
        dst_vertices = cv2.approxPolyDP(cnt, epsilon, True)
        if len(dst_vertices) > 4:
            epsilon *= 2
        else:
            epsilon -= 0.2
        if i > 1000:
            break
        i += 1

    dst_vertices = np.array([x[0] for x in dst_vertices]).tolist()
    if len(dst_vertices) != 4:
        return np.array([[0,0],[0,0],[0,0],[0,0]])
    dst_vertices = sorted(dst_vertices, key=lambda x: x[0])

    if dst_vertices[0][1] > dst_vertices[1][1]:
        dst_vertices[0], dst_vertices[1] = dst_vertices[1], dst_vertices[0]
    if dst_vertices[2][1] < dst_vertices[3][1]:
        dst_vertices[2], dst_vertices[3] = dst_vertices[3], dst_vertices[2]

    return dst_vertices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_video_path")
    parser.add_argument("image_path")
    parser.add_argument("output_video_path")

    args = parser.parse_args()
    print(args)

    img_embedding = cv2.imread(args.image_path)
    src_vertices = [[0, 0], [0, img_embedding.shape[0]], [img_embedding.shape[1], img_embedding.shape[0]], [img_embedding.shape[1], 0]]
    img_embedding = cv2.blur(img_embedding, (21, 21))
    dst_vertices_in_use = [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]
    
    cap = cv2.VideoCapture(args.input_video_path)
    
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if i == 0:
            if platform.system == 'Window':
                out = cv2.VideoWriter(args.output_video_path + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame.shape[1], frame.shape[0]))
            else:
                out = cv2.VideoWriter(args.output_video_path + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
        if cv2.waitKey(1) & 0xFF == ord('q') or frame is None:
            break
        # if i < 1000:
        #     i+=1
        #     print(i)
        #     continue
        # if i > 1000:
        #     break
        if i % 100 == 0:
            print(i, " frames processed.")
        i += 1  

        image = frame # cv2.imread('4.jpg')
    
        image_copy = np.copy(image)
        # cv2.imshow('original', image_copy)
    
        start = time.time()
    
        green_areas = extract_green_areas(image)
    
        if debug:
            print("1 Time taken = ", time.time() - start)
    
        flag, mask_largestCC = extract_largest_cc(green_areas)
    
        if flag:
            mask_smoothed = cv2.blur(cv2.cvtColor(cv2.dilate(mask_largestCC, kernel, iterations=3), cv2.COLOR_GRAY2BGR), (9, 9))
        
            masked_image = np.uint8(np.clip(np.int16(image) - np.int16(mask_smoothed), 0, 255))
        
            if debug:
                print("3 Time taken = ", time.time() - start)
        
            dst_vertices = extract_vertices(mask_largestCC.copy())
        
            if debug:
                print("4 Time taken = ", time.time() - start)    
        
            if debug:
                print(dst_vertices)
        
            if sum([((dst_vertices[i][0] - dst_vertices_in_use[i][0])**2 + (dst_vertices[i][1] - dst_vertices_in_use[i][1])**2)/max(image.shape) for i in range(4)]) > 0.05:
                dst_vertices_in_use = dst_vertices
            else:
                dst_vertices_in_use = 0.1 * np.array(dst_vertices) + 0.9 * np.array(dst_vertices_in_use)
        
            if debug:
                print(src_vertices)    
        
            if debug:
                print("5.1 Time taken = ", time.time() - start)    
        
            transform = cv2.getPerspectiveTransform(np.array(src_vertices, dtype=np.float32), np.array(dst_vertices_in_use, dtype=np.float32))
            img_transformed = cv2.warpPerspective(
                img_embedding,
                transform,
                (masked_image.shape[1], masked_image.shape[0]),
                flags=cv2.INTER_CUBIC)
        
            if debug:
                print("6 Time taken = ", time.time() - start)    

            mask_smoothed = cv2.blur(cv2.cvtColor(mask_largestCC, cv2.COLOR_GRAY2BGR), (5, 5))
            img_transformed = np.uint8(np.clip(np.int16(img_transformed) - np.int16(255 - mask_smoothed), 0, 255))
        
            final_image = masked_image + img_transformed
                 
            if debug:
                print("Time taken = ", time.time() - start)    
        else:
            final_image = image
    
        cv2.imshow('final', final_image)
        out.write(final_image)
    
    cap.release()
    cv2.destroyAllWindows()    
