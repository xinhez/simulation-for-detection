"""
 This script reads the rgb and segmented images dumped by carla and
 creates bounding boxes around fire hydrants using connected components.
"""

import os
import cv2
import shutil
import numpy as np

TAG = 'fire_hydrant'

target_size = 'L'
target_color = 'Red'

des_rgb_path = './data/town01_%s_%s_rgb' % (target_size, target_color)
des_seg_path = './data/town01_%s_%s_seg' % (target_size, target_color)
anno_path = './data/town01_%s_%s.txt' % (target_size, target_color)
os.makedirs(des_rgb_path, exist_ok=True)
os.makedirs(des_seg_path, exist_ok=True)

def main():
    src_rgb_path = './town01_%s_%s_rgb' % (target_size, target_color)
    src_seg_path = './town01_%s_%s_seg' % (target_size, target_color)
    verbose = 0

    with open(anno_path, 'w') as fd:
        for name in sorted(os.listdir(src_seg_path)):
            img = cv2.imread(src_seg_path + '/' + name)
            rgb_img = cv2.imread(src_rgb_path + '/' + name)
            _, timg = cv2.threshold(img[:, :, 2], 110, 220, cv2.THRESH_BINARY)
            output = cv2.connectedComponentsWithStats(timg, 4, cv2.CV_32S)

            num_labels = output[0]
            stats = output[2]

            x = None
            descrs = []
            for i in range(num_labels):
                if (
                    stats[i, cv2.CC_STAT_AREA] > 0 and 
                    stats[i, cv2.CC_STAT_WIDTH] != 1920 and stats[i, cv2.CC_STAT_HEIGHT] != 1080 and 
                    stats[i, cv2.CC_STAT_HEIGHT] > 10 # Check for chains 
                    ): 
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    descrs.append(des_rgb_path[7:] + '/' + name + ',' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(y + h) + ',' + TAG + '\n')
                    if verbose == 1:
                        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (128, 255, 0))
            if len(descrs) == 1:
                descr = descrs[0]
                fd.write(descr)
                shutil.move(src_rgb_path + '/' + name, des_rgb_path + '/' + name)
                shutil.move(src_seg_path + '/' + name, des_seg_path + '/' + name)

            if verbose == 1:
                cv2.imshow('Boundig Boxes', rgb_img)
                cv2.waitKey(0)
if __name__ == '__main__':
    main()
