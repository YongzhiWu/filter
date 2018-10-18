#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:23:37 2018

@author: wuyz
"""

import cv2 as cv
import numpy as np

def gaussian(x, sigma):
    return (1.0 / sigma) * np.exp(-(x ** 2).sum() / (2 * (sigma ** 2)))
    #return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))

'''
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
'''
def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)
    padding_image = image.astype(np.int)
    radius = int(diameter / 2)
    if image.ndim == 2:
        channels = 1
        padding_image = np.pad(padding_image, ((radius, radius), (radius, radius)), 'constant', constant_values=(0, 0))
    if image.ndim == 3:
        channels = image.shape[2]
        padding_image = np.pad(padding_image, ((radius, radius), (radius, radius), (0, 0)), 'constant', constant_values=(0, 0,))
    for row in range(new_image.shape[0]):
        for col in range(new_image.shape[1]):
            wp_total = 0
            filtered_image = np.zeros(channels)
            for k in range(diameter):
                for l in range(diameter):
                    n_x = row + k
                    n_y = col + l
                    gi = gaussian((padding_image[n_x][n_y]) - (padding_image[row + radius][col + radius]), sigma_i)
                    gs = gaussian(np.array([n_x - row - radius, n_y - col - radius]), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) +  ((padding_image[n_x][n_y]) * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = (np.round(filtered_image))
    return new_image.astype(np.uint8)

if __name__ == '__main__':
    image = cv.imread("input_image.png", 0)
    
    filtered_image_opencv = cv.bilateralFilter(image, 7, 30, 11)
    cv.imwrite("filtered_image_opencv.png", filtered_image_opencv)
    filtered_image_my_filter = bilateral_filter(image, 7, 30, 11)
    cv.imwrite("filtered_image_my_filter.png", filtered_image_my_filter)
    