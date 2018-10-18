#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:21 2018

@author: wuyz
"""

import my_bilateral_filter as bf
import numpy as np
import cv2 as cv

face = cv.imread("input_face.jpg")
sigma_r = 30
sigma_s = 12
kernel_size_bf = 5
kernel_size_median = 3
kernel_size_gs = 5
output_face = cv.medianBlur(face, kernel_size_median)
output_face_bf = bf.bilateral_filter(output_face, kernel_size_bf, sigma_r, sigma_s)
cv.imwrite("bf_filtered_face.jpg", output_face_bf)

output_face_gs = cv.GaussianBlur(output_face, (kernel_size_gs, kernel_size_gs), sigma_s)
cv.imwrite("gs_filtered_face.jpg", output_face_gs)