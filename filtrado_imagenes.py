#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:10:17 2023

@author: dhvilleg
"""

import cv2
import numpy as np

bananos = cv2.imread('/home/dhvilleg/Documents/proyectos/DeepVision/bananos.jpg')

cv2.imshow("fOriginal",bananos)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_3x3 = np.ones((3,3))/(3*3)
output_3x3 = cv2.filter2D(bananos, -1, kernel_3x3)
cv2.imshow("filtrp 3x3",output_3x3)
cv2.waitKey(0)
cv2.destroyAllWindows()


kernel_11x11 = np.ones((11,11))/(11*11)
output_11x11 = cv2.filter2D(bananos, -1, kernel_11x11)
cv2.imshow("filtrp 11x11",output_11x11)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_37x37 = np.ones((37,37))/(37*37)
output_37x37 = cv2.filter2D(bananos, -1, kernel_37x37)
cv2.imshow("filtrp 37x37",output_37x37)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_97x97 = np.ones((97,97))/(97*97)
output_97x97 = cv2.filter2D(bananos, -1, kernel_97x97)
cv2.imshow("filtrp 97x97",output_97x97)
cv2.waitKey(0)
cv2.destroyAllWindows()