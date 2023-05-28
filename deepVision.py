# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

bananas = cv2.imread('/home/dhvilleg/Documents/proyectos/DeepVision/bananos.jpg')


print(bananas[169,207])

r = bananas[:,:,2]
g = bananas[:,:,1]
b = bananas[:,:,0]

r_ = bananas[326,274,2]
g_ = bananas[326,274,1]
b_ = bananas[326,274,0]

# cv2.imshow("bababababanas",r)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_gray = cv2.cvtColor(bananas, cv2.COLOR_BGR2GRAY)

# cv2.imshow("gray",img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

binaria = np.uint8(255*(img_gray < 220))

# cv2.imshow("gray",binaria)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray_segmentada = np.uint8(img_gray * (binaria/255))

# cv2.imshow("gray",gray_segmentada)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

seg_color = bananas.copy()

seg_color[:,:,0] = np.uint8(b * (binaria/255))
seg_color[:,:,1] = np.uint8(g * (binaria/255))
seg_color[:,:,2] = np.uint8(r * (binaria/255))

# cv2.imshow("color_segmentada", seg_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# plt.hist(img_gray.flatten(),bins=3)
# plt.show()

th_otsu,_ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# print("th_otsu: {}, THRESH_BINARY: {}, THRESH_OTSU: {}".format(th_otsu, cv2.THRESH_BINARY, cv2.THRESH_OTSU))

binaria_otsu = np.uint8(255*(img_gray > th_otsu))

# cv2.imshow("color_segmentada", binaria_otsu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#para lunar
lunar = cv2.imread('/home/dhvilleg/Downloads/Telegram Desktop/Deep Learning Avanzado para Computer Vision con TensorFlow/2. Introducción al Procesamiento de Imágenes/13.1 Archive/000078.jpg')
img_gray_lunar = cv2.cvtColor(lunar, cv2.COLOR_BGR2GRAY)

th_otsu_lunar,_ = cv2.threshold(img_gray_lunar, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imshow("color_segmentada", th_otsu_lunar)
cv2.waitKey(0)
cv2.destroyAllWindows()



