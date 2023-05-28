#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:50:02 2023

@author: dhvilleg
"""

import cv2
import numpy as np

#img = cv2.imread('/home/dhvilleg/Documents/proyectos/DeepVision/2.1 datasetLunares/datasetLunares/dysplasticNevi/train/dysplasticNevi4.jpg')


def getFeatures(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    threshold,_ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    mask = np.uint8(1*(gray < threshold) )
    
    b = (1/255) * np.sum(img[:,:,0] * mask) / np.sum(mask)
    g = (1/255) * np.sum(img[:,:,1] * mask) / np.sum(mask)
    r = (1/255) * np.sum(img[:,:,2] * mask) / np.sum(mask)
    
    return [b, g, r]


"""
Generacion del dataset de caracteristicas
"""
import glob

paths = ['/home/dhvilleg/Documents/proyectos/DeepVision/2.1 datasetLunares/datasetLunares/dysplasticNevi/train/',
         '/home/dhvilleg/Documents/proyectos/DeepVision/2.1 datasetLunares/datasetLunares/spitzNevus/train/']

labels = []
features = []


for label, path in enumerate(paths):
    for filename in glob.glob(path+"*.jpg"):
        img = cv2.imread(filename)
        features.append(getFeatures(img))
        labels.append(label)
        
        
features = np.array(features)
labels = np.array(labels) 
labels = 2 * labels -1


#visualizacion del dataset en el espacio de caracteristicas

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, feature_row in enumerate(features):
    if labels[[i]] == -1:
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker='*', c='k')
        
    else:
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker='*', c='r')

ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')

#error en función de las constantes del hiperplano "W"

subFeatures = features[:,1::]

loss = []

for w1 in np.linspace(-6, 6, 100):
    for w2 in np.linspace(-6, 6, 100):
        totaError=0
        for i,feature_row in enumerate(subFeatures):
            sampleError = (w1*feature_row[0]+w2*feature_row[1] - labels[i])**2
            totaError+=sampleError
        loss.append([w1,w2,totaError])


loss = np.array(loss)

from matplotlib import cm
fig=plt.figure()
ax1=fig.add_subplot(111, projection='3d')


ax1.plot_trisurf(loss[:,0], loss[:,1], loss[:,2], cmap=cm.jet, linewidth=0)
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
ax1.set_zlabel("loss")



#calculo del hiperplano que separa las dos clases de forma optima
A=np.zeros((4,4))
b=np.zeros((4,1))

for i, feature_row in enumerate(features):
    x=np.append([1],feature_row)
    x= x.reshape((4,1))
    y=labels[i]
    A=A+x*x.T
    b=b+x*y

invA=np.linalg.inv(A)

W=np.dot(invA,b)

X=np.arange(0,1,0.1)
Y=np.arange(0,1,0.1)
X,Y=np.meshgrid(X,Y)

##W[3]*Z+W[1]*X+W[2]*Y+W[0]=0

Z=-(W[1]*X+W[2]*Y+W[0])/W[3]

ax.plot_surface(X,Y,Z, cmap=cm.Blues)


#error de entrenamiento

prediction = 1*(W[0] + np.dot(features,W[1::]))>=0

prediction = 2*prediction-1

error=np.sum(prediction != labels.reshape(-1,1))/len(labels)

efectividad = 1-error


#prediccion para una imagen

#path_img='/home/dhvilleg/Documents/proyectos/DeepVision/2.1 datasetLunares/datasetLunares/spitzNevus/train/spitzNevus8.jpg'
path_img='/home/dhvilleg/Documents/proyectos/DeepVision/2.1 datasetLunares/datasetLunares/dysplasticNevi/train/dysplasticNevi3.jpg'

img=cv2.imread(path_img)

feature_vector=np.array(getFeatures(img))

result=np.sign(W[0]+np.dot(feature_vector,W[1::]))

if result == -1:
    print("es un dysplasticNevi")
else:
    print("es un spitzNevus")









    






















