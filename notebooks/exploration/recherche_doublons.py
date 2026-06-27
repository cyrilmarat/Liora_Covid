import numpy as np
import cv2 #import OpenCV
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
import fnmatch
import os

workingpath="/home/cyril/liora/projet"
normal_files="/COVID-19_Radiography_Dataset/Normal"
covid_files="/COVID-19_Radiography_Dataset/COVID"
lungOpacity_files="/COVID-19_Radiography_Dataset/Lung_Opacity"
viralPneumonia_files="/COVID-19_Radiography_Dataset/Viral Pneumonia"
size_img=299
size_small_img=50

#read normal images
nb_files_Normal=len(fnmatch.filter(os.listdir(workingpath+normal_files+"/images/"), "*.png"))
all_small_Normal = np.zeros((size_small_img, size_small_img, nb_files_Normal),  dtype=np.uint8)

for i in range(0, nb_files_Normal):
    img_Normal = cv2.imread(workingpath+normal_files +"/images/Normal-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)
    all_small_Normal[:,:,i]= cv2.resize(img_Normal, (size_small_img, size_small_img), 0, 0, cv2.INTER_NEAREST)


#read covid images
nb_files_Covid=len(fnmatch.filter(os.listdir(workingpath+covid_files+"/images/"), "*.png"))
all_small_Covid = np.zeros((size_small_img, size_small_img, nb_files_Covid),  dtype=np.uint8)

for i in range(0, nb_files_Covid):
    img_Covid = cv2.imread(workingpath+covid_files +"/images/COVID-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)
    all_small_Covid[:,:,i]= cv2.resize(img_Covid, (size_small_img, size_small_img), 0, 0, cv2.INTER_NEAREST)

# read lung opacity images
nb_files_LungOpacity=len(fnmatch.filter(os.listdir(workingpath+lungOpacity_files+"/images/"), "*.png"))
all_small_LungOpacity = np.zeros((size_small_img, size_small_img, nb_files_LungOpacity),  dtype=np.uint8)

for i in range(0, nb_files_LungOpacity):
    img_LungOpacity = cv2.imread(workingpath+lungOpacity_files +"/images/Lung_Opacity-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)
    all_small_LungOpacity[:,:,i]= cv2.resize(img_LungOpacity, (size_small_img, size_small_img), 0, 0, cv2.INTER_NEAREST)

# read viral pneumonia images
nb_files_ViralPneumonia=len(fnmatch.filter(os.listdir(workingpath+viralPneumonia_files+"/images/"), "*.png"))
all_small_ViralPneumonia = np.zeros((size_small_img, size_small_img, nb_files_ViralPneumonia),  dtype=np.uint8)
for i in range(0, nb_files_ViralPneumonia):
    img_ViralPneumonia = cv2.imread(workingpath+viralPneumonia_files +"/images/Viral Pneumonia-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)
    all_small_ViralPneumonia[:,:,i]= cv2.resize(img_ViralPneumonia, (size_small_img, size_small_img), 0, 0, cv2.INTER_NEAREST)


def mse(imageA, imageB):

    assert imageA.shape == imageB.shape, "Images must be the same size."

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

simular_files_normal=[]

for i in range(0, nb_files_Normal):
    for j in range(i, nb_files_Normal):
        if i != j:
            diff= mse(all_small_Normal[:,:,i], all_small_Normal[:,:,j])
            if (diff <10):
                simular_files_normal.append((i,j))

print(simular_files_normal)