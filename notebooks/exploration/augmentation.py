import numpy as np
import cv2
import matplotlib.pyplot as plt
import fnmatch
import os
import seaborn as sns
import albumentations as A

workingpath="/Users/antoine/Desktop/Liora/Projet"
normal_files="/COVID-19_Radiography_Dataset/Normal"
covid_files="/COVID-19_Radiography_Dataset/COVID"
lungOpacity_files="/COVID-19_Radiography_Dataset/Lung_Opacity"
viralPneumonia_files="/COVID-19_Radiography_Dataset/Viral Pneumonia"

classes = ["Normal", "Covid", "Lung_Opacity", "Viral_Pneumonia"]
#classes = ["Covid"]
class_counts = {"Normal": 10192, "Covid": 3616, "Lung_Opacity": 6012, "Viral_Pneumonia": 1345}
#class_counts = {"Covid": 3616}

size_img=299

nb_files_Covid=len(fnmatch.filter(os.listdir(workingpath+covid_files+"/images/"), "*.png"))
all_img_Covid = np.zeros((size_img, size_img, nb_files_Covid),  dtype=np.uint8)
for i in range(0, nb_files_Covid):
    img_Covid = cv2.imread(workingpath+covid_files +"/images/COVID-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE)  

    img_Covid_mask = cv2.imread(workingpath+covid_files +"/masks/COVID-"+str(i+1)+".png", cv2.IMREAD_GRAYSCALE) # pour lire

    img_mask_resized = cv2.resize(img_Covid_mask, (img_Covid.shape[0], img_Covid.shape[1]), 0, 0, cv2.INTER_NEAREST)
    img_Covid_and_mask = cv2.bitwise_and(img_Covid,img_mask_resized)
    img_Covid_and_mask_resized = cv2.resize(img_Covid_and_mask, (size_img, size_img), 0, 0, cv2.INTER_NEAREST)
    all_img_Covid[:,:,i] = img_Covid_and_mask_resized


print("all_image_Covid", all_img_Covid)
print(type(all_img_Covid))



#Définition des modifications
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
])


#Application des modifications à une classe d'images
def augment_images(images, target_count):
    liste_images=[]
    augmented_images = images.copy()
    for k in np.arange(0, augmented_images.shape[2]):
        liste_images.append(augmented_images[:,:,k])
    print("longueur liste",len(liste_images))
    while len(liste_images) < target_count:
        print("augmented_images shape",augmented_images.shape[2])
        # Sélectionner une image aléatoire
        idx = np.random.randint(0, len(images))
        img = images[:,:,idx]
        # Appliquer l'augmentation
        augmented = augmentation(image=img)
        print("augmented", augmented)
        augmented_img = augmented["image"]
        print(augmented_img)
        print(np.min(img))
        print(np.max(img))
        liste_images.append(augmented_img)
        print(augmented_images.shape)
        augmented_images=np.uint8(liste_images)
    return augmented_images



#Rééquilibrage des classes
def balance_classes():
    # Charger toutes les images par classe
    class_images = {}
    for class_name in classes:
        class_images[class_name] = all_img_Covid
        print("dimension de la classe",class_images[class_name].shape[1])

    # Déterminer le nombre cible (classe majoritaire)
    target_count = max(class_counts.values())

    # Appliquer l'augmentation pour chaque classe minoritaire
    balanced_images = {}
    for class_name in classes:
        current_count = len(class_images[class_name])
        print(current_count)
        if current_count < target_count:
            print(f"Augmentation de la classe {class_name} : {current_count} -> {target_count}")
            balanced_images[class_name] = augment_images(class_images[class_name], target_count)
        else:
            balanced_images[class_name] = class_images[class_name]

    return balanced_images

#class_images[class_name]
#Exécution du rééquilibrage
balanced_images = balance_classes()

# Vérification des nouveaux nombres d'images
for class_name, images in balanced_images.items():
    print(f"{class_name} : {len(images)} images")

