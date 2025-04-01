import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Chemin de l'image d'entrée et de sortie
input_image_path = r'test\iii1.jpg'
output_directory = r'sortie'
output_image_path = os.path.join(output_directory, 'image_amelioree.jpg')

# Chargement de l'image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"L'image {image_path} n'a pas été trouvée.")
    return image

# Égalisation de l'histogramme
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Seuillage d'Otsu
def otsu_threshold(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

# Amélioration de la luminosité
def enhance_brightness(image, beta=50):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

# Affichage des images
def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Pipeline de traitement
image = load_image(input_image_path)
show_image(image, "Image originale")

image_eq = histogram_equalization(image)
show_image(image_eq, "Image après égalisation de l'histogramme")

image_otsu = otsu_threshold(image_eq)
show_image(image_otsu, "Image après seuillage d'Otsu")

image_bright = enhance_brightness(image_eq)
show_image(image_bright, "Image avec amélioration de la luminosité")

# Sauvegarde de l'image améliorée
cv2.imwrite(output_image_path, image_bright)
print(f"L'image améliorée a été sauvegardée sous {output_image_path}")
