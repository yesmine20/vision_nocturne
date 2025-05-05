import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_image_path =  "test/i7.png"
output_directory = 'sortie'
output_image_path = os.path.join(output_directory, 'image_amelioree.jpg')

# Chargement de l'image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"L'image {image_path} n'a pas été trouvée.")
    return image

# Amélioration du contraste avec CLAHE (meilleur que equalizeHist pour les images sombres)
def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

# Ajustement modéré de la luminosité (sans saturation)
def enhance_brightness(image, beta=20):  # Beta réduit
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

# Affichage rapide
def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Pipeline
image = load_image(input_image_path)
show_image(image, "Image originale")

image_clahe = apply_clahe(image)
show_image(image_clahe, "Après CLAHE")

image_bright = enhance_brightness(image_clahe)
show_image(image_bright, "Après ajustement de la luminosité")

cv2.imwrite(output_image_path, image_bright)
print(f"L'image améliorée a été sauvegardée sous {output_image_path}")
