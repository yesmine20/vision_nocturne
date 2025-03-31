import cv2
import matplotlib.pyplot as plt
 

# Chargement de l'image
image = cv2.imread('test\iii1.jpg', cv2.IMREAD_COLOR)


# Affichage de l'image originale
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.show()
# Égalisation de l'histogramme
image_eq = cv2.equalizeHist(image)


# Affichage de l'image après égalisation
plt.imshow(image_eq, cmap='gray')
plt.title("Image après égalisation")
plt.show()
# Seuillage d'Otsu
_, image_otsu = cv2.threshold(image_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Affichage de l'image après seuillage d'Otsu
plt.imshow(image_otsu, cmap='gray')
plt.title("Image après seuillage d'Otsu")
plt.show()
# Amélioration de la luminosité
image_bright = cv2.convertScaleAbs(image_eq, alpha=1, beta=50)


# Affichage de l'image avec plus de luminosité
plt.imshow(image_bright, cmap='gray')
plt.title("Image avec plus de luminosité")
plt.show()
# Sauvegarde de l'image améliorée
cv2.imwrite('image_amelioree.jpg', image_bright)
