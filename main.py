import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"L'image {image_path} est introuvable.")
    return image

def adjust_gamma(image, gamma=0.92):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adaptive_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    target_brightness = 70
    current_brightness = np.mean(v)
    gain = target_brightness / (current_brightness + 1e-7)

    v = cv2.multiply(v, min(gain, 3.0))
    s = cv2.multiply(s, 0.95)  # Réduction légère de saturation
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 15)

def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# === AJOUT : Égalisation globale de l'histogramme ===
def equalize_global_histogram(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# === AJOUT : Seuillage d'Otsu (pour affichage/visualisation uniquement) ===
def otsu_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def show_image(image, title, cmap=None):
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap or 'gray')
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Pipeline complet
image = load_image('test/i3.jpg')
show_image(image, "Image originale")

# 1. Contraste
image_contrast = enhance_contrast(image)

# 2. Gamma
image_gamma = adjust_gamma(image_contrast, gamma=0.92)

# 3. Luminosité adaptative
image_brightness = adaptive_brightness(image_gamma)

# 4. Débruitage
image_denoised = denoise(image_brightness)

# 5. Sharpen final
image_final = sharpen(image_denoised)

# === Égalisation globale d'histogramme (optionnelle) ===
image_hist_eq = equalize_global_histogram(image_final)
show_image(image_hist_eq, "Après égalisation globale de l'histogramme")

# === Seuillage d’Otsu (visualisation) ===
image_otsu = otsu_threshold(image_final)
show_image(image_otsu, "Seuillage d'Otsu", cmap='gray')

# Affichage final
show_image(image_final, "Image finale améliorée")
cv2.imwrite("sortie/image_amelioree.jpg", image_final)
cv2.imwrite("sortie/image_hist_eq.jpg", image_hist_eq)
cv2.imwrite("sortie/image_otsu.jpg", image_otsu)
