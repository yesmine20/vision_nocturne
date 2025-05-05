import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Fonctions outils ===

def apply_clahe(img, clip_limit=4.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def estimate_gamma(img_gray):
    # Moyenne de l'intensité → gamma inversement proportionnel
    mean_intensity = np.mean(img_gray)
    gamma = 1.0 if mean_intensity == 0 else np.clip(128 / mean_intensity, 0.5, 2.5)
    return gamma

def adjust_gamma(img, gamma):
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, look_up)

def normalize_brightness(img):
    # Étalonne l’histogramme par canal
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# === Pipeline principal ===

# Charger l'image
image = cv2.imread("test/i7.png")
if image is None:
    print("Erreur : Image non chargée. Vérifiez le chemin.")
else:
    # 1. Amélioration locale du contraste (CLAHE)
    clahe_img = apply_clahe(image)

    # 2. Dénuage avec filtre bilatéral (préserve les bords)
    denoised = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Gamma dynamique
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    gamma = estimate_gamma(gray)
    gamma_corrected = adjust_gamma(denoised, gamma)

    # 4. Normalisation de la luminosité
    normalized = normalize_brightness(gamma_corrected)

    # 5. Renforcement des détails
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 3)
    sharpened = cv2.addWeighted(normalized, 1.2, gaussian, -0.2, 0)

    # === Affichage et sauvegarde ===
       # === Affichage côte à côte : avant et après amélioration ===
    plt.figure(figsize=(12, 6))

    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image originale")
    plt.axis('off')

    # Image améliorée
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title("Image améliorée")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
