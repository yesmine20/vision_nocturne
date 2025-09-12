import cv2
import numpy as np

# --- Fonctions de traitement d'image ---
def corriger_gamma(image, gamma=1.2):  # Éclaircissement plus fort
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def renforcer_nettete(image):
    noyau = np.array([[0, -1, 0],
                      [-1, 5, -1],  # Filtre ajusté pour plus de précision
                      [0, -1, 0]])
    return cv2.filter2D(image, -1, noyau)

def ameliorer_image(image):
    # Étape 1 : Amélioration du contraste local avec CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Augmentation du clipLimit
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Étape 2 : Correction gamma plus agressive
    image_gamma = corriger_gamma(image_clahe, gamma=1.2)

    # Étape 3 : Augmenter luminosité et contraste
    alpha = 1.1  # Contraste plus fort
    beta = 15    # Luminosité accrue
    image_amelioree = cv2.convertScaleAbs(image_gamma, alpha=alpha, beta=beta)

    return image_amelioree

# --- Détection de mouvement ---
def detecter_mouvement(frame_courante, frame_precedente):
    gris = cv2.cvtColor(frame_courante, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (7, 7), 2)  # Flou plus prononcé pour réduire le bruit

    if frame_precedente is None:
        return None, gris

    diff = cv2.absdiff(frame_precedente, gris)
    _, seuil = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)  # Seuil ajusté
    seuil = cv2.dilate(seuil, None, iterations=3)  # Dilatation renforcée
    contours, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_resultat = frame_courante.copy()
    boites = []
    for c in contours:
        if cv2.contourArea(c) < 800:  # Ignorer les petits mouvements parasites
            x, y, w, h = cv2.boundingRect(c)
            boites.append((x, y, x + w, y + h))

    boites_fusionnees = fusionner_boites(boites, seuil=50)  # Fusion élargie des zones de mouvement
    for (x1, y1, x2, y2) in boites_fusionnees:
        region = gris[y1:y2, x1:x2]
        if np.mean(region) > 40:
            cv2.rectangle(frame_resultat, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resultat, "Mvt", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame_resultat, gris

def fusionner_boites(boites, seuil=50):
    if not boites:
        return []
    boites = sorted(boites)
    groupes = []
    for boite in boites:
        fusionne = False
        for i in range(len(groupes)):
            bx1, by1, bx2, by2 = groupes[i]
            x1, y1, x2, y2 = boite
            if not (x2 + seuil < bx1 or x1 - seuil > bx2 or y2 + seuil < by1 or y1 - seuil > by2):
                nx1 = min(bx1, x1)
                ny1 = min(by1, y1)
                nx2 = max(bx2, x2)
                ny2 = max(by2, y2)
                groupes[i] = (nx1, ny1, nx2, ny2)
                fusionne = True
                break
        if not fusionne:
            groupes.append(boite)
    return groupes

# --- Traitement Vidéo ---
cap = cv2.VideoCapture("v22.avi")
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la vidéo.")
    exit()

prev_sombre = None
prev_amelioree = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # Étape 1 : Image d'origine
    sombre = frame.copy()

    # Étape 2 : Amélioration
    amelioree = ameliorer_image(sombre)

    # Étape 3 : Détection de mouvement
    sombre_detectee, prev_sombre = detecter_mouvement(sombre, prev_sombre)
    amelioree_detectee, prev_amelioree = detecter_mouvement(amelioree, prev_amelioree)

    # Affichage comparatif
    if sombre_detectee is not None and amelioree_detectee is not None:
        cv2.putText(sombre_detectee, "Avant amelioration", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(amelioree_detectee, "Apres amelioration", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        comparaison = np.hstack((sombre_detectee, amelioree_detectee))
        cv2.imshow("Comparaison Vision Nocturne", comparaison)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()