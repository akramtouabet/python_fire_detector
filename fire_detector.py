"""
Detecteur d'incendie en temps reel - version amelioree
========================================================

Ameliorations par rapport a une simple detection par couleur :

1. COULEUR + MOUVEMENT : une zone n'est consideree comme du feu que si elle
   a l'apparence d'une flamme vue par une webcam (coeur BLANC surexpose
   entoure d'un halo orange/jaune plus sature, voir detecter_couleur_feu)
   ET si elle bouge (le feu scintille en permanence, contrairement a un mur
   ou une lampe qui restent statiques). On utilise un soustracteur de fond
   (cv2.createBackgroundSubtractorMOG2) pour detecter le mouvement.
   Un simple seuil "orange sature" ne marche pas : il detecte la peau
   (visage, mains) et rate la flamme, dont le coeur est blanc a l'image.

2. MEILLEURE IMAGE ENVOYEE : au lieu d'envoyer la premiere image qui
   depasse le seuil, le programme garde en memoire l'image ou le feu est
   le plus visible pendant toute la fenetre de confirmation, et envoie
   celle-la.

3. NOTIFICATION NON BLOQUANTE : l'envoi Telegram se fait dans un thread
   separe pour ne pas geler l'affichage video pendant l'upload.

4. JOURNAL CSV : chaque detection est enregistree (horodatage, surface,
   ratio de mouvement, chemin de la photo) dans detections/journal_detections.csv
   -> utile pour illustrer/justifier le fonctionnement dans un rapport.

5. CONFIGURATION EXTERNALISEE dans config.json (pas besoin de toucher au code).
"""

import cv2
import numpy as np
import requests
import time
import sys
from collections import deque
import os
import json
import csv
import threading
from datetime import datetime

CONFIG_PATH = "config.json"


def charger_configuration(chemin=CONFIG_PATH):
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)


def detecter_couleur_feu(frame, config):
    """
    Detection par couleur d'une flamme, telle qu'une webcam la voit.

    Une webcam sature completement sur une flamme : son coeur apparait BLANC
    (V max, saturation quasi nulle) et non pas orange. L'orange n'est visible
    que dans le halo qui entoure ce coeur. On cherche donc :

      1. un COEUR BLANC : pixels tres lumineux (V >= 250) et peu satures ;
      2. autour de ce coeur, un HALO CHAUD : teinte orange/jaune (H 3-35)
         nettement plus saturee que le coeur (gradient de saturation).

    Ce gradient "blanc au centre -> orange autour" est propre au feu :
      - la peau est orange mais jamais blanche au centre (V plafonne ~245) ;
      - une fenetre / un mur surexpose est blanc mais son entourage l'est
        aussi (pas de gradient de saturation) ;
      - un mur beige eclaire est uniformement peu sature.

    Retourne (masque_couleur, surface_couleur) ou masque_couleur couvre le
    coeur + le halo des zones validees.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    coeur_v_min = config.get("coeur_v_min", 250)
    coeur_s_max = config.get("coeur_s_max", 60)
    min_coeur_area = config.get("min_coeur_area", 300)
    halo_epaisseur = config.get("halo_epaisseur_px", 15)
    halo_s_min = config.get("halo_s_min", 55)
    halo_gradient_s_min = config.get("halo_gradient_s_min", 30)
    halo_h_min, halo_h_max = config.get("halo_h_min", 3), config.get("halo_h_max", 35)

    masque_coeur = cv2.inRange(hsv, np.array([0, 0, coeur_v_min]), np.array([180, coeur_s_max, 255]))
    masque_coeur = cv2.morphologyEx(masque_coeur, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    nb, labels, stats, _ = cv2.connectedComponentsWithStats(masque_coeur)
    kernel_halo = np.ones((2 * halo_epaisseur + 1, 2 * halo_epaisseur + 1), np.uint8)

    masque_couleur = np.zeros(masque_coeur.shape, np.uint8)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < min_coeur_area:
            continue
        coeur = (labels == i).astype(np.uint8)
        dilate = cv2.dilate(coeur, kernel_halo)
        halo = (dilate - coeur).astype(bool)
        if not halo.any():
            continue
        s_coeur = np.median(s[coeur.astype(bool)])
        s_halo = np.median(s[halo])
        h_halo = np.median(h[halo])
        if (
            s_halo >= halo_s_min
            and s_halo - s_coeur >= halo_gradient_s_min
            and halo_h_min <= h_halo <= halo_h_max
        ):
            masque_couleur[dilate.astype(bool)] = 255

    return masque_couleur, cv2.countNonZero(masque_couleur)


def detecter_zone_feu(frame, bg_subtractor, config):
    """
    Combine le masque de couleur (coeur blanc + halo orange, voir
    detecter_couleur_feu) avec un masque de mouvement, pour ne retenir que
    les zones qui ressemblent a une flamme ET qui bougent/scintillent.

    Retourne :
      - masque final utilise pour la decision
      - surface totale detectee (pixels)
      - le plus grand contour (pour dessiner le rectangle)
      - ratio de mouvement (utile pour le score de confiance et le journal)
      - surface couleur brute (avant filtre mouvement)
    """
    masque_couleur, surface_couleur = detecter_couleur_feu(frame, config)

    kernel = np.ones((5, 5), np.uint8)
    masque_mouvement = bg_subtractor.apply(frame)
    # Une seule iteration de dilatation : un mouvement rapide (grand geste)
    # laisse deja une trainee assez large dans le masque de mouvement, pas
    # la peine d'en rajouter beaucoup au risque de recouvrir toute la piece.
    masque_mouvement = cv2.dilate(masque_mouvement, kernel, iterations=1)

    if config["require_motion"]:
        masque_final = cv2.bitwise_and(masque_couleur, masque_mouvement)
    else:
        masque_final = masque_couleur

    contours, _ = cv2.findContours(masque_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    surface_totale = 0
    plus_grand_contour = None
    plus_grande_surface = 0
    for c in contours:
        aire = cv2.contourArea(c)
        surface_totale += aire
        if aire > plus_grande_surface:
            plus_grande_surface = aire
            plus_grand_contour = c

    ratio_mouvement = surface_totale / surface_couleur if surface_couleur > 0 else 0.0

    return masque_final, surface_totale, plus_grand_contour, ratio_mouvement, surface_couleur


def envoyer_notification_telegram(token, chat_id, chemin_image, message):
    """Envoie une photo + un message texte via un bot Telegram (fonction bloquante,
    a lancer dans un thread pour ne pas figer la video)."""
    if token == "VOTRE_TOKEN_ICI":
        print("[ATTENTION] Configure telegram_bot_token et telegram_chat_id dans config.json")
        return False

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(chemin_image, "rb") as photo:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "caption": message},
                files={"photo": photo},
                timeout=15,
            )
        if response.status_code == 200:
            print("[OK] Notification envoyee avec succes.")
            return True
        else:
            print(f"[ERREUR] Telegram a repondu : {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERREUR] Envoi de la notification impossible : {e}")
        return False


def envoyer_notification_async(token, chat_id, chemin_image, message):
    """Lance l'envoi Telegram dans un thread separe (non bloquant)."""
    thread = threading.Thread(
        target=envoyer_notification_telegram,
        args=(token, chat_id, chemin_image, message),
        daemon=True,
    )
    thread.start()


def enregistrer_dans_journal(chemin_log, horodatage, surface, ratio_mouvement, chemin_photo):
    """Ajoute une ligne au journal CSV des detections (cree le fichier + entetes si besoin)."""
    fichier_existe = os.path.exists(chemin_log)
    os.makedirs(os.path.dirname(chemin_log), exist_ok=True)
    with open(chemin_log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not fichier_existe:
            writer.writerow(["horodatage", "surface_pixels", "ratio_mouvement", "photo"])
        writer.writerow([horodatage, int(surface), round(ratio_mouvement, 3), chemin_photo])


def main():
    config = charger_configuration()

    dossier_detections = config["detections_folder"]
    if config["save_detections"] and not os.path.exists(dossier_detections):
        os.makedirs(dossier_detections)

    source = config["camera_source"]

    if isinstance(source, int):
        # Sur macOS, forcer le backend AVFoundation evite souvent les
        # echecs d'ouverture/lecture avec le backend par defaut.
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la camera. Verifie 'camera_source' dans config.json.")
        return

    # Phase de demarrage : certaines webcams (notamment sur macOS) mettent
    # quelques dizaines de millisecondes a fournir leur toute premiere image.
    # On retente plusieurs fois avant d'abandonner, plutot que d'echouer
    # directement sur la premiere lecture.
    ret = False
    for tentative in range(30):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)

    if not ret:
        print("[ERREUR] La camera s'est ouverte mais ne renvoie aucune image.")
        print("         Verifie qu'aucune autre application n'utilise deja la camera,")
        print("         et que l'acces camera est bien autorise pour ce terminal.")
        cap.release()
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    # --- Calibration du fond ---
    # Au tout debut, MOG2 n'a encore aucune idee de ce qu'est "le fond" de
    # l'image : il considere donc tout comme du mouvement (y compris une
    # personne immobile), ce qui peut declencher une fausse alerte des les
    # premieres secondes. On "nourrit" le modele avec quelques dizaines
    # d'images avant de commencer la detection, idealement sans personne
    # dans le champ pour que le fond appris soit une piece vide/stable.
    frames_calibration = 60
    print(f"[INFO] Calibration du fond en cours ({frames_calibration} images, evite de bouger)...")
    for _ in range(frames_calibration):
        ret_calib, frame_calib = cap.read()
        if ret_calib:
            bg_subtractor.apply(frame_calib)
    print("[INFO] Calibration terminee.")

    print("[INFO] Detecteur d'incendie demarre. Appuie sur 'q' pour quitter.")

    compteur_frames_consecutives = 0
    dernier_envoi = 0
    meilleure_image = None
    meilleure_surface = 0
    meilleur_ratio_mouvement = 0.0

    dernier_temps = time.time()
    fps_affiche = 0.0
    dernier_countdown_affiche = -1
    historique_surface_couleur = deque(maxlen=config.get("flicker_frames_historique", 15))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire une image de la camera.")
            break

        # Calcul du FPS (affichage uniquement)
        maintenant = time.time()
        delta = maintenant - dernier_temps
        dernier_temps = maintenant
        if delta > 0:
            fps_affiche = 0.9 * fps_affiche + 0.1 * (1.0 / delta)

        # --- Countdown du cooldown dans la console ---
        if dernier_envoi > 0:
            restant = config["cooldown_seconds"] - (maintenant - dernier_envoi)
            if restant > 0:
                restant_arrondi = int(restant) + 1
                if restant_arrondi != dernier_countdown_affiche:
                    dernier_countdown_affiche = restant_arrondi
                    sys.stdout.write(f"\r[INFO] Prochaine alerte possible dans {restant_arrondi:2d}s... ")
                    sys.stdout.flush()
            elif dernier_countdown_affiche != 0:
                dernier_countdown_affiche = 0
                sys.stdout.write("\r[INFO] Pret a detecter une nouvelle alerte.                    \n")
                sys.stdout.flush()

        masque, surface, contour, ratio_mouvement, surface_couleur_brute = detecter_zone_feu(
            frame, bg_subtractor, config
        )

        # --- Analyse du scintillement ---
        # Une lumiere ambiante (ampoule orange, lampadaire) reste globalement
        # stable dans le temps : la surface de couleur "feu" detectee varie peu
        # d'une image a l'autre. Une vraie flamme, elle, scintille en
        # permanence : sa surface colorée fluctue nettement. On mesure cette
        # fluctuation avec le coefficient de variation (ecart-type / moyenne)
        # de la surface couleur sur les dernieres images.
        historique_surface_couleur.append(surface)
        scintillement = 0.0
        if len(historique_surface_couleur) == historique_surface_couleur.maxlen:
            moyenne_surface = np.mean(historique_surface_couleur)
            ecart_type_surface = np.std(historique_surface_couleur)
            # Plancher de securite : evite qu'un bruit minime (quelques pixels
            # parasites) ne produise un ratio enorme quand la moyenne est
            # quasiment nulle. Le scintillement ne devient significatif que
            # s'il y a deja un minimum de surface coloree reelle.
            plancher_surface = max(config["min_fire_area"] * 0.05, 50)
            scintillement = ecart_type_surface / max(moyenne_surface, plancher_surface)

        utiliser_flicker = config.get("utiliser_flicker", True)
        seuil_flicker = config.get("seuil_flicker", 0.15)
        flicker_ok = (
            not utiliser_flicker
            or len(historique_surface_couleur) < historique_surface_couleur.maxlen
            or scintillement >= seuil_flicker
        )

        feu_detecte_sur_cette_image = (
            surface > config["min_fire_area"]
            and (not config["require_motion"] or ratio_mouvement >= config["motion_overlap_ratio_min"])
            and flicker_ok
        )

        affichage = frame.copy()

        if feu_detecte_sur_cette_image:
            compteur_frames_consecutives += 1

            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(affichage, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(affichage, "FEU DETECTE", (x, max(y - 15, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # On garde la meilleure image (feu le plus visible) de la sequence en cours
            if surface > meilleure_surface:
                meilleure_surface = surface
                meilleure_image = affichage.copy()
                meilleur_ratio_mouvement = ratio_mouvement
        else:
            compteur_frames_consecutives = 0
            meilleure_image = None
            meilleure_surface = 0
            meilleur_ratio_mouvement = 0.0

        # --- Overlay d'informations a l'ecran ---
        confiance = min(100, int(100 * compteur_frames_consecutives / config["consecutive_frames_required"]))
        cv2.putText(affichage, f"FPS: {fps_affiche:.1f}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(affichage, f"Confiance feu: {confiance}%", (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(affichage, f"Scintillement: {scintillement:.2f} (seuil {config.get('seuil_flicker', 0.15)})",
                    (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        if dernier_envoi > 0:
            restant_affichage = config["cooldown_seconds"] - (maintenant - dernier_envoi)
            if restant_affichage > 0:
                cv2.putText(affichage, f"Cooldown: {int(restant_affichage) + 1}s", (15, 195),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

        alerte_confirmee = compteur_frames_consecutives >= config["consecutive_frames_required"]
        cooldown_ecoule = (maintenant - dernier_envoi) > config["cooldown_seconds"]

        if alerte_confirmee and cooldown_ecoule:
            horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_a_envoyer = meilleure_image if meilleure_image is not None else affichage
            nom_fichier = (
                f"{dossier_detections}/incendie_{horodatage}.jpg"
                if config["save_detections"] else f"incendie_{horodatage}.jpg"
            )
            cv2.imwrite(nom_fichier, image_a_envoyer)

            message = f"🔥 Incendie detecte le {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')} !"
            envoyer_notification_async(
                config["telegram_bot_token"], config["telegram_chat_id"], nom_fichier, message
            )

            enregistrer_dans_journal(
                config["log_file"], horodatage, meilleure_surface, meilleur_ratio_mouvement, nom_fichier
            )

            dernier_envoi = maintenant
            compteur_frames_consecutives = 0
            meilleure_image = None
            meilleure_surface = 0

            if not config["save_detections"]:
                os.remove(nom_fichier)

        cv2.imshow("Detecteur d'incendie - camera", affichage)
        cv2.imshow("Masque de detection (debug)", masque)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()