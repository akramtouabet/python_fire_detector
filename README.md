# Détecteur d'incendie par caméra (version améliorée)

Détecte le feu en temps réel via une caméra et envoie une notification
Telegram (avec photo) sur ton téléphone dès qu'un incendie est confirmé.

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Créer le bot Telegram (5 minutes)

1. Ouvre Telegram, cherche **@BotFather**, envoie `/newbot`.
2. Donne un nom et un nom d'utilisateur à ton bot (ex: `MonDetecteurFeuBot`).
3. BotFather te donne un **token** du style `123456789:ABCdefGhIJKlmNoPQRstuVwxYZ`.
   → Colle-le dans `"telegram_bot_token"` dans `config.json`.
4. Envoie **n'importe quel message** à ton nouveau bot (obligatoire, sinon il
   ne peut pas t'envoyer de messages).
5. Récupère ton `chat_id` en ouvrant dans un navigateur :
   `https://api.telegram.org/bot/getUpdates`
   → tu verras un champ `"chat":{"id": 123456789, ...}`.
   → Colle ce nombre dans `"telegram_chat_id"` dans `config.json`.

## 3. Lancer le programme

```bash
python fire_detector.py
```

Deux fenêtres s'ouvrent : la vidéo (avec rectangle rouge + FPS + niveau de
confiance) et le masque de détection (debug). Appuie sur `q` pour quitter.

## 4. Fichier `config.json`

Tous les réglages sont dans `config.json`, pas besoin de toucher au code :

| Paramètre | Rôle |
|---|---|
| `camera_source` | `0` = webcam. Mets une URL RTSP/HTTP pour une caméra IP. |
| `min_fire_area` | Surface minimum (pixels, cœur + halo) pour considérer une zone comme du feu. |
| `min_coeur_area` | Surface minimum (pixels) du cœur blanc d'une flamme. |
| `coeur_v_min`, `coeur_s_max` | Un pixel est "cœur blanc" si sa luminosité V ≥ `coeur_v_min` et sa saturation S ≤ `coeur_s_max`. |
| `halo_epaisseur_px` | Épaisseur (pixels) de l'anneau analysé autour du cœur. |
| `halo_s_min`, `halo_gradient_s_min` | Le halo doit être saturé (S ≥ `halo_s_min`) et nettement plus saturé que le cœur (écart ≥ `halo_gradient_s_min`). |
| `halo_h_min`, `halo_h_max` | Teinte HSV admise pour le halo (3-35 = rouge/orange/jaune). |
| `consecutive_frames_required` | Nombre d'images consécutives avant de déclencher l'alerte. |
| `cooldown_seconds` | Délai minimum entre deux notifications. |
| `require_motion` | Si `true`, exige que la zone flamme soit aussi en mouvement (recommandé). |
| `motion_overlap_ratio_min` | Sensibilité du filtre de mouvement (0 à 1). |
| `save_detections` | Garde une copie locale des photos envoyées. |
| `log_file` | Fichier CSV où chaque détection est journalisée. |

## Comment ça marche (pour ton dossier/soutenance)

**Détection couleur** : conversion en HSV, puis recherche de la signature
d'une flamme **telle qu'une caméra la voit** : la caméra sature sur la
flamme, dont le cœur apparaît donc **blanc** (luminosité maximale, saturation
quasi nulle), entouré d'un **halo orange/jaune** nettement plus saturé. On
cherche d'abord les taches blanches très lumineuses, puis on ne garde que
celles dont l'anneau environnant est orange et plus saturé que le centre
(gradient de saturation "blanc → orange").

Pourquoi pas un simple seuil "orange saturé" ? Testé sur 17 captures réelles,
il détectait le visage et les mains (peau = orange saturé, H≈10 S≈118 V≈170)
et **ratait la flamme du briquet** (cœur S≈10 V≈255, trop peu saturé). Le
critère cœur blanc + halo, lui, trouve les 3 flammes et ignore les 14 images
sans feu. Le script `test_captures.py` permet de rejouer ce test :

```bash
python test_captures.py --flammes 194642 195900 200816
```

**Filtre de mouvement** : un simple filtre couleur confond facilement le feu
avec un mur orange, un coucher de soleil ou une lampe allumée, car ces objets
ont la bonne couleur mais restent fixes. Le programme utilise donc en plus un
**soustracteur de fond** (`cv2.createBackgroundSubtractorMOG2`), qui détecte
les pixels en mouvement. Une zone n'est retenue comme "feu" que si elle est
**à la fois** de la bonne couleur **et** en mouvement — ce qui correspond au
scintillement caractéristique des flammes. C'est le principe de base utilisé
dans la plupart des systèmes de détection d'incendie par vision par
ordinateur (couleur + dynamique temporelle).

**Confirmation multi-images + meilleure photo** : pour éviter qu'un reflet
ponctuel ne déclenche une fausse alerte, il faut que le feu soit détecté sur
plusieurs images consécutives. Pendant cette fenêtre, le programme garde en
mémoire l'image où le feu est le plus visible et c'est celle-ci qui est
envoyée (pas la première image, souvent moins nette).

**Notification asynchrone** : l'envoi vers l'API Telegram (`sendPhoto`) se
fait dans un thread séparé, pour que l'affichage vidéo ne se fige pas
pendant l'upload.

**Journalisation** : chaque détection confirmée est ajoutée à
`detections/journal_detections.csv` (horodatage, surface détectée, ratio de
mouvement, chemin de la photo) — pratique pour illustrer le fonctionnement
et présenter des statistiques dans ton rapport.

## Pour aller encore plus loin (optionnel)

La détection couleur + mouvement est fiable et suffisante pour un projet,
mais la référence en la matière reste un **modèle de deep learning entraîné
spécifiquement sur des images de feu/fumée** (ex: YOLOv8 avec un dataset
"fire detection" public sur Roboflow ou Kaggle). Le principe resterait le
même (capture caméra → détection → notification), seule la fonction
`detecter_zone_feu()` serait remplacée par une inférence du modèle. C'est
plus long à mettre en place (entraînement ou récupération d'un modèle
pré-entraîné fiable) mais donne de meilleurs résultats sur des scènes
complexes. Si tu veux, je peux te faire cette version aussi.

## Limites restantes

- Une caméra fixe braquée sur une flamme immobile de faible intensité
  (bougie, briquet) pourrait, selon les réglages, ne générer que peu de
  mouvement détectable — ajuste `motion_overlap_ratio_min` si besoin.
- Le système suppose un bon éclairage ; en vision nocturne infrarouge, les
  couleurs ne sont plus fiables (il faudrait alors se baser uniquement sur
  la luminosité/mouvement, ou un capteur thermique).
