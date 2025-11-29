# Docker Setup pour Chatterbox Multilingual

Ce répertoire contient la configuration Docker pour exécuter le projet Chatterbox Multilingual.

## Fichiers inclus

- `Dockerfile` : Configuration de l'image Docker avec Python 3.11
- `docker-compose.yml` : Configuration pour orchestrer le container
- `README.md` : Ce fichier de documentation

## Utilisation

### 1. Construire et lancer avec docker-compose

```bash
cd docker
docker-compose up --build
```

### 2. Lancer uniquement (si l'image existe déjà)

```bash
cd docker
docker-compose up
```

### 3. Lancer en arrière-plan

```bash
cd docker
docker-compose up -d
```

### 4. Arrêter le container

```bash
cd docker
docker-compose down
```

## Fonctionnalités

- **Auto-exécution** : Le fichier `example_tts.py` est automatiquement exécuté au démarrage
- **Volume externe** : Le code source est monté depuis l'hôte, pas besoin de reconstruire l'image pour les modifications de code
- **Python 3.11** : Version spécifiée comme demandé
- **Toutes les dépendances** : Installation automatique via `pyproject.toml`
- **Sortie audio** : Les fichiers générés sont accessibles dans le répertoire `output/`

## Structure des volumes

- `../:/app` : Code source monté depuis le répertoire parent
- `./output:/app/output` : Répertoire pour les fichiers audio générés

## Support GPU

Le support GPU NVIDIA est activé par défaut dans la configuration. Assurez-vous que :

1. **NVIDIA Container Toolkit** est installé sur votre système
2. **nvidia-docker2** est configuré
3. Votre système a un GPU NVIDIA compatible

Pour désactiver le GPU (utiliser CPU uniquement), commentez les lignes dans `docker-compose.yml` :

```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

## Dépendances système installées

- Python 3.11
- ffmpeg
- libsndfile1
- build-essential
- git
- wget

## Variables d'environnement

- `PYTHONPATH=/app/src` : Chemin Python configuré pour le projet
- `PYTHONUNBUFFERED=1` : Sortie non mise en buffer
- `PYTHONDONTWRITEBYTECODE=1` : Pas de fichiers .pyc