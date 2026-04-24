#!/bin/bash
# =============================================================
# setup_esp_matter_source.sh
# Script a executer UNE SEULE FOIS sur ta machine locale
# pour copier l'exemple ESP-Matter light dans ton repo GitHub.
#
# Apres ce script :
#   - esp-matter/examples/light/main/  est dans ton repo
#   - git commit + git push → ci.yml peut appliquer les patches
#
# Usage :
#   chmod +x setup_esp_matter_source.sh
#   ./setup_esp_matter_source.sh
# =============================================================

set -e

DOCKER_IMAGE="hajerrchaieb/matter-light:v1"
DEST_DIR="esp-matter/examples/light/main"

echo "======================================================"
echo " ESP-Matter Light Source — Setup Script"
echo "======================================================"
echo ""
echo "Ce script copie les sources C++ de l'exemple light"
echo "depuis le container Docker vers ton repo GitHub."
echo ""
echo "Docker image : $DOCKER_IMAGE"
echo "Destination  : $DEST_DIR"
echo ""

# ── 1. Vérifier que Docker est disponible ─────────────────
if ! command -v docker &>/dev/null; then
  echo "ERREUR : Docker n'est pas installe ou n'est pas dans le PATH"
  echo "Installe Docker Desktop : https://www.docker.com/products/docker-desktop"
  exit 1
fi
echo "Docker : OK"

# ── 2. Vérifier qu'on est dans un repo git ────────────────
if ! git rev-parse --git-dir &>/dev/null; then
  echo "ERREUR : Ce repertoire n'est pas un repo git"
  echo "Navigue vers la racine de ton repo DevSecOps-pipline-light-esp32c3"
  exit 1
fi
REPO_ROOT=$(git rev-parse --show-toplevel)
echo "Repo Git : $REPO_ROOT"

# ── 3. Créer la structure de dossiers ────────────────────
echo ""
echo ">>> Etape 1 : Creation des dossiers..."
mkdir -p "$DEST_DIR"
mkdir -p "esp-matter/examples/light/test"
mkdir -p "esp-matter/examples/light/CMakeLists.txt" 2>/dev/null || true

# ── 4. Pull l'image Docker ───────────────────────────────
echo ""
echo ">>> Etape 2 : Pull du container Docker ($DOCKER_IMAGE)..."
docker pull "$DOCKER_IMAGE"
echo "Docker image prete."

# ── 5. Copier les fichiers source depuis Docker ──────────
echo ""
echo ">>> Etape 3 : Copie des sources ESP-Matter light..."

docker run --rm \
  -v "$(pwd):/workspace" \
  "$DOCKER_IMAGE" \
  /bin/bash -c "
    set -e
    SRC='/opt/espressif/esp-matter/examples/light/main'
    DEST='/workspace/$DEST_DIR'

    echo 'Fichiers disponibles dans le container :'
    ls \"\$SRC\" 2>/dev/null || echo 'Dossier main non trouve'

    echo 'Copie en cours...'
    cp -r \"\$SRC\"/* \"\$DEST\"/ 2>/dev/null || true

    # Copier aussi les CMakeLists du projet
    cp /opt/espressif/esp-matter/examples/light/CMakeLists.txt \
       /workspace/esp-matter/examples/light/ 2>/dev/null || true
    cp /opt/espressif/esp-matter/examples/light/main/CMakeLists.txt \
       /workspace/$DEST_DIR/ 2>/dev/null || true

    echo 'Copie terminee.'
    echo 'Fichiers copies :'
    ls \"\$DEST\"
  "

# ── 6. Vérifier ce qui a été copié ──────────────────────
echo ""
echo ">>> Etape 4 : Verification..."
FILES=$(ls "$DEST_DIR" 2>/dev/null | wc -l)
echo "Fichiers copies dans $DEST_DIR : $FILES"

if [ "$FILES" -eq "0" ]; then
  echo ""
  echo "ERREUR : Aucun fichier copie !"
  echo "Verifier que l'image Docker contient bien :"
  echo "  /opt/espressif/esp-matter/examples/light/main/"
  docker run --rm "$DOCKER_IMAGE" ls /opt/espressif/esp-matter/examples/light/main/ 2>/dev/null || true
  exit 1
fi

echo ""
echo "Fichiers copiés :"
ls -la "$DEST_DIR"

# ── 7. Créer le .gitignore pour les fichiers binaires ────
echo ""
echo ">>> Etape 5 : Creation du .gitignore pour esp-matter/..."
cat > esp-matter/.gitignore << 'GITEOF'
# Fichiers de build ESP-IDF (ne pas commiter)
build/
sdkconfig
sdkconfig.old
*.o
*.a
*.elf
*.bin
*.map
*.d
*.cmake
CMakeCache.txt
CMakeFiles/
GITEOF
echo ".gitignore créé"

# ── 8. Créer le README dans le dossier source ────────────
cat > esp-matter/README.md << 'READEOF'
# ESP-Matter Light Source

Ce dossier contient les fichiers source de l'exemple `light` d'ESP-Matter,
copiés depuis le container Docker `hajerrchaieb/matter-light:v1`.

## Contenu

```
esp-matter/
  examples/
    light/
      main/
        app_main.cpp      ← Fichier principal
        app_driver.cpp    ← Pilote LED/PWM
        app_priv.h        ← Headers privés
        CMakeLists.txt    ← Config build
```

## Pourquoi ces fichiers sont dans le repo ?

L'AutoFix Agent (Agent 8) génère des patches unifiés pour ces fichiers.
Le Stage 4c du pipeline applique ces patches, commite sur une branche
`autofix/run-XXX`, et déclenche un nouveau pipeline CI complet.

## Ne pas modifier manuellement

Ces fichiers sont gérés par le pipeline DevSecOps.
Les modifications viennent des patches générés par Agent 8.
READEOF

# ── 9. Ajouter à git ─────────────────────────────────────
echo ""
echo ">>> Etape 6 : Ajout a git..."
git add esp-matter/
git status --short esp-matter/

echo ""
echo "======================================================"
echo " Sources copiees avec succes !"
echo "======================================================"
echo ""
echo "Prochaines etapes :"
echo ""
echo "  1. Verifier les fichiers :"
echo "     ls esp-matter/examples/light/main/"
echo ""
echo "  2. Commiter :"
echo "     git commit -m 'feat: add ESP-Matter light source for AutoFix patches'"
echo ""
echo "  3. Pusher :"
echo "     git push origin main"
echo ""
echo "  4. Le prochain run CI pourra appliquer les patches"
echo "     directement sur ces fichiers dans le repo."
echo ""
echo "Taille du dossier :"
du -sh esp-matter/ 2>/dev/null || true