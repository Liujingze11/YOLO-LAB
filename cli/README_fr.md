# YOLO Lab CLI

[English](README.md) | [中文](README_zh.md) | [Español](README_es.md)

Outil de formation en ligne de commande pour la segmentation YOLO, basé sur Ultralytics.

## Fonctionnalités

- Trois modes d'entraînement : Nouveau / Reprendre / Ajuster
- Augmentation de données activable
- Validation automatique avec journalisation CSV (métriques globales et par classe)
- Isolation des expériences : chaque exécution crée des répertoires et journaux indépendants
- Paramètres CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Détection automatique de la langue système (zh/en/fr/es), avec `--lang` pour forcer

## Démarrage Rapide

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python scripts/train_segment.py
```

## Prérequis

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## Modes d'Entraînement

Lancez `python scripts/train_segment.py` et choisissez :

- **1** — Nouvel entraînement depuis les poids initiaux
- **2** — Reprendre depuis last.pt
- **3** — Ajuster depuis le best.pt historique

## Options CLI

```bash
python scripts/train_segment.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mon_experience
```

La langue est détectée automatiquement. Forcer avec `--lang` :

```bash
python scripts/train_segment.py --lang fr   # Français
python scripts/train_segment.py --lang en   # English
python scripts/train_segment.py --lang zh   # 中文
python scripts/train_segment.py --lang es   # Español
```

## Résultats

- Résultats : `result/<experiment_name>/weights/` (best.pt, last.pt)
- Journaux CSV : `train_logs/`

## License

MIT
