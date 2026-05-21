# YOLO Lab CLI

[English](README.md) | [中文](README_zh.md) | [Français](README_fr.md)

Herramienta de línea de comandos para entrenamiento de segmentación YOLO, basada en Ultralytics.

## Funcionalidades

- Tres modos de entrenamiento: Nuevo / Reanudar / Ajustar
- Aumento de datos activable
- Validación automática con registro CSV (métricas globales y por clase)
- Aislamiento de experimentos: cada ejecución crea directorios y registros independientes
- Parámetros CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)

## Inicio Rápido

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python scripts/train_segment.py
```

## Requisitos

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## Modos de Entrenamiento

Ejecute `python scripts/train_segment.py` y elija:

- **1** — Nuevo entrenamiento desde pesos iniciales
- **2** — Reanudar desde last.pt
- **3** — Ajustar desde best.pt histórico

## Opciones CLI

```bash
python scripts/train_segment.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mi_experimento
```

## Resultados

- Resultados: `result/<experiment_name>/weights/` (best.pt, last.pt)
- Registros CSV: `train_logs/`

## Licencia

MIT
