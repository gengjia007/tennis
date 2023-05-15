# lightning deals for tennis
## 1. Prepare Environment
* create conda virtual environment
```bash
conda create -n py3.8 python=3.8
```
* activate the py3.8
```bash
conda activate py3.8
```
* install packages
```bash
pip install -r requirements
```

## Run
```bash
python tennis.py --source ./screenshot/area --weights runs/train/area_model/weights/best.pt --puzzle-source ./screenshot/puzzle --puzzle-weights runs/train/puzzle_model/weights/best.pt
```

