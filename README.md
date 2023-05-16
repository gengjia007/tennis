# lightning deals for tennis
## Update
* 2023-05-16
  * 减少无效步骤，初始界面在抢场地界面
  * 减少无效time.sleep
  * 由于本地时间比标准北京时间慢300ms，设置起始时间为11:59:59.7
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

## 2.Run
```bash
python tennis.py --source ./screenshot/area --weights runs/train/area_model/weights/best.pt --puzzle-source ./screenshot/puzzle --puzzle-weights runs/train/puzzle_model/weights/best.pt
```

## 3.(**important**)position json file
* please check the position.json before running
```json
{
  "tennis_window": [406, 226, 816, 1007],
  "enter": [320, 733],
  "indoor": [209, 413],
  "last_time": [368, 492],
  "submit_button": [337, 718],
  "puzzle_button_y": 520
}

```
* tennis_window[x1, y1, x2, y2]:(x1, y1) is the left-top position and (x2, y2) is the right-down position
* enter[x1, y1]: the position of enter
* indoor[x1, y1]: the position of indoor
* last_time[x1, y1]: the latest time position
* submit_button[x1, y1]: submit button position
* puzzle_button_y[y1]: the y index of puzzle_button
