# lightning deals for tennis
## Update
* 2023-07-06
  * 优化程序交互问题，增加等待处理框的判断，使得有效动作更快
* 2023-05-18
  * 增加目标检测轮询，解决速度过快，场地未刷新的问题
* 2023-05-16
  * 减少无效步骤，初始界面在抢场地界面
  * 减少无效time.sleep
  * 更新本地时间，同步国家授时中心时间，增加阿里云ntp服务：ntp.aliyun.com
  ![img.png](img/img.png)
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
  "tennis_window": [0, 37, 414, 817],
  "wilson": [85, 241],
  "enter": [321, 724],
  "indoor": [206, 413],
  "last_time": [368, 492],
  "submit_button": [337, 718],
  "puzzle_button_y": 520,
  "black_window": [180, 375]
}

```
* tennis_window[x1, y1, x2, y2]:(x1, y1) is the left-top position and (x2, y2) is the right-down position
* enter[x1, y1]: the position of enter
* indoor[x1, y1]: the position of indoor
* last_time[x1, y1]: the latest time position
* submit_button[x1, y1]: submit button position
* puzzle_button_y[y1]: the y index of puzzle_button
* black_window[x1, y1]: the position of the wait interaction
