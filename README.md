# horizon project

TODO:

* [ ] 在 filtering 车的时候，应该更合理写，比如当前右转的时候，右转的自行车会侵占车道，可能也需要考虑。是否考虑增加一个小圈(2~3米)，无论那个车道都需要考虑

#### bugs

* [x] pad车未做坐标变换

* [x] pad 车的位置错误

* [x] 在step的时候子车消失的问题，不知道问什么小时，但一旦捕获了该错误，则把子车再加回去

* [ ] 当某个类别的车辆数量是零的时候会导致bug

* [ ] 自车会停在路口，即使没有红灯虚拟车

* [ ] 周车 太少了

* [ ] 终点设置的位置的， 和参考轨迹终点的位置。

* [ ] 左下角的路口怎么处理


#### TEMP: 兼容每一个十字路口
* [X] EE6
* [ ] ES8
* [x] ES9
* [x] EE10
* [x] EN13
* [x] EW8
* [x] EW7
* [x] EW6
* [x] EW5
* [ ] EW4
* [x] EN4
* [x] EE2
* [x] ES7

#### 环境配置说明

SUMO 需要使用`1.7.0`版本， 需添加环境变量`SUMO_HOME`，Ubuntu需要添加至`/etc/profile`, （仅添加至`.bashrc` pycharm读不到）

Windows版的besizer包有bug,会报找不到DLL的错，可以将如下四行代码添加到besizer包的 `__init__.py`文件中（添加到全部非注释代码之前）

```python
import ctypes
import os
extral_dll_path = "bezier-8f9b8e7c.dll"
file_path = os.path.dirname(__file__)
ctypes.cdll.LoadLibrary(os.path.join(file_path, 'extra-dll', extral_dll_path))
```

其他依赖库请参考 `requirements.txt`

#### TIPS

若 clone 的时候报错（SSL），使用 `SSH` 代替 `HTTPS` 该解决该问题。

打不开 GitHub 是在可使用 dev-sidecar

#### 未来可能存在的问题
1、考虑周车尺寸改变

2、感知加上关于红灯虚拟车的后处理

3、感知获得状态后要做状态归一化才能输入到网络，感知拿到的是相对坐标

4、训练时是假定车辆为小轿车，双圆约束较小，如果利用bus的中心点，会撞bus

5、能否利用真是的数据验证算法。

