### 1. What

深度学习神经网络微框架:
* 可支持任意DAG(有向无环网络)
* 可支持自定义Layer
* 无第三方库依赖，只依赖于C++　STL标准库，非常小巧
* 等等。。。

### 2. 使用说明

##### 2.1 训练 mnist 训练集60k
打开命令行，cd到源码目录，然后执行下面命令，即可开始训练
```
./main train
```
* 所有训练的超参数设置在net.cpp里，包括lr, momentum, maxIter等；
* 默认会训练10个Epoch，一个Epoch有600000个Iter；
* 默认会每一个Epoch自动保存一次Model参数，名为iter_xxx.model，其中xxx为iter数；
* 默认会每一个Epoch自动执行一次mnist测试集10k的测试，并输出精度;

##### 2.2 测试 mnist 测试集10k
打开命令行，cd到源码目录，然后执行下面命令，即可开始测试
```
./main test good_98.75.model
```
注意替换上面的 good_98.75.model 为已训练好的模型名字

##### 2.3 预测一个文件夹的图片
打开命令行，cd到源码目录，然后执行下面命令，即可开始预测
```
./main predict good_98.75.model ./images/
```
* 注意替换上面的good_98.75.model为已训练好的模型名字，替换./images/为你图片文件夹路径；
* 比较尴尬的是，时间关系没有写C++读取图片的接口，所以这个读取其它图片的接口现在是依赖于OpenCV的；
* 即需要使用此功能，需要打开MakeFile文件，将其中的 OPENCV_ENABLE 值改为1，然后执行 make clean && make　重新编译；
* 注：训练和测试mnist不依赖于OpenCV，只是读取第三方图片需要依赖；

### 3. 编译说明

##### 3.1 测试环境
* Ubuntu 14.04
* 编译环境 g++ 4.8.4 (c++11支持即可)
* 仅依赖于C++　STL标准库
* (可选：OpenCV 2.4.x)

##### 3.2 编译
打开命令行，cd到源码目录，然后执行下面命令，即可编译
```
make clean && make -j4
```

##### 3.3 可选项
如果不需要读取非mnist图片，可跳过。
为了可以读取非mnist图片，需要打开MakeFile，设置OPENCV_ENABLE = 1，然后重新执行
```
make clean && make -j4
```

### 4. 代码结构说明
    5_Tupu	# 存储源码文件
	    \__ layer               # 各个层的代码文件保存
		    \__ layer.h         # 每个定义层必须集成的基类
		    \__ convLayer.cpp 	# 卷积层
		    \__ dataLayer.cpp　　　# mnist数据输入层
		    \__ fcLayer.cpp		# 全连接层
		    \__ imageInputLayer.cpp	# 文件夹图片读取层
            \__ lossLayer.cpp	# 损失层
            \__ poolingLayer.cpp# 池化层
            \__ tanhLayer.cpp   # 激活函数层
		\__ images		        # 用于predict测试用图片保存文件夹
		\__ datas		        # mnist数据保存文件夹
		\__ net.cpp             # 网络结构定义，训练测试控制源码
		\__ main.cpp            # 入口代码
		\__ MakeFile            #　编译文件

### 5. TODO
* 待增加多CPU支持，加速训练与预测
* Layer中的神经元计算待优化
* Layer的初始化需要构建一个通用类，可自动计算输入输出尺寸
* 网络结构定义待简化
* 等等...

Bob.Liu in 2017.06.22


