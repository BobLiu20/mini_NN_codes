### What
* 这里使用纯Python实现了标准的LSTM网络
* 训练完后可保存Model，并在预测的时候加载训练好的Model

### How
1. 训练
在命令行执行
```
python training.py
```
通过按Ctrl+C终止训练(early stopping)，并会自动保存Model为test.model

2. 预测
直接在命令行执行
```
python predict.py
```

### Dependencies
Python 2.7 环境
* Python numpy　(用于矩阵计算)
* Python pickle (用于序列化保存model)

