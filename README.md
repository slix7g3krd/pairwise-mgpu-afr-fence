# pairwise-mgpu-afr-fence

## 领域

游戏、图形与多媒体

## 主题

多 GPU 交替帧

## 说明

跨适配器同步、显存合并与交换链重建。

初始状态下 `graphics/mgpu_afr_fence.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
