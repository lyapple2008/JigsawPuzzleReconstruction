# iOS WDA 连接问题记录

## 问题描述

运行 `tidevice wdaproxy -p 8100` 时报错：
- `'tuple' object has no attribute 'cert_reqs'`
- `No app matches com.*.xctrunner`

## 报错详情

```
[W 260317 21:59:25 _wdaproxy:65] [00008150-00152C6A117A401C] Unknown exception: 'tuple' object has no attribute 'cert_reqs'
[E 260317 21:59:26 _wdaproxy:153] [00008150-00152C6A117A401C] wda started failed
tidevice.exceptions.MuxError: [Errno No app matches] com.*.xctrunner
```

## 根本原因分析

### 原因1：WDA Bundle ID 不匹配

设备上安装的 WDA (WebDriverAgent) 的 bundle ID 与 tidevice 默认查找的不匹配：

| 项目 | 值 |
|------|-----|
| 实际 WDA Bundle ID | `xyz.beyoung.WebDriverAgentRunner.xctrunner` |
| tidevice 查找匹配 | `com.*.xctrunner` |

### 原因2：SSL 配置问题

tidevice 内部 SSL 配置与新版 Python 依赖存在兼容性问题。

## 解决方案

**推荐方案：使用 facebook-wda 直接 USB 连接**

facebook-wda 可以直接通过 USB 连接到 iOS 设备，无需启动 wdaproxy 服务。

### 安装依赖

```bash
conda activate jigsaw
pip install facebook-wda
```

### 测试连接

```python
import wda

# 通过 USB 连接
c = wda.USBClient(udid='00008150-00152C6A117A401C')

# 检查状态
print(c.status())
# 输出: {'build': {...}, 'os': {...}, 'message': 'WebDriverAgent is ready to accept commands', 'ready': True, ...}

# 截图
img = c.screenshot()
img.save('screenshot.png')
```

### 在项目代码中使用

ios_auto 模块已支持 facebook-wda：

```python
from ios_auto.connector import DeviceConnector

# USB 连接（推荐）
connector = DeviceConnector(udid='00008150-00152C6A117A401C')

# 或 HTTP 连接（需要先启动 wdaproxy）
connector = DeviceConnector(url='http://localhost:8100')

with connector as dev:
    screenshot = dev.client.screenshot()
    screenshot.save('game.png')
```

## 设备信息

| 属性 | 值 |
|------|-----|
| UDID | 00008150-00152C6A117A401C |
| WDA Bundle ID | xyz.beyoung.WebDriverAgentRunner.xctrunner |
| iOS 版本 | 26.3.1 |
| 屏幕分辨率 | 1206 x 2622 |

## 相关文件

- `ios_auto/connector.py` - 设备连接管理
- `ios_auto/screenshot.py` - 截图功能
- `ios_auto/gesture.py` - 手势操作
- `tasks/ios_auto_puzzel_solver.md` - 项目设计文档
