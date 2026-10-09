# Python 图像处理工具

## 运行

Windows PowerShell（建议 Python 3.10+）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

程序支持打开彩色或灰度图片、原图/结果双视图、滚轮缩放和拖动平移（两视图缩放同步），
线性、对数及 gamma 变换。参数修改采用约 180ms 防抖预览，可在 NumPy 与 OpenCV 后端间切换。
右侧显示尺寸、灰度统计和直方图；“撤销/暂存”保存最近变换，支持单张 PNG/JPEG 保存及
多图序列导出。序列输出目录会生成 `export_log.json`（含参数、输入输出和成功状态）。


路径由 Qt 对话框处理，内部统一使用 `pathlib.Path`，适用于 Windows 路径。