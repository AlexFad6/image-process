# Python 图像处理工具

## 用户运行

从右侧Release下载[image-process](https://github.com/AlexFad6/image-process/releases/latest)最新发行版即可运行。

## 快捷键
- 左键长按拖动图片
- 滚轮缩放图片

## 开发者
Windows PowerShell（建议 Python 3.10+）：

```powershell
python -m pip install -r requirements.txt
python main.py
```

### 多语言翻译

翻译源文件位于 `translations/`。安装 PySide6 后运行：

```powershell
python update_translations.py
```

该脚本使用 `pyside6-lupdate`（或 `pylupdate6`）提取字符串，并使用
`pyside6-lrelease` 将 `.ts` 编译为运行时加载的 `.qm` 文件。