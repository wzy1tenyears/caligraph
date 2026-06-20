# caligraph

汉字笔顺书写动画生成工具。支持从 Hanziwu 获取笔顺数据，生成 MP4/GIF 动画。

## 功能

- 多字输入
- 交互式中文 CLI
- 自定义分辨率
- 自定义字体/字体文件
- 自定义 FPS
- 自定义动画速度
- 输出 MP4、GIF 或同时输出

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 使用

交互模式：

```powershell
python __main__.py
```

命令行模式：

```powershell
python __main__.py --text "\u6c38\u6c38" --font simkai --resolution 88 --fps 12 --speed 1.5 --format mp4 --color "#00ffcc" --threads 4 --cuda auto --output output.mp4
```

查看可用字体：

```powershell
python __main__.py --list-fonts
```

## 打包

目录版 EXE：

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm --noupx --onedir --name caligraph --copy-metadata imageio --copy-metadata imageio-ffmpeg __main__.py
```

产物位于：

```text
dist\caligraph\caligraph.exe
```

发布 Release 时可压缩整个 `dist\caligraph` 目录。
