# Arras.io YOLO Overlay 🎮🤖

Real-time object detection overlay for [Arras.io](https://arras.io) using YOLO (You Only Look Once). Detect bullets, tanks, shapes, and other game objects with an AI-powered visual overlay.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Detected Objects](#-detected-objects)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## ✨ Features

- 🎯 **Real-time Detection**: Identifies game objects with ~1 FPS detection rate
- 🌈 **Color-coded Overlay**: Different colors for each object class
- ⚡ **Adjustable Confidence**: Slider to tune detection sensitivity
- 📊 **Live Stats**: Detection count and latency monitoring
- 🖥️ **Browser-based**: Simple web interface on `localhost:7280`
- 🎨 **Non-intrusive**: Transparent overlay with pointer-events pass-through

## 🔧 Prerequisites

> [!IMPORTANT]
> You must have Python 3.8 or higher installed on your system. Check with `python3 --version`.

- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge recommended)
- Webcam/Screen capture permissions

> [!NOTE]
> This project uses YOLOv8 from Ultralytics for object detection. The model file (`best.pt`) should be present in the project root.

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/maple-underscore/arras_overlay.git
   cd arras_overlay
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   The following packages will be installed:
   - `numpy>=1.26.0` - Numerical computing
   - `opencv-python>=4.9.0` - Computer vision operations
   - `ultralytics>=8.1.0` - YOLO implementation

> [!TIP]
> Use a virtual environment to avoid package conflicts:
> ```bash
> python3 -m venv venv
> source venv/bin/activate  # On macOS/Linux
> # or
> venv\Scripts\activate  # On Windows
> pip install -r requirements.txt
> ```

## 🚀 Usage

1. **Start the server**
   ```bash
   python3 arras.py
   ```

2. **Open in browser**
   
   The script will automatically open `http://localhost:7280` in your default browser.

3. **Start detection**
   - Click the **"Start YOLO"** button in the top-right controls panel
   - Grant screen capture permissions when prompted
   - Select the Arras.io window/tab to monitor

4. **Adjust settings**
   - Use the **Confidence** slider to adjust detection sensitivity
   - Lower values (0.05-0.3): More detections, possibly more false positives
   - Higher values (0.4-0.95): Fewer but more accurate detections

> [!TIP]
> For best results, set the confidence threshold to around 0.2-0.3. This provides a good balance between detecting objects and avoiding false positives.

## ⚙️ Configuration

Edit the configuration section in `arras.py`:

```python
MODEL_PATH = "best.pt"              # Path to YOLO model
CLASSES_FILE = "dataset/classes.txt"  # Class names file
PORT = 7280                         # Server port
CONF_THRESHOLD = 0.2                # Default confidence (0.0-1.0)
```

> [!WARNING]
> Changing the `MODEL_PATH` requires a compatible YOLOv8 model trained on Arras.io objects. Using a different model may result in poor or no detections.

## 🎯 Detected Objects

The overlay can detect the following Arras.io game elements:

| Object | Color | Description |
|--------|-------|-------------|
| 🔴 Bullet | Red | Projectiles fired by tanks |
| ⚪ Egg | White | Spawning objects |
| 🔷 Hexagon | Cyan | Hexagonal shapes |
| 🟣 Pentagon | Purple | Pentagon shapes |
| 🟢 Player | Green | Player-controlled tanks |
| 🟡 Square | Yellow | Square shapes |
| 🟠 Triangle | Orange | Triangle shapes |

> [!NOTE]
> Detection accuracy depends on game visibility, object size, and confidence threshold settings.

## 🐛 Troubleshooting

### Iframe Not Loading

> [!CAUTION]
> If the Arras.io iframe fails to load due to browser security policies, you'll see a fallback message. In this case:
> 1. Open [arras.io/#wpd](https://arras.io/#wpd) in a separate tab
> 2. Click "Start YOLO" and select that tab when prompted

### Low FPS / High Latency

> [!TIP]
> If detection is slow:
> - Close other resource-intensive applications
> - Ensure your GPU drivers are up to date (if using CUDA)
> - Consider reducing the game window size
> - Lower the capture resolution in the browser

### Model Not Found Error

```
FileNotFoundError: [Errno 2] No such file or directory: 'best.pt'
```

> [!IMPORTANT]
> The trained YOLO model (`best.pt`) must be in the project root directory. If you don't have this file, you'll need to train your own model or obtain it from the project maintainer.

### No Detections Appearing

> [!TIP]
> Try these solutions:
> - Lower the confidence threshold using the slider
> - Ensure the game is clearly visible in the captured window
> - Check that objects are large enough to be detected
> - Verify that `dataset/classes.txt` contains the correct class names

### Permission Denied for Screen Capture

> [!NOTE]
> Your browser needs permission to capture screen content. When prompted:
> - Grant screen capture/sharing permissions
> - Select the correct window or tab
> - On macOS, you may need to grant permissions in System Preferences → Security & Privacy → Screen Recording

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> [!TIP]
> **Want to contribute?** Feel free to open issues or submit pull requests on GitHub!

**Made with ❤️ for the Arras.io community**
