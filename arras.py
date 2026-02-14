#!/usr/bin/env python3
"""
Arras.io YOLO Overlay Server

Usage:
    python arras.py

Opens http://localhost:7280 showing arras.io/#wpd with a YOLO detection overlay.
Click "Start YOLO" to begin tab capture and real-time object detection (~1fps).
"""

import http.server
import json
import base64
import os
import sys
import webbrowser
import threading
import numpy as np
import cv2
from ultralytics import YOLO

# --------------- Configuration ---------------
MODEL_PATH = "best.pt"
CLASSES_FILE = os.path.join("dataset", "classes.txt")
PORT = 7280
CONF_THRESHOLD = 0.2  # Detection confidence threshold (0.0 - 1.0)
FPS_CAP = 60  # Maximum detection FPS (frames per second)

# --------------- Load model & classes ---------------
print(f"Loading YOLO model ({MODEL_PATH})...")
model = YOLO(MODEL_PATH)

class_names = []
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE) as f:
        class_names = [line.strip() for line in f if line.strip()]
print(f"Classes: {class_names}")

# --------------- Serve HTML + detection API ---------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arras.io — YOLO Overlay</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { overflow: hidden; background: #111; font-family: 'Segoe UI', system-ui, sans-serif; }

  #game-frame {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    border: none; z-index: 1;
  }

  #overlay {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 10; pointer-events: none;
  }

  #controls {
    position: fixed; top: 12px; right: 12px;
    z-index: 100; pointer-events: auto;
    background: rgba(0,0,0,0.75); color: #eee;
    border-radius: 10px; padding: 12px 16px;
    min-width: 180px; backdrop-filter: blur(8px);
    font-size: 13px; user-select: none;
    border: 1px solid rgba(255,255,255,0.1);
  }
  #controls h3 { margin-bottom: 8px; font-size: 14px; color: #fff; }
  #controls .row { display: flex; align-items: center; justify-content: space-between; margin: 4px 0; }
  #controls label { color: #aaa; font-size: 12px; }
  #controls .val { color: #fff; font-size: 12px; font-variant-numeric: tabular-nums; }

  #start-btn {
    width: 100%; padding: 8px; margin-top: 8px;
    border: none; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 600;
    background: #22c55e; color: #fff;
    transition: background 0.2s;
  }
  #start-btn:hover { background: #16a34a; }
  #start-btn.running { background: #ef4444; }
  #start-btn.running:hover { background: #dc2626; }

  input[type=range] { width: 100%; margin: 4px 0; accent-color: #22c55e; }

  #legend { margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; margin: 2px 0; }
  .legend-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

  #iframe-fallback {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 2; display: none;
    justify-content: center; align-items: center;
    background: #111; color: #aaa; font-size: 18px;
    flex-direction: column; gap: 12px;
  }
  #iframe-fallback a { color: #22c55e; }
</style>
</head>
<body>

<iframe id="game-frame" src="https://arras.io/#wpd"
  sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
  allow="autoplay; fullscreen"></iframe>

<div id="iframe-fallback">
  <p>Could not embed arras.io in an iframe.</p>
  <p>Open <a href="https://arras.io/#wpd" target="_blank">arras.io/#wpd</a> in another tab, then click Start YOLO below and select that window.</p>
</div>

<canvas id="overlay"></canvas>

<div id="controls">
  <h3>YOLO Overlay</h3>
  <div class="row"><label>Status</label><span class="val" id="status">Idle</span></div>
  <div class="row"><label>Detections</label><span class="val" id="det-count">0</span></div>
  <div class="row"><label>Latency</label><span class="val" id="latency">—</span></div>
  <div class="row"><label>Confidence</label><span class="val" id="conf-val">__CONF__</span></div>
  <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="__CONF__">
  <button id="start-btn" onclick="toggleCapture()">Start YOLO</button>
  <div id="legend"></div>
</div>

<script>
// -------- Constants --------
// bullet, egg, hexagon, pentagon, player, square, triangle, wall
const CLASS_COLORS = [
  'rgb(255,38,0)',   // bullet
  'rgb(255,255,255)',// egg
  'rgb(48,242,229)', // hexagon
  'rgb(135,78,254)', // pentagon
  'rgb(0,249,1)',    // player
  'rgb(254,199,0)',  // square
  'rgb(255,147,0)',  // triangle
  'rgb(122,122,122)' // wall
];

// -------- State --------
let classNames = [];
let stream = null;
let isRunning = false;
let detecting = false;
let detections = [];
let detImgW = 640, detImgH = 480;
let confThreshold = __CONF__;
let fpsDelay = __FPS_DELAY__;
let detectInterval = null;

const videoEl  = document.createElement('video');
const capCanvas = document.createElement('canvas');
const capCtx   = capCanvas.getContext('2d');
const overlay  = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

// -------- Init --------
fetch('/classes').then(r => r.json()).then(names => {
  classNames = names;
  const legend = document.getElementById('legend');
  names.forEach((name, i) => {
    legend.innerHTML += `<div class="legend-item">
      <span class="legend-dot" style="background:${CLASS_COLORS[i % CLASS_COLORS.length]}"></span>
      ${name}</div>`;
  });
});

// Check if iframe loaded
const iframe = document.getElementById('game-frame');
iframe.addEventListener('load', () => {
  try { iframe.contentWindow.location.href; } catch(e) { /* cross-origin, that's fine */ }
});
setTimeout(() => {
  // If iframe is blank (blocked), show fallback message
  try {
    if (!iframe.contentWindow || iframe.contentWindow.length === 0) {
      // Might be blocked — show note but don't hide iframe (it might still work)
    }
  } catch(e) {}
}, 5000);

// -------- Confidence slider --------
document.getElementById('conf-slider').addEventListener('input', e => {
  confThreshold = parseFloat(e.target.value);
  document.getElementById('conf-val').textContent = confThreshold.toFixed(2);
});

// -------- Capture control --------
async function toggleCapture() {
  if (isRunning) {
    stopCapture();
  } else {
    await startCapture();
  }
}

async function startCapture() {
  try {
    // Check if getDisplayMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      alert('Screen capture not supported in this browser.\n\nPlease use Chrome, Edge, or Firefox.');
      document.getElementById('status').textContent = 'Not Supported';
      return;
    }

    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { 
        frameRate: { ideal: 30 },
        displaySurface: 'browser'
      },
      audio: false,
      preferCurrentTab: true,
      selfBrowserSurface: 'include',
      surfaceSwitching: 'include',
      systemAudio: 'exclude'
    });
  } catch (e) {
    console.error('Screen capture error:', e);
    let errorMsg = 'Screen capture failed: ' + e.name;
    
    if (e.name === 'NotAllowedError') {
      errorMsg = 'Permission denied. Click Start YOLO and select a tab/window to share.';
    } else if (e.name === 'NotSupportedError') {
      errorMsg = 'Screen capture not supported. Try Chrome or use HTTPS.';
    } else if (e.name === 'NotFoundError') {
      errorMsg = 'No screen/window available to capture.';
    } else if (e.name === 'AbortError') {
      errorMsg = 'Screen capture cancelled.';
    }
    
    document.getElementById('status').textContent = 'Error';
    alert(errorMsg + '\n\nError: ' + e.name + '\n' + (e.message || ''));
    return;
  }

  // Handle user stopping share via browser UI
  stream.getVideoTracks()[0].addEventListener('ended', stopCapture);

  videoEl.srcObject = stream;
  videoEl.muted = true;
  await videoEl.play();

  isRunning = true;
  const btn = document.getElementById('start-btn');
  btn.textContent = 'Stop YOLO';
  btn.classList.add('running');
  document.getElementById('status').textContent = 'Running';

  // Start detection loop (~1fps)
  detectLoop();
  // Start render loop (60fps)
  renderLoop();
}

function stopCapture() {
  isRunning = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  videoEl.srcObject = null;
  detections = [];
  clearTimeout(detectInterval);

  const btn = document.getElementById('start-btn');
  btn.textContent = 'Start YOLO';
  btn.classList.remove('running');
  document.getElementById('status').textContent = 'Idle';
  document.getElementById('det-count').textContent = '0';
  document.getElementById('latency').textContent = '—';

  // Clear overlay
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

// -------- Detection loop (1fps) --------
function detectLoop() {
  if (!isRunning) return;

  if (!detecting && videoEl.videoWidth > 0) {
    detecting = true;
    const t0 = performance.now();

    // Resize to max 640px wide
    const aspect = videoEl.videoHeight / videoEl.videoWidth;
    const sendW = Math.min(640, videoEl.videoWidth);
    const sendH = Math.round(sendW * aspect);
    capCanvas.width = sendW;
    capCanvas.height = sendH;
    capCtx.drawImage(videoEl, 0, 0, sendW, sendH);

    const dataUrl = capCanvas.toDataURL('image/jpeg', 0.7);

    fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, conf: confThreshold })
    })
    .then(r => r.json())
    .then(data => {
      detections = data.detections || [];
      detImgW = data.imgW || sendW;
      detImgH = data.imgH || sendH;
      document.getElementById('det-count').textContent = detections.length;
      document.getElementById('latency').textContent = Math.round(performance.now() - t0) + 'ms';
      detecting = false;
    })
    .catch(err => {
      console.error('Detection error:', err);
      detecting = false;
    });
  }

  detectInterval = setTimeout(detectLoop, fpsDelay);
}

// -------- Render loop (60fps) --------
function renderLoop() {
  if (!isRunning) return;

  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
  const ctx = overlayCtx;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (detections.length === 0) {
    requestAnimationFrame(renderLoop);
    return;
  }

  const scaleX = overlay.width / detImgW;
  const scaleY = overlay.height / detImgH;

  for (const det of detections) {
    if (det.conf < confThreshold) continue;

    const x1 = det.x1 * scaleX;
    const y1 = det.y1 * scaleY;
    const x2 = det.x2 * scaleX;
    const y2 = det.y2 * scaleY;
    const w = x2 - x1;
    const h = y2 - y1;
    const color = CLASS_COLORS[det.cls % CLASS_COLORS.length];

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    // Label background
    const label = `${det.name} ${(det.conf * 100).toFixed(0)}%`;
    ctx.font = 'bold 13px system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    const lh = 18;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x1, y1 - lh, tw + 8, lh);
    ctx.globalAlpha = 1.0;

    // Label text
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 4, y1 - 4);
  }

  requestAnimationFrame(renderLoop);
}
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = HTML_PAGE.replace("__CONF__", str(CONF_THRESHOLD))
            html = html.replace("__FPS_DELAY__", str(int(1000 / FPS_CAP)))
            self._send(200, "text/html", html.encode())
        elif self.path == "/classes":
            self._send(200, "application/json", json.dumps(class_names).encode())
        elif self.path in ("/favicon.ico", "/robots.txt", "/sitemap.xml"):
            # Silently ignore common browser requests
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/detect":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                data = json.loads(body)

                # Decode image
                img_b64 = data["image"]
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    self._send(400, "application/json", b'{"error":"bad image"}')
                    return

                conf = float(data.get("conf", CONF_THRESHOLD))

                # Run YOLO
                results = model(img, verbose=False, conf=conf)

                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        c = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = class_names[cls] if cls < len(class_names) else str(cls)
                        detections.append({
                            "x1": round(x1, 1), "y1": round(y1, 1),
                            "x2": round(x2, 1), "y2": round(y2, 1),
                            "conf": round(c, 3), "cls": cls, "name": name
                        })

                resp = json.dumps({
                    "detections": detections,
                    "imgW": img.shape[1],
                    "imgH": img.shape[0]
                })
                self._send(200, "application/json", resp.encode())

            except Exception as e:
                self._send(500, "application/json",
                           json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress SSL/TLS handshake errors and other noise
        if args and len(args) > 1:
            msg = str(args[1]) if len(args) > 1 else ""
            # Skip SSL/TLS related bad request errors
            if "Bad request version" in msg or "Bad request syntax" in msg:
                return
        # Only log actual errors (5xx)
        if args and str(args[0]).startswith("5"):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    server = http.server.HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  Arras.io YOLO Overlay")
    print(f"  {url}")
    print(f"  Model: {MODEL_PATH}  |  Classes: {class_names}")
    print(f"  Press Ctrl+C to stop\n")

    # Auto-open in Chrome after a short delay (cross-platform)
    def open_in_chrome():
        try:
            # Try common Chrome browser names (works on Windows, macOS, Linux)
            chrome = webbrowser.get('chrome') if hasattr(webbrowser, 'get') else None
            if chrome:
                chrome.open(url)
            else:
                # Fallback: try by name
                for name in ['google-chrome', 'chrome', 'chromium']:
                    try:
                        webbrowser.get(name).open(url)
                        return
                    except:
                        continue
                # If Chrome not found, use default browser
                webbrowser.open(url)
        except:
            webbrowser.open(url)
    
    threading.Timer(1.0, open_in_chrome).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
        sys.exit(0)
