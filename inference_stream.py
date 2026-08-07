import time
import json
import threading
import queue
from collections import deque
import argparse
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import sounddevice as sd
import torch
from config import TARGET_SAMPLE_RATE, STREAM_WINDOW, STREAM_HOP, STREAM_THRESHOLD, STREAM_SMOOTHING_ALPHA, STREAM_DEVICE
from features.feature_extraction import extract_log_mel
from preprocessing.audio_cleaning import normalize_audio, pad_or_crop
from inference import load_model
from utils import int_to_label, risk_from_probability
from config import TARGET_SAMPLE_RATE, MODEL_DIR, FEATURE_TYPE
SSE_CLIENTS = []
SSE_LOCK = threading.Lock()

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Live Spoof Detection</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .status { font-size: 1.2rem; margin-bottom: 1rem; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-width: 540px; }
    .label { font-size: 1.6rem; font-weight: bold; }
    .high { color: #b00; }
    .low { color: #080; }
    .info { margin-top: 8px; }
    pre { background: #f5f5f5; padding: 12px; border-radius: 8px; overflow: auto; }
  </style>
</head>
<body>
  <div class="card">
    <div class="status">Live stream connected. Waiting for events...</div>
    <button id="recordButton">Record & Analyze 4s</button>
    <div class="label" id="label">-</div>
    <div class="info"><strong>AI probability:</strong> <span id="prob">-</span></div>
    <div class="info"><strong>Risk:</strong> <span id="risk">-</span></div>
    <div class="info"><strong>Raw JSON:</strong></div>
    <pre id="json">{}</pre>
  </div>
  <script>
    const eventSource = new EventSource('/events');
    const labelEl = document.getElementById('label');
    const probEl = document.getElementById('prob');
    const riskEl = document.getElementById('risk');
    const jsonEl = document.getElementById('json');
    const statusEl = document.querySelector('.status');
    const recordButton = document.getElementById('recordButton');

    eventSource.onopen = () => {
      statusEl.textContent = 'Connected to stream.';
    };
    eventSource.onerror = () => {
      statusEl.textContent = 'Connection lost; retrying...';
    };
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      statusEl.textContent = 'Last update: ' + new Date().toLocaleTimeString();
      labelEl.textContent = data.label;
      labelEl.className = data.label === 'AI-generated' ? 'label high' : 'label low';
      probEl.textContent = (data.ai_probability * 100).toFixed(2) + '%';
      riskEl.textContent = data.risk;
      jsonEl.textContent = JSON.stringify(data, null, 2);
    };

    recordButton.addEventListener('click', async () => {
      recordButton.disabled = true;
      recordButton.textContent = 'Processing...';
      statusEl.textContent = 'Recording and analyzing...';
      try {
        const response = await fetch('/record', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
          statusEl.textContent = 'Recorded result received.';
          labelEl.textContent = data.label;
          labelEl.className = data.label === 'AI-generated' ? 'label high' : 'label low';
          probEl.textContent = (data.ai_probability * 100).toFixed(2) + '%';
          riskEl.textContent = data.risk;
          jsonEl.textContent = JSON.stringify(data, null, 2);
        } else {
          statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
        }
      } catch (err) {
        statusEl.textContent = 'Request failed: ' + err.message;
      } finally {
        recordButton.disabled = false;
        recordButton.textContent = 'Record & Analyze 4s';
      }
    });
  </script>
</body>
</html>
"""


def process_window(model, device, window_audio, feature_type="log_mel"):
    # normalize / pad to expected length
    audio = normalize_audio(window_audio)
    audio = pad_or_crop(audio, target_length=int(TARGET_SAMPLE_RATE * STREAM_WINDOW))
    if feature_type == "mfcc":
        # fallback: use log_mel extractor if mfcc not supported in streaming yet
        features = extract_log_mel(audio, sample_rate=TARGET_SAMPLE_RATE)
    else:
        features = extract_log_mel(audio, sample_rate=TARGET_SAMPLE_RATE)
    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return float(probs[1]), {"human": float(probs[0]), "ai_generated": float(probs[1])}


def run_stream(model_path: str, feature: str = "log_mel", window: float = STREAM_WINDOW, hop: float = STREAM_HOP,
               threshold: float = STREAM_THRESHOLD, smoothing_alpha: float = STREAM_SMOOTHING_ALPHA,
               device_override=None, socket_host: str = None, socket_port: int = None,
               web_host: str = None, web_port: int = None):
    # load model
    model, device = load_model(model_path, device=device_override or STREAM_DEVICE)

    sr = TARGET_SAMPLE_RATE
    window_samples = int(window * sr)

    buffer = deque(maxlen=window_samples * 2)
    buffer_lock = threading.Lock()
    stop_event = threading.Event()
    ema_score = None
    client_conn = None

    # simple TCP socket server to forward JSON events
    def socket_server_thread(host, port):
        nonlocal client_conn
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(1.0)
        while not stop_event.is_set():
            try:
                conn, addr = srv.accept()
                client_conn = conn
                # keep connection until closed
                while not stop_event.is_set():
                    time.sleep(0.1)
                    if conn.fileno() == -1:
                        break
                try:
                    conn.close()
                except Exception:
                    pass
                client_conn = None
            except socket.timeout:
                continue
        try:
            srv.close()
        except Exception:
            pass

    socket_thread = None
    if socket_host and socket_port:
        socket_thread = threading.Thread(target=socket_server_thread, args=(socket_host, socket_port), daemon=True)
        socket_thread.start()

    def record_result():
        if len(buffer) < window_samples:
            return {"error": "Not enough audio captured yet"}
        with buffer_lock:
            arr = np.array(buffer, dtype=np.float32)
            if arr.size < window_samples:
                arr = arr[-window_samples:]
            else:
                arr = arr[-window_samples:]
        ai_prob, probs = process_window(model, device, arr, feature)
        score = float(ai_prob)
        label = int_to_label(1 if score >= threshold else 0)
        return {
            "timestamp": time.time(),
            "label": label,
            "ai_probability": score,
            "risk": risk_from_probability(score),
            "probs": probs,
        }

    http_server = None
    if web_host and web_port:
        class SSEHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(HTML_PAGE.encode('utf-8'))))
                    self.end_headers()
                    self.wfile.write(HTML_PAGE.encode('utf-8'))
                elif self.path == '/events':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    with SSE_LOCK:
                        SSE_CLIENTS.append(self.wfile)
                    try:
                        while not stop_event.is_set():
                            time.sleep(0.5)
                    except Exception:
                        pass
                    with SSE_LOCK:
                        if self.wfile in SSE_CLIENTS:
                            SSE_CLIENTS.remove(self.wfile)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/record':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    result = self.server.record_result()
                    self.wfile.write(json.dumps(result).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

        class SSEHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True

        http_server = SSEHTTPServer((web_host, web_port), SSEHandler)
        http_server.daemon_threads = True
        http_server.record_result = record_result
        try:
            http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
            http_thread.start()
            print(json.dumps({"event": "web_ui_started", "url": f"http://{web_host}:{web_port}/"}))
        except Exception as e:
            print(f"Failed to start web UI server on {web_host}:{web_port}: {e}")
            http_server = None

    def audio_callback(indata, frames, time_info, status):
        # indata shape (frames, channels)
        if status:
            print(f"InputStream status: {status}")
        samples = indata[:, 0]
        with buffer_lock:
            buffer.extend(samples.tolist())

    # start input stream
    try:
        stream = sd.InputStream(samplerate=sr, channels=1, callback=audio_callback)
        stream.start()
    except Exception as e:
        print("Failed to open input stream:", e)
        return

    print(json.dumps({"event": "stream_started", "window": window, "hop": hop}))

    try:
        while not stop_event.is_set():
            time.sleep(hop)
            if len(buffer) < window_samples:
                # not enough samples yet
                continue
            # get last window_samples
            with buffer_lock:
                arr = np.array(buffer, dtype=np.float32)
                if arr.size < window_samples:
                    arr = arr[-window_samples:]
                else:
                    arr = arr[-window_samples:]

            ai_prob, probs = process_window(model, device, arr, feature)
            if ema_score is None:
                ema_score = ai_prob
            else:
                ema_score = smoothing_alpha * ai_prob + (1 - smoothing_alpha) * ema_score

            label = int_to_label(1 if ema_score >= threshold else 0)
            risk = risk_from_probability(float(ema_score))
            out = {
                "timestamp": time.time(),
                "label": label,
                "ai_probability": float(ema_score),
                "risk": risk,
                "probs": probs,
            }
            text = json.dumps(out)
            print(text)
            # forward to connected socket client if any
            if client_conn:
                try:
                    client_conn.sendall((text + "\n").encode("utf-8"))
                except Exception:
                    try:
                        client_conn.close()
                    except Exception:
                        pass
                    client_conn = None
            # forward to SSE web clients
            with SSE_LOCK:
                for client in SSE_CLIENTS[:]:
                    try:
                        client.write(f"data: {text}\n\n".encode("utf-8"))
                        client.flush()
                    except Exception:
                        SSE_CLIENTS.remove(client)
    except KeyboardInterrupt:
        print(json.dumps({"event": "stopped_by_user"}))
    finally:
        stop_event.set()
        if socket_thread:
            socket_thread.join(timeout=1.0)
        if http_server:
            http_server.shutdown()
            http_server.server_close()
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Stream microphone and run real-time spoof detection.")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--feature", type=str, default="log_mel", choices=["log_mel", "mfcc"])
    parser.add_argument("--window", type=float, default=STREAM_WINDOW)
    parser.add_argument("--hop", type=float, default=STREAM_HOP)
    parser.add_argument("--threshold", type=float, default=STREAM_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=STREAM_SMOOTHING_ALPHA)
    parser.add_argument("--device", type=str, default=None, help="mps/cuda/cpu or leave empty for auto")
    parser.add_argument("--socket-host", type=str, default=None, help="Local socket host to serve JSON events")
    parser.add_argument("--socket-port", type=int, default=None, help="Local socket port to serve JSON events")
    parser.add_argument("--web-host", type=str, default=None, help="Local host for the web UI")
    parser.add_argument("--web-port", type=int, default=None, help="Local port for the web UI")
    args = parser.parse_args()

    run_stream(model_path=args.model, feature=args.feature, window=args.window, hop=args.hop,
               threshold=args.threshold, smoothing_alpha=args.alpha, device_override=args.device,
               socket_host=args.socket_host, socket_port=args.socket_port,
               web_host=args.web_host, web_port=args.web_port)


if __name__ == "__main__":
    main()
