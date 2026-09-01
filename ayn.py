import os
import cv2
import threading
import time
import urllib.parse
import numpy as np
from datetime import datetime
from collections import deque

from flask import Flask, Response, render_template_string, request, jsonify
import pymysql
from ultralytics import YOLO

app = Flask(__name__)

# ==================== CONFIGURATION ====================
DB_CONFIG = {
    'host': '92.113.22.7',
    'database': 'u451829952_ayna',
    'user': 'u451829952_ayna',
    'password': 'Flex@1984#',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# ==================== GENDER MODEL ====================
GENDER_NET = None
GENDER_LIST = ['Qadin', 'Kisi']


def load_gender_model():
    global GENDER_NET
    try:
        GENDER_NET = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
        print("[OK] Gender model loaded successfully.")
    except Exception as e:
        print(f"[Info] Gender model not found, falling back to heuristic: {e}")


load_gender_model()


def predict_gender(face_img):
    if GENDER_NET is not None:
        try:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            GENDER_NET.setInput(blob)
            preds = GENDER_NET.forward()
            return GENDER_LIST[preds[0].argmax()]
        except Exception:
            pass

    h, w = face_img.shape[:2]
    ratio = w / h if h > 0 else 0
    if ratio > 1.2:
        return 'Qadin'
    return 'Qadin' if ratio < 0.9 else 'Kisi'


# ==================== AGE MODEL ====================
AGE_NET = None
AGE_RANGES = [
    '(0-2)', '(4-6)', '(8-12)',       # -> usaq
    '(15-20)', '(25-32)', '(38-43)',  # -> cavan
    '(48-53)', '(60-100)'             # -> qoca
]


def load_age_model():
    global AGE_NET
    try:
        AGE_NET = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
        print("[OK] Age model loaded successfully.")
    except Exception as e:
        print(f"[Info] Age model not found, age will default to 'cavan': {e}")


load_age_model()


def predict_age(face_img):
    """Return (age_label, age_cat) where age_cat is 'usaq' | 'cavan' | 'qoca'."""
    if AGE_NET is not None:
        try:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            AGE_NET.setInput(blob)
            preds = AGE_NET.forward()
            idx = int(preds[0].argmax())
            label = AGE_RANGES[idx] if 0 <= idx < len(AGE_RANGES) else 'Namelum'
            if idx <= 2:
                cat = 'usaq'
            elif idx <= 5:
                cat = 'cavan'
            else:
                cat = 'qoca'
            return label, cat
        except Exception:
            pass

    return 'Namelum', 'cavan'


# ==================== DATABASE ====================
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def save_visitor_to_db(station_id, gender, age_cat=None, direction='giris'):
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            now = datetime.now()
            saat = now.strftime('%H:%M:%S')

            if direction == 'giris':
                kisi = 1 if gender == 'Kisi' else 0
                qadin = 1 if gender == 'Qadin' else 0
                usaq = 1 if age_cat == 'usaq' else 0
                cavan = 1 if age_cat == 'cavan' else 0
                qoca = 1 if age_cat == 'qoca' else 0
                gunluk, giris, cixis = 1, 1, 0
            else:  # cixis
                kisi, qadin, usaq, cavan, qoca = 0, 0, 0, 0, 0
                gunluk, giris, cixis = 0, 0, 1

            sql = """
                INSERT INTO visitor_logs
                (ayna_dayanacaq_id, kisi, qadin, usaq, cavan, qoca, saat,
                 gunluk_girisler_say, giris_sayi, cixis_sayi, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(sql, (station_id, kisi, qadin, usaq, cavan, qoca,
                                  saat, gunluk, giris, cixis))
            conn.commit()
            print(f"[DB OK] {direction.upper()} - Gender: {gender} - Age: {age_cat} - Station: {station_id}")
        conn.close()
    except Exception as e:
        print(f"[DB Error] {e}")


# ==================== TRACKING ====================
class PersonTrack:
    def __init__(self, track_id, bbox, gender, age_label='Namelum', age_cat='cavan'):
        self.id = track_id
        self.bbox = bbox
        self.centroid = self._calc_centroid(bbox)
        self.gender = gender
        self.age_label = age_label
        self.age_cat = age_cat
        self.age = 0
        self.missed = 0
        self.history = deque(maxlen=15)
        self.history.append(self.centroid)
        self.confirmed = False
        self.gender_fixed = False
        self.db_entry_written = False
        self.db_exit_written = False

    def _calc_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox, gender=None):
        self.bbox = bbox
        self.centroid = self._calc_centroid(bbox)
        self.history.append(self.centroid)
        self.age += 1
        self.missed = 0
        if gender and gender != 'Namelum' and not self.gender_fixed:
            self.gender = gender
            self.gender_fixed = True
        if self.age > 5:
            self.confirmed = True

    def mark_missed(self):
        self.missed += 1

    def get_direction(self, line_y):
        if len(self.history) < 5:
            return None
        prev_y = self.history[-5][1]
        curr_y = self.history[-1][1]
        if abs(curr_y - prev_y) < 4:
            return None
        if prev_y < line_y and curr_y >= line_y:
            return 'in'
        elif prev_y > line_y and curr_y <= line_y:
            return 'out'
        return None


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ==================== STATION WORKER ====================
class StationWorker:
    def __init__(self, station_id, station_name, source):
        self.station_id = station_id
        self.station_name = station_name
        self.source = source

        self.model = YOLO('yolo11s.pt')

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_detections = []
        self.tracks_dict = {}
        self.det_lock = threading.Lock()

        self.entry_count = 0
        self.exit_count = 0
        self.stats_lock = threading.Lock()

        self.capture_running = False
        self.inference_running = False

        self._capture_thread = None
        self._inference_thread = None

    def is_alive(self):
        return self.capture_running or self.inference_running

    def start(self):
        if self.is_alive():
            return
        self.capture_running = True
        self.inference_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self):
        self.capture_running = False
        self.inference_running = False

    # ---------------- capture ----------------
    def _capture_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        def open_cap(src):
            if isinstance(src, str) and src.startswith('rtsp'):
                c = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                c = cv2.VideoCapture(src)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c

        cap = open_cap(self.source)

        while self.capture_running:
            if not cap.isOpened():
                time.sleep(2)
                cap = open_cap(self.source)
                continue

            ret, frame = cap.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
                time.sleep(0.01)
            else:
                time.sleep(0.02)

        cap.release()

    # ---------------- inference ----------------
    def _inference_loop(self):
        next_track_id = 1
        local_tracks = {}

        while self.inference_running:
            start_time = time.time()

            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            height, width = frame.shape[:2]

            roi_polygon = np.array([
                [int(width * 0.28), int(height * 0.98)],
                [int(width * 0.38), int(height * 0.25)],
                [int(width * 0.65), int(height * 0.25)],
                [int(width * 0.95), int(height * 0.98)]
            ], np.int32)

            line_y = int(height * 0.60)

            results = self.model(frame, conf=0.40, iou=0.45, classes=[0], verbose=False, imgsz=1080)

            detections = []
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int(y2 - (y2 - y1) * 0.3)
                    is_inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)
                    if is_inside >= 0:
                        detections.append((x1, y1, x2, y2))

            matched_tracks = set()
            matched_dets = set()

            if detections:
                for det_idx, (x1, y1, x2, y2) in enumerate(detections):
                    best_iou = 0.25
                    best_track = None
                    for tid, track in local_tracks.items():
                        if tid in matched_tracks:
                            continue
                        iou = compute_iou(track.bbox, (x1, y1, x2, y2))
                        if iou > best_iou:
                            best_iou = iou
                            best_track = tid

                    if best_track is not None:
                        local_tracks[best_track].update((x1, y1, x2, y2))
                        matched_tracks.add(best_track)
                        matched_dets.add(det_idx)
                    else:
                        face_img = frame[y1:int(y1 + (y2 - y1) * 0.4), x1:x2]
                        if face_img.size > 0:
                            gender = predict_gender(face_img)
                            age_label, age_cat = predict_age(face_img)
                        else:
                            gender = 'Qadin'
                            age_label, age_cat = 'Namelum', 'cavan'
                        local_tracks[next_track_id] = PersonTrack(
                            next_track_id, (x1, y1, x2, y2), gender, age_label, age_cat
                        )
                        next_track_id += 1

            to_remove = []
            for tid, track in local_tracks.items():
                if tid not in matched_tracks:
                    track.mark_missed()
                    if track.missed > 30:
                        to_remove.append(tid)
            for tid in to_remove:
                del local_tracks[tid]

            with self.stats_lock:
                for tid, track in local_tracks.items():
                    if track.confirmed and not track.db_entry_written:
                        track.db_entry_written = True
                        self.entry_count += 1
                        save_visitor_to_db(self.station_id, track.gender, track.age_cat, 'giris')

                    if track.confirmed:
                        direction = track.get_direction(line_y)
                        if direction == 'out' and not track.db_exit_written:
                            track.db_exit_written = True
                            self.exit_count += 1
                            save_visitor_to_db(self.station_id, track.gender, track.age_cat, 'cixis')

            with self.det_lock:
                self.latest_detections = [
                    (t.bbox, t.gender, t.id, t.db_entry_written, t.missed, t.age_label, t.age_cat)
                    for t in local_tracks.values()
                ]
                self.tracks_dict = {
                    'roi': roi_polygon,
                    'line_y': line_y,
                    'width': width,
                    'height': height
                }

            elapsed = time.time() - start_time
            time.sleep(max(0, 0.05 - elapsed))

    # ---------------- stats ----------------
    def get_stats(self):
        with self.stats_lock:
            with self.det_lock:
                active = [d for d in self.latest_detections if d[4] < 5]
                inside = len(active)
                male = len([d for d in active if d[1] == 'Kisi'])
                female = len([d for d in active if d[1] == 'Qadin'])
                usaq = len([d for d in active if d[6] == 'usaq'])
                cavan = len([d for d in active if d[6] == 'cavan'])
                qoca = len([d for d in active if d[6] == 'qoca'])
            return {
                'entry': self.entry_count,
                'exit': self.exit_count,
                'inside': inside,
                'male': male,
                'female': female,
                'usaq': usaq,
                'cavan': cavan,
                'qoca': qoca
            }

    # ---------------- video output ----------------
    def generate_frames(self):
        while self.capture_running or self.inference_running:
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_frame.copy()

            with self.det_lock:
                detections = list(self.latest_detections)
                roi = self.tracks_dict.get('roi')
                line_y = self.tracks_dict.get('line_y')

            if roi is not None:
                cv2.polylines(frame, [roi], isClosed=True, color=(0, 0, 255), thickness=2)
            if line_y is not None:
                h, w = frame.shape[:2]
                cv2.line(frame, (int(w * 0.33), line_y), (int(w * 0.80), line_y), (0, 255, 0), 2)

            for bbox, gender, tid, crossed, missed, age_label, age_cat in detections:
                if missed > 3:
                    continue
                x1, y1, x2, y2 = bbox
                if gender == 'Kisi':
                    color = (255, 150, 50)
                elif gender == 'Qadin':
                    color = (200, 50, 255)
                else:
                    color = (200, 200, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {gender} {age_label}"
                if crossed:
                    label += " OK"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

            male = len([d for d in detections if d[4] < 5 and d[1] == 'Kisi'])
            female = len([d for d in detections if d[4] < 5 and d[1] == 'Qadin'])
            with self.stats_lock:
                info = (f"Giris:{self.entry_count} Cixis:{self.exit_count} "
                        f"Kisi:{male} Qadin:{female} Track:{len(detections)}")
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.033)


# ==================== CABIN MANAGER ====================
stations_lock = threading.Lock()
active_stations = {}


def build_source(station):
    ip = station.get('ip_address')
    port = station.get('port') if station.get('port') else '554'
    login = station.get('login')
    passw = station.get('pass')

    if ip and str(ip).isdigit():
        return int(ip)
    elif ip:
        enc_login = urllib.parse.quote(str(login)) if login else ''
        enc_passw = urllib.parse.quote(str(passw)) if passw else ''
        if enc_login and enc_passw:
            return f"rtsp://{enc_login}:{enc_passw}@{ip}:{port}/Streaming/Channels/101"
        else:
            return f"rtsp://{ip}:{port}/live"
    else:
        return 0


# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    stations = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, station_name, unvan FROM stations")
            stations = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"[DB Error] {e}")

    with stations_lock:
        running_ids = [sid for sid, w in active_stations.items() if w.is_alive()]

    html_page = """
    <!DOCTYPE html>
    <html lang="az">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ayna - Canli Kamera Sistemi</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%); color: #fff; min-height: 100vh; }
            .select-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                           border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
                           padding: 30px; margin: 0 auto 30px auto; }
            .video-container { border-radius: 12px; overflow: hidden;
                               box-shadow: 0 0 40px rgba(0,123,255,0.2);
                               border: 1px solid rgba(255,255,255,0.1); }
            .stats-bar { background: rgba(0,0,0,0.5); border-radius: 12px; padding: 10px;
                         backdrop-filter: blur(10px);
                         border: 1px solid rgba(255,255,255,0.05); margin-bottom: 10px; }
            .stat-box   { text-align: center; padding: 6px; flex: 1; }
            .stat-number{ font-size: 1.4rem; font-weight: bold; }
            .stat-label { font-size: 0.75rem; opacity: 0.7; }
            .station-badge { background: rgba(0,123,255,0.2);
                             border: 1px solid rgba(0,123,255,0.4);
                             color: #6eb5ff; padding: 6px 16px;
                             border-radius: 20px; display: inline-block; font-weight: 600; }
            .station-row { display:flex; align-items:center; justify-content:space-between;
                           background: rgba(255,255,255,0.04); border-radius: 10px;
                           padding: 12px 16px; margin-bottom: 10px; }
        </style>
    </head>
    <body>
    <div class="container py-4">
        <h1 class="mb-1 text-center">&#127970; Ayna Kamera Sistemi</h1>
        <p class="text-muted mb-4 text-center">YOLO11 AI ilə canlı adam sayma və analiz — bir neçə kabin eyni anda</p>

        <div class="select-card" style="max-width:700px;">
            <h5 class="mb-3">Kabinlər <span class="text-muted small fw-normal">(bir neçəsini seçib birlikdə başlada bilərsiniz)</span></h5>
            <div id="stationList">
                {% for s in stations %}
                <div class="station-row" id="row-{{ s['id'] }}">
                    <div class="form-check d-flex align-items-center gap-2 mb-0">
                        <input class="form-check-input mt-0" type="checkbox"
                               id="chk-{{ s['id'] }}"
                               {% if s['id'] in running_ids %}checked disabled{% endif %}>
                        <label class="form-check-label" for="chk-{{ s['id'] }}">
                            <div class="fw-bold">{{ s['station_name'] }}</div>
                            {% if s['unvan'] %}<div class="text-muted small">{{ s['unvan'] }}</div>{% endif %}
                        </label>
                    </div>
                    {% if s['id'] in running_ids %}
                    <button class="btn btn-sm btn-outline-danger stop-btn" onclick="stopStation({{ s['id'] }})">Dayandır</button>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            <button class="btn btn-primary w-100 mt-3" id="startSelectedBtn" onclick="startSelected()">
                Seçilmiş kabinləri başlat
            </button>
            <div id="startError" class="text-danger mt-2 d-none small"></div>
        </div>

        <div class="row" id="camerasGrid"></div>
    </div>

    <script>
    const runningAtLoad = {{ running_ids|tojson }};

    function startSelected() {
        const startBtn = document.getElementById('startSelectedBtn');
        const errBox = document.getElementById('startError');
        errBox.classList.add('d-none');

        const checked = Array.from(
            document.querySelectorAll('#stationList input[type=checkbox]:checked:not(:disabled)')
        );
        if (checked.length === 0) {
            errBox.textContent = 'Ən azı bir kabin seçin.';
            errBox.classList.remove('d-none');
            return;
        }

        startBtn.disabled = true;
        startBtn.textContent = 'Başladılır...';

        const jobs = checked.map(chk => {
            const id = chk.id.replace('chk-', '');
            const row = document.getElementById('row-' + id);
            const name = row.querySelector('.fw-bold').textContent;
            chk.disabled = true;
            return fetch('/select_station', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'station_id=' + encodeURIComponent(id)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    addPanel(id, data.station_name);
                    markRunning(id, row);
                } else {
                    chk.disabled = false;
                    chk.checked = false;
                    errBox.textContent = (name + ': ' + (data.error || 'Xəta baş verdi'));
                    errBox.classList.remove('d-none');
                }
            })
            .catch(() => {
                chk.disabled = false;
                errBox.textContent = name + ': şəbəkə xətası';
                errBox.classList.remove('d-none');
            });
        });

        Promise.all(jobs).finally(() => {
            startBtn.disabled = false;
            startBtn.textContent = 'Seçilmiş kabinləri başlat';
        });
    }

    function markRunning(id, row) {
        row = row || document.getElementById('row-' + id);
        if (!row || row.querySelector('.stop-btn')) return;
        const stopBtn = document.createElement('button');
        stopBtn.className = 'btn btn-sm btn-outline-danger stop-btn';
        stopBtn.textContent = 'Dayandır';
        stopBtn.onclick = () => stopStation(id);
        row.appendChild(stopBtn);
    }

    function stopStation(id) {
        fetch('/stop_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'station_id=' + encodeURIComponent(id)
        })
        .then(r => r.json())
        .then(() => {
            removePanel(id);
            const row = document.getElementById('row-' + id);
            if (row) {
                const chk = row.querySelector('input[type=checkbox]');
                if (chk) { chk.checked = false; chk.disabled = false; }
                const stopBtn = row.querySelector('.stop-btn');
                if (stopBtn) stopBtn.remove();
            }
        });
    }

    function addPanel(id, name) {
        if (document.getElementById('panel-' + id)) return;
        const grid = document.getElementById('camerasGrid');
        const col = document.createElement('div');
        col.className = 'col-lg-6 mb-4';
        col.id = 'panel-' + id;
        col.innerHTML = `
            <div class="station-badge mb-2">${name}</div>
            <div class="video-container mb-2">
                <img src="/video_feed/${id}?${Date.now()}" class="img-fluid" style="width:100%;">
            </div>
            <div class="stats-bar d-flex justify-content-around flex-wrap">
                <div class="stat-box"><div class="stat-number text-success" id="entry-${id}">0</div><div class="stat-label">Giriş</div></div>
                <div class="stat-box"><div class="stat-number text-danger" id="exit-${id}">0</div><div class="stat-label">Çıxış</div></div>
                <div class="stat-box"><div class="stat-number text-warning" id="inside-${id}">0</div><div class="stat-label">Kabində</div></div>
                <div class="stat-box"><div class="stat-number text-info" id="male-${id}">0</div><div class="stat-label">Kişi</div></div>
                <div class="stat-box"><div class="stat-number" style="color:#ff69b4" id="female-${id}">0</div><div class="stat-label">Qadın</div></div>
            </div>`;
        grid.appendChild(col);
        startStatsPolling(id);
    }

    function removePanel(id) {
        const el = document.getElementById('panel-' + id);
        if (el) el.remove();
        stopStatsPolling(id);
    }

    const pollTimers = {};
    function startStatsPolling(id) {
        stopStatsPolling(id);
        pollTimers[id] = setInterval(() => {
            fetch('/stats/' + id).then(r => r.json()).then(d => {
                const set = (elId, val) => { const el = document.getElementById(elId); if (el) el.textContent = val; };
                set('entry-' + id, d.entry);
                set('exit-' + id, d.exit);
                set('inside-' + id, d.inside);
                set('male-' + id, d.male);
                set('female-' + id, d.female);
            }).catch(() => {});
        }, 1000);
    }
    function stopStatsPolling(id) {
        if (pollTimers[id]) { clearInterval(pollTimers[id]); delete pollTimers[id]; }
    }

    document.querySelectorAll('.station-row').forEach(row => {
        const id = row.id.replace('row-', '');
        if (runningAtLoad.includes(parseInt(id))) {
            const name = row.querySelector('.fw-bold').textContent;
            addPanel(id, name);
        }
    });
    </script>
    </body>
    </html>
    """
    return render_template_string(html_page, stations=stations, running_ids=running_ids)


@app.route('/select_station', methods=['POST'])
def select_station():
    station_id = request.form.get('station_id')
    if not station_id:
        return jsonify({'success': False, 'error': 'Filial seçilməyib'})

    try:
        station_id = int(station_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Yanlış filial ID'})

    with stations_lock:
        existing = active_stations.get(station_id)
        if existing and existing.is_alive():
            return jsonify({'success': True, 'station_name': existing.station_name})

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM stations WHERE id = %s", (station_id,))
            station = cursor.fetchone()
        conn.close()

        if not station:
            return jsonify({'success': False, 'error': 'Filial tapılmadı'})

        source = build_source(station)
        station_name = station.get('station_name', 'Naməlum Filial')

        worker = StationWorker(station_id, station_name, source)
        worker.start()

        with stations_lock:
            active_stations[station_id] = worker

        return jsonify({'success': True, 'station_name': station_name})
    except Exception as e:
        print(f"[Error] {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    station_id = request.form.get('station_id')
    if not station_id:
        return jsonify({'success': False, 'error': 'station_id tələb olunur'})
    try:
        station_id = int(station_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Yanlış filial ID'})

    with stations_lock:
        worker = active_stations.pop(station_id, None)
    if worker:
        worker.stop()
    return jsonify({'success': True})


@app.route('/stats/<int:station_id>')
def stats(station_id):
    with stations_lock:
        worker = active_stations.get(station_id)
    if not worker:
        return jsonify({'error': 'not running'}), 404
    return jsonify(worker.get_stats())


@app.route('/video_feed/<int:station_id>')
def video_feed(station_id):
    with stations_lock:
        worker = active_stations.get(station_id)
    if not worker:
        return Response(status=404)
    return Response(worker.generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


# ==================== AUTOSTART ALL CABINS ====================
def autostart_all_cabins():
    """Fetches all stations from the database and starts YOLO inference instantly on script launch."""
    print("[Info] Auto-starting all stations in the background...")
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM stations")
            stations = cursor.fetchall()
        conn.close()

        with stations_lock:
            for station in stations:
                station_id = station['id']
                if station_id not in active_stations:
                    source = build_source(station)
                    station_name = station.get('station_name', f'Naməlum Filial {station_id}')
                    print(f"[Info] Starting background worker for {station_name}...")
                    worker = StationWorker(station_id, station_name, source)
                    worker.start()
                    active_stations[station_id] = worker
                    
    except Exception as e:
        print(f"[Error] Failed to auto-start stations: {e}")


# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    autostart_all_cabins()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)