import eventlet
eventlet.monkey_patch() 

import base64
import io
import random
import time
import logging
import json
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image, ImageOps
from ultralytics import YOLO

# --- CONFIG ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'NEON_PULSE_NO_REPEATS'

socketio = SocketIO(app, 
                    async_mode='eventlet', 
                    cors_allowed_origins='*',
                    max_http_buffer_size=1e8,
                    ping_timeout=60,
                    ping_interval=25)

print("\n[SYSTEM] LOADING AI MODEL...")
model = YOLO("yolov8n.pt") 
print("[SYSTEM] AI READY.\n")

# --- DATABASE ---
DB_FILE = "user_data.json"

def save_database():
    try:
        export_data = {}
        for sid, p_data in PLAYERS.items():
            export_data[p_data['name']] = {
                'score': p_data['score'],
                'last_seen': time.ctime()
            }
        with open(DB_FILE, 'w') as f:
            json.dump(export_data, f, indent=4)
    except Exception as e:
        print(f"[DB ERROR] {e}")

# --- GAME STATE ---
GAME_STATE = {
    "status": "IDLE",
    "round": 0,
    "max_rounds": 10,
    "duration": 25
}

PLAYERS = {}

# --- CLEANED ITEM LIST (No rare items) ---
TARGET_MANIFEST = [
    "bottle", "cup", "keyboard", "mouse", "cell phone", "laptop", 
    "remote", "scissors", "book", "backpack", "spoon", "fork", 
    "chair", "banana", "apple", "sandwich", "orange", "bowl"
]

@app.route('/')
def index():
    return render_template('index.html')

# --- SOCKETS ---

@socketio.on('connect')
def on_connect():
    print(f"[NET] CONNECTED: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    if sid in PLAYERS:
        del PLAYERS[sid]
        broadcast_lobby_state()

@socketio.on('player_join')
def handle_join(data):
    sid = request.sid
    name = data.get('name', 'UNKNOWN')[:10].upper()
    colors = ['#FF0055', '#00FF41', '#00E5FF', '#FFFF00', '#BD00FF']
    
    PLAYERS[sid] = {
        'name': name, 
        'score': 0, 
        'color': random.choice(colors),
        'target': None,
        'has_scored': False,
        'target_queue': [] # Queue for this specific game
    }
    save_database()
    broadcast_lobby_state()

@socketio.on('player_exit')
def handle_exit():
    sid = request.sid
    if sid in PLAYERS:
        del PLAYERS[sid]
        emit('force_disconnect') 
        broadcast_lobby_state()

@socketio.on('request_start')
def handle_start(data):
    if GAME_STATE['status'] == "IDLE":
        try:
            duration = int(data.get('duration', 25))
            GAME_STATE['duration'] = max(5, min(duration, 90))
        except:
            GAME_STATE['duration'] = 25
        
        # --- PREPARE NO-REPEAT LISTS ---
        # For every player, give them a unique shuffled list of items
        for pid in PLAYERS:
            # Create a copy of the list and shuffle it
            deck = list(TARGET_MANIFEST)
            random.shuffle(deck)
            PLAYERS[pid]['target_queue'] = deck

        socketio.start_background_task(game_engine_loop)

@socketio.on('request_lobby_return')
def handle_return():
    GAME_STATE['status'] = "IDLE"
    GAME_STATE['round'] = 0
    for p in PLAYERS:
        PLAYERS[p]['score'] = 0
        PLAYERS[p]['has_scored'] = False
        PLAYERS[p]['target_queue'] = []
    socketio.emit('return_to_lobby')
    broadcast_lobby_state()

@socketio.on('image_submission')
def process_image(data):
    if GAME_STATE['status'] != "ACTIVE": return
    sid = request.sid
    if sid not in PLAYERS: return
    
    if PLAYERS[sid]['has_scored']:
        emit('upload_ack', {'success': False, 'msg': "ALREADY SCORED!"}, to=sid)
        return

    try:
        header, encoded = data['image'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)

        # AI Check
        results = model(img, conf=0.25, verbose=False)
        
        found_classes = []
        target = PLAYERS[sid]['target'] 
        target_found = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                found_classes.append(label)
                if label == target:
                    target_found = True

        unique_found = list(set(found_classes))

        if target_found:
            PLAYERS[sid]['score'] += 100
            PLAYERS[sid]['has_scored'] = True
            save_database()
            
            emit('upload_ack', {'success': True, 'msg': f"CORRECT! FOUND {target.upper()}"}, to=sid)
            socketio.emit('game_event', {'event': 'capture', 'player': PLAYERS[sid]['name']})
            broadcast_lobby_state()
        else:
            saw_str = ", ".join(unique_found[:2]).upper() if unique_found else "NOTHING"
            emit('upload_ack', {'success': False, 'msg': f"WRONG. SAW: {saw_str}"}, to=sid)

    except Exception as e:
        print(f"Error: {e}")
        emit('upload_ack', {'success': False, 'msg': "ERROR PROCESSING IMAGE"}, to=sid)

def game_engine_loop():
    with app.app_context():
        GAME_STATE['status'] = "ACTIVE"
        GAME_STATE['round'] = 0
        
        for pid in PLAYERS: 
            PLAYERS[pid]['score'] = 0
            PLAYERS[pid]['has_scored'] = False
            
        broadcast_lobby_state()
        socketio.emit('game_start_sequence')
        time.sleep(3)

        while GAME_STATE['round'] < GAME_STATE['max_rounds']:
            GAME_STATE['round'] += 1
            
            # --- ASSIGN TARGETS FROM QUEUE ---
            for pid in PLAYERS:
                PLAYERS[pid]['has_scored'] = False
                
                # Pop next item from their shuffled deck
                queue = PLAYERS[pid]['target_queue']
                
                if len(queue) > 0:
                    personal_target = queue.pop(0)
                else:
                    # Fallback if we run out of items (rare)
                    personal_target = random.choice(TARGET_MANIFEST)
                
                PLAYERS[pid]['target'] = personal_target
                
                socketio.emit('round_start', {
                    'round': GAME_STATE['round'],
                    'target': personal_target.upper()
                }, room=pid)

            # Timer
            dur = GAME_STATE['duration']
            for i in range(dur):
                socketio.emit('timer_tick', {'time_left': dur - i, 'total': dur})
                time.sleep(1)
            
        # End Game
        GAME_STATE['status'] = "ENDED"
        save_database()
        
        if PLAYERS:
            sorted_players = sorted(PLAYERS.values(), key=lambda x: x['score'], reverse=True)
            winner = sorted_players[0]
            socketio.emit('game_over', {
                'winner': winner['name'], 
                'score': winner['score'],
                'leaderboard': sorted_players
            })
        else:
            socketio.emit('game_over', {'winner': "NO ONE", 'score': 0, 'leaderboard': []})

def broadcast_lobby_state():
    leaderboard = [{'name': p['name'], 'score': p['score'], 'color': p['color']} for p in PLAYERS.values()]
    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    socketio.emit('lobby_update', {'players': leaderboard})

if __name__ == '__main__':
    PORT = 5002
    print(f"\n[SYSTEM] NEON SERVER ACTIVE ON PORT {PORT}")
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False)