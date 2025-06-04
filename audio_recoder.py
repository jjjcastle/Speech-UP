# audio_recoder.py
import os
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np

audio_recording_thread = None
is_recording_audio = False
audio_buffer = []
fs = 44100
current_session_id = None  # 세션 ID 저장

def _callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"[오디오] 상태 경고: {status}")
    audio_buffer.append(indata.copy())
    print("🎧 프레임 저장 중", flush=True)

def _record_audio_loop():
    global is_recording_audio, audio_buffer
    audio_buffer = []
    print("🎙️ 오디오 녹음 시작", flush=True)

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=_callback):
            while is_recording_audio:
                sd.sleep(100)  # 0.1초씩 슬립하며 계속 녹음
    except Exception as e:
        print(f"[오디오] InputStream 오류: {e}")

    print("🛑 오디오 녹음 종료 준비", flush=True)

def start_audio_recording(session_id):
    global is_recording_audio, audio_recording_thread, current_session_id

    # 장치 수동 설정: 필요 시 아래 인덱스 수정 (query_devices() 참고)
    sd.default.device = (1, None)  # (입력 장치 번호, 출력은 None)

    current_session_id = session_id
    is_recording_audio = True
    audio_recording_thread = threading.Thread(target=_record_audio_loop)
    audio_recording_thread.start()

def stop_audio_recording(session_id):
    global is_recording_audio, audio_buffer
    is_recording_audio = False
    if audio_recording_thread is not None:
        audio_recording_thread.join(timeout=5)

    if not audio_buffer:
        print(f"⚠️ {session_id} 오디오 버퍼가 비어있습니다. 저장하지 않음.")
        return

    try:
        audio = np.concatenate(audio_buffer, axis=0)
        print(f"🔢 총 프레임 수: {len(audio)} (약 {len(audio)/fs:.2f}초)")

        if len(audio)/fs < 0.5:
            print("⚠️ 녹음 시간이 0.5초 미만입니다. Whisper 전사 생략.")
            return

        folder = os.path.join("audio", session_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{session_id}.wav")
        sf.write(path, audio, fs)

        file_size = os.path.getsize(path)
        print(f"📦 저장된 wav 파일 크기: {file_size} bytes")
        if file_size < 1000:
            print(f"⚠️ {path} 용량이 너무 작습니다. (무효 파일일 수 있음)")

        print(f"✅ 오디오 저장 완료: {path}")
        return path
    except Exception as e:
        print(f"❌ 오디오 저장 중 오류 발생: {e}")
