import os
import whisper
import subprocess
import tempfile
import numpy as np
import soundfile as sf

# 모델 로딩
model = whisper.load_model("base")

# ffmpeg 경로 (절대 경로)
FFMPEG_PATH = r"C:/Users/school/Downloads/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
import whisper.audio
whisper.audio.FFMPEG_BINARY = FFMPEG_PATH  # 명시적으로 설정

# Whisper의 audio.py 함수 override
def patched_load_audio(path, sr=16000):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    cmd = [
        FFMPEG_PATH,
        "-y",  # overwrite
        "-i", path,
        "-f", "wav",
        "-vn",  # disable video stream
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sr),
        temp_path
    ]

    print(f"🎬 ffmpeg 변환 실행 중: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ffmpeg stderr: {result.stderr}")
        raise RuntimeError(f"⚠️ ffmpeg 변환 실패 또는 파일 손상: {temp_path}")

    audio, _ = sf.read(temp_path)
    audio = audio.astype(np.float32)
    os.unlink(temp_path)
    return audio

# 🔧 Whisper 내부 함수 재정의
whisper.audio.load_audio = patched_load_audio

def transcribe_audio(audio_path: str, session_id: str) -> str:
    print(f"🔍 Whisper 전사 시작: {audio_path}")
    abs_path = os.path.abspath(audio_path).replace("\\", "/")
    print(f"📂 Whisper에 전달된 경로: {abs_path}")

    try:
        result = model.transcribe(abs_path, language="en")
        transcript = result["text"]
        save_transcript(session_id, transcript)
        print("✅ 전사 완료")
        return transcript
    except Exception as e:
        print(f"❌ 전사 오류: {e}")
        return "(전사 실패)"

def save_transcript(session_id: str, transcript: str):
    folder = os.path.join("transcripts", session_id)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{session_id}_original.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"📁 저장 완료: {path}")
