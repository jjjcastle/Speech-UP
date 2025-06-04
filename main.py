import cv2
import time
import threading
import os
from fastapi import FastAPI, Response, Form, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from uuid import uuid4
from voice_model import evaluate_wav
from corrector import correct_sentence
#from vision_model import run_vision_model
from final_vision_model import run_vision_model
from final_video_model import run_vision_video_model
from audio_recoder import start_audio_recording, stop_audio_recording
from Whisper_model import transcribe_audio, save_transcript
from gpt_feedback import generate_final_feedback
app = FastAPI()
FRAME_SAVE_INTERVAL = 1.0 / 12.0  # 초당 6프레임
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# ===== 전역 공유 자원 =====
shared_camera = cv2.VideoCapture(1)
shared_frame = None
capture_running = False

# ===== CORS & 정적 파일 =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# ===== 프레임 실시간 업데이트 =====
def frame_updater():
    global shared_frame, capture_running
    while capture_running:
        ret, frame = shared_camera.read()
        if ret:
            shared_frame = frame
        time.sleep(0.01)

# ===== 프레임 저장 함수 =====
def save_frames(session_id):
    global shared_frame
    session_folder = os.path.join(output_folder, session_id)
    os.makedirs(session_folder, exist_ok=True)

    frame_counter = 1
    last_save_time = time.time()

    while capture_running:
        if shared_frame is None:
            continue

        current_time = time.time()
        if current_time - last_save_time >= FRAME_SAVE_INTERVAL:
            filename = os.path.join(session_folder, f"{session_id}_{frame_counter}.jpg")
            cv2.imwrite(filename, shared_frame)
            frame_counter += 1
            last_save_time = current_time
        time.sleep(0.01)

# ===== 영상 스트리밍 =====
def video_stream_generator():
    global shared_frame
    while True:
        if shared_frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', shared_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_stream_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ===== 영상 및 오디오 캡처 시작 =====
@app.post("/start-video")
def start_video(session_id: str = Form(None)):
    global capture_running
    if session_id is None:
        session_id = str(uuid4())[:8]

    capture_running = True
    threading.Thread(target=frame_updater, daemon=True).start()
    threading.Thread(target=save_frames, args=(session_id,), daemon=True).start()
    threading.Thread(target=start_audio_recording, args=(session_id,), daemon=True).start()

    return {"message": "Video & Audio streaming started", "session_id": session_id}

# ===== 캡처 종료 =====
@app.post("/stop-video")
def stop_video(session_id: str = Form(None)):
    global capture_running
    capture_running = False
    if session_id:
        stop_audio_recording(session_id)
    return {"message": "Video streaming stopped"}

# ===== 이미지 → mp4 변환 =====
def images_to_video(session_folder, output_path, fps=6):
    # JPG 파일만 필터링하고, 번호 기준 정렬
    images = sorted(
        [img for img in os.listdir(session_folder) if img.lower().endswith(".jpg")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])  # 숫자 추출 정렬
    )

    if not images:
        raise ValueError("No images found in the folder.")

    # 첫 이미지로 영상 사이즈 설정
    first_image_path = os.path.join(session_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape

    # 영상 생성기 설정
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img_path = os.path.join(session_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height))  # 혹시 사이즈 다르면 맞춰줌
        out.write(frame)

    out.release()
    print(f"[✅ 영상 저장 완료] → {output_path}")


# ===== 처리 요청 =====
@app.post("/process-video")
async def process_video(request: Request):
    data = await request.form()
    session_id = data.get("session_id")
    question = data.get("question")
    if not session_id:
        return {"error": "세션 ID가 없습니다."}

    session_folder = os.path.join(output_folder, session_id)
    mp4_path = os.path.join(session_folder, f"{session_id}.mp4")
    audio_path = os.path.join("audio", session_id, f"{session_id}.wav")

    try:
        images_to_video(session_folder, mp4_path)
    except Exception as e:
        return {"error": f"영상 생성 실패: {str(e)}"}

    #result = run_vision_model(mp4_path=mp4_path)
    result = run_vision_model(folder_path=session_folder)
    #result = run_vision_video_model(mp4_path)
    transcript = transcribe_audio(audio_path, session_id)
    audio_analysis = evaluate_wav(audio_path)
    ted = correct_sentence(transcript)

    # z_score, user_score 기본값 설정
    z_score, user_score = 0.0, 0.0
    if isinstance(audio_analysis, dict):
        z_score = round(audio_analysis.get("z_score", 0.0), 2)
        user_score = round(audio_analysis.get("user_score", 0.0), 4)

    # GPT 기반 피드백 생성
    final_feedback = generate_final_feedback(
        z_score=z_score,
        user_score=user_score,
        vision_feedback=result,
        original=transcript,
        corrected=ted,
        question=question
    )

    return {
        "session_id": session_id,
        "result": result,
        "transcript": transcript,
        "audio_eval": audio_analysis,
        "lan_correct": ted,
        "final_feedback": final_feedback
    }

