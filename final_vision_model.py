import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
import mediapipe as mp
from collections import Counter
import os
import torch.nn as nn

def generate_feedback(expression_score, pose_score, pose_log=None, expression_counts=None):
    # 표정 피드백
    if expression_score >= 90:
        expression_feedback = "😊 표정이 정말 좋습니다!"
    elif expression_score >= 80:
        expression_feedback = "🙂 전반적으로 표정이 좋습니다."
    elif expression_score >= 70:
        expression_feedback = "😐 조금 더 웃는 표정을 연습해보세요!"
    elif expression_score >= 60:
        expression_feedback = "😕 좋지 않은 표정이 가끔 나타납니다."
    else:
        expression_feedback = "😟 표정에 조금 더 신경 써주세요!"

    # 시선 피드백
    if pose_score >= 90:
        pose_feedback = "👁️ 시선 처리가 매우 안정적입니다!"
    elif pose_score >= 80:
        pose_feedback = "👀 시선이 약간 흔들렸지만 괜찮은 편입니다."
    elif pose_score >= 70:
        pose_feedback = "😶‍🌫️ 카메라 응시에 조금만 신경 써주세요!"
    else:
        pose_feedback = "😵 시선이 매우 불안정했습니다. 카메라 응시 연습이 필요합니다."

    # ✅ 시선 이탈 요약 (가장 많이 나온 1개 방향만)
    if pose_log:
        filtered = [gaze for _, _, gaze in pose_log if gaze not in ["Center", "Level"]]
        if pose_score < 85 :
            if filtered:
                most_common_dir, count = Counter(filtered).most_common(1)[0]
                pose_feedback += f"\n  - 주로 {most_common_dir} 방향을 많이 쳐다봅니다."

    expression_summary = ""
    if expression_counts:
        frown_count = expression_counts.get("frown", 0)
        neutral_count = expression_counts.get("neutral", 0)
        total_count = sum(expression_counts.values())

        if expression_score < 90:
            if frown_count >= max(5, total_count * 0.3):
                expression_summary += f"  - 불안한 표정이 자주 보입니다. 표정 연습이 필요합니다!\n"
            if neutral_count >= max(5, total_count * 0.3):
                expression_summary += f"  - 무표정이 많이 보였습니다. 웃는 표정을 연습해보세요!.\n"

    # 통합 피드백 반환
    return f"""{expression_feedback}
{expression_summary if expression_summary else ""}
{pose_feedback}
"""

def run_vision_model(folder_path):
    model_path = "C:/Users/school/Desktop/Speech_To_Text/vgg16_epoch5_sd.pth"
    class_labels = ["neutral", "smile", "frown"]
    fps = 12
    use_cuda = True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = vgg16(weights=None)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(class_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    expression_score = 100
    expression_counts = {label: 0 for label in class_labels}
    prev_expr = None
    confirmed_expr = None
    expression_streak = 0
    expression_log = []

    pose_score = 100
    pose_log = []
    prev_pose = None
    pose_count = 0

    prev_pitch = None
    pitch_count = 0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def apply_penalty(expr, streak):
        if expr == "frown" and streak >= 4:
            return 5
        elif expr == "neutral" and streak >= 4:
            return 1
        return 0

    def apply_bonus(expr, streak):
        if expr == "smile" and streak >= 12:
            return 1
        return 0

    for i, fname in enumerate(image_files):
        img_path = os.path.join(folder_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 이미지 불러오기 실패: {img_path}")
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            max_prob = np.max(probs)
            max_idx = np.argmax(probs)

        if max_prob >= 0.5:
            expr = class_labels[max_idx]
            #print(f"[표정 예측] Frame {i}: 확률={max_prob:.2f}, 예측 표정={expr}")
            expression_counts[expr] += 1

            expression_streak = expression_streak + 1 if expr == prev_expr else 1
            prev_expr = expr
        else:
            continue

        # 감점 구간
        if expression_streak >= 6 and expression_streak % (fps * 2) == 0:
            penalty = apply_penalty(expr, expression_streak)
            if penalty > 0:
                expression_score -= penalty
                print(f"[표정 -] Frame {i}: {expr} 지속으로 -{penalty}점 → 점수: {expression_score}")
                expression_log.append((i, expr))

        # 보너스 구간
        if expression_streak >= 24 and expression_streak % (fps * 2) == 0:
            bonus = apply_bonus(expr, expression_streak)
            if bonus > 0:
                expression_score += bonus
                print(f"[표정 +] Frame {i}: {expr} 지속으로 +{bonus}점 → 점수: {expression_score}")
                expression_log.append((i, expr))

        # 표정 변화에 따른 점수 변화
        if expression_streak >= 6 and expr != confirmed_expr:
            before_expr = confirmed_expr
            confirmed_expr = expr
            expression_counts[expr] += 1

            if expr == "smile":
                expression_score += 5
            elif expr == "frown":
                expression_score -= 5
            elif expr == "neutral":
                expression_score -= 1

            print(f"[표정 변화] Frame {i}: 표정 {before_expr} → {expr}, 점수: {expression_score}")
            expression_log.append((i, expr))  # ✅ 확정 점수 변경도 로그




        # 시선 분석
        gaze_direction = "Center"
        pitch_dir = "Level"
        result = face_mesh.process(image_rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            face_points = {"nose_tip": 1, "chin": 152, "left_eye_corner": 263,
                           "right_eye_corner": 33, "left_mouth_corner": 287, "right_mouth_corner": 57}
            model_points = np.array([
                (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
            ], dtype=np.float64)

            image_points = np.array([
                (landmarks[face_points["nose_tip"]].x * w, landmarks[face_points["nose_tip"]].y * h),
                (landmarks[face_points["chin"]].x * w, landmarks[face_points["chin"]].y * h),
                (landmarks[face_points["left_eye_corner"]].x * w, landmarks[face_points["left_eye_corner"]].y * h),
                (landmarks[face_points["right_eye_corner"]].x * w, landmarks[face_points["right_eye_corner"]].y * h),
                (landmarks[face_points["left_mouth_corner"]].x * w, landmarks[face_points["left_mouth_corner"]].y * h),
                (landmarks[face_points["right_mouth_corner"]].x * w, landmarks[face_points["right_mouth_corner"]].y * h)
            ], dtype=np.float64)

            cx = landmarks[face_points["nose_tip"]].x * w
            cy = landmarks[face_points["nose_tip"]].y * h
            camera_matrix = np.array([[w, 0, cx], [0, w, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            _, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch = angles[1], angles[0]

            if yaw > 20:
                gaze_direction = "Right"
            elif yaw < -20:
                gaze_direction = "Left"
            else:
                gaze_direction = "Center"

            if pitch > 15:
                pitch_dir = "Down"
            elif pitch < -15:
                pitch_dir = "Up"
            else:
                pitch_dir = "Level"
            #print(f"[시선 예측] Frame {i}: Gaze → {gaze_direction}, Pitch → {pitch_dir}")

        frame_index = int(os.path.splitext(fname)[0].split("_")[-1])
        timestamp = frame_index / fps

        if gaze_direction == prev_pose:
            pose_count += 1
        else:
            pose_count = 1
            prev_pose = gaze_direction

        # Pitch 기반 감점 (Down 또는 Up)
        if pitch_dir == prev_pitch:
            pitch_count += 1
        else:
            pitch_count = 1
            prev_pitch = pitch_dir

            # Yaw 기반 감점
        if gaze_direction in ["Left", "Right"]:
            if pose_count >= fps and pose_count % fps == 0:
                pose_score -= 3
                print(f"[시선 -] {gaze_direction} 방향 지속 → -3점 → 점수: {pose_score}")
                pose_log.append((i, timestamp, gaze_direction))

        if pitch_dir in ["Up", "Down"]:
            if pitch_count >= fps and pitch_count % fps == 0:
                pose_score -= 2
                print(f"[시선 -] {pitch_dir} 방향 기울어짐 지속 → -2점 → 점수: {pose_score}")
                pose_log.append((i, timestamp, pitch_dir))


    log_path = os.path.join(folder_path, "image_expression_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for frame_idx, expr in expression_log:
            time_sec = frame_idx / fps
            f.write(f"{frame_idx}프레임 ({time_sec:.2f}초): {expr}\n")

    gaze_log_path = os.path.join(folder_path, "image_gaze_log.txt")
    with open(gaze_log_path, "w", encoding="utf-8") as f:
        for frame_index, timestamp, gaze in pose_log:
            f.write(f"{frame_index}프레임 ({timestamp:.2f}초): {gaze}\n")

    feedback = generate_feedback(expression_score, pose_score, pose_log, expression_counts)
    with open(os.path.join(folder_path, "image_final_feedback.txt"), "w", encoding="utf-8") as f:
        f.write(feedback)

    print(f"표정 로그 저장됨: {log_path}")
    print(f"시선 로그 저장됨: {gaze_log_path}")
    print(feedback)
    print(f'''📊 최종 점수 요약 :
        - 표정 점수: {expression_score}
        - 시선 점수: {pose_score}''')
    return feedback

# run_vision_model(folder_path='C:/Users/school/Desktop/Speech_To_Text/captured_frames/32')