import pyaudio
import wave
from pydub import AudioSegment
import keyboard
import time
import os
import re

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


def get_next_file_number():
    """audio_숫자.wav 형식의 파일 중 가장 높은 번호를 찾아 +1 리턴"""
    existing_files = os.listdir(".")
    pattern = re.compile(r"audio_(\d+)\.wav")
    numbers = [int(pattern.search(f).group(1)) for f in existing_files if pattern.search(f)]
    return max(numbers, default=0) + 1


def record_audio():
    """스페이스바를 눌러 시작/중지하는 녹음 및 파일 저장 함수"""
    audio = pyaudio.PyAudio()
    frames = []
    is_recording = False
    stream = None

    print("스페이스바를 눌러 녹음을 시작/중지하세요.")

    try:
        while True:
            if keyboard.is_pressed('space'):
                if not is_recording:
                    print("녹음 시작...")

                    # 파일 이름 결정
                    file_num = get_next_file_number()
                    wave_filename = f"audio_{file_num}.wav"
                    mp3_filename = f"audio_{file_num}.mp3"

                    stream = audio.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)
                    is_recording = True
                    frames = []

                else:
                    print("녹음 종료")
                    is_recording = False
                    stream.stop_stream()
                    stream.close()

                    # WAV 저장
                    with wave.open(wave_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))

                    # MP3 변환
                    audio_segment = AudioSegment.from_wav(wave_filename)
                    audio_segment.export(mp3_filename, format="mp3")
                    print(f"MP3 파일로 저장 완료: {mp3_filename}")
                    break  # 루프 종료

                while keyboard.is_pressed('space'):
                    time.sleep(0.1)  # 중복 방지

            if is_recording:
                data = stream.read(CHUNK)
                frames.append(data)

    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")

    audio.terminate()


# 단독 실행 시 동작
if __name__ == "__main__":
    record_audio()
