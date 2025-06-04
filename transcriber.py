import whisper
import language_tool_python
import os
from pydub import AudioSegment
from uuid import uuid4

model = whisper.load_model("base")
tool = language_tool_python.LanguageTool("en-US")


def convert_to_standard_wav(input_path, output_path):
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_frame_rate(44100).set_channels(1).set_sample_width(2)
    sound.export(output_path, format="wav")

def transcribe_and_correct(audio_path):
    temp_wav = f"converted_{uuid4()}.wav"
    convert_to_standard_wav(audio_path, temp_wav)

    result = model.transcribe(temp_wav, language="en")
    text = result["text"]
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)

    os.remove(audio_path)
    os.remove(temp_wav)

    # 문법 이슈 리스트 구성
    issues = []
    for match in matches:
        issues.append({
            "ruleId": match.ruleId,
            "message": match.message,
            "replacements": match.replacements,
            "context": match.context
        })

    return text.strip(), corrected.strip(), issues
