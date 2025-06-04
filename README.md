
# 🧠 LLaMA3 기반 문장 교정 모듈 (Corrector)

이 프로젝트는 LoRA 방식으로 파인튜닝된 LLaMA3 모델을 사용하여 잘못된 영어 문장을 교정하는 기능을 제공합니다.  
Whisper 등으로부터 생성된 텍스트를 입력하면, 올바른 문장으로 교정해주는 기능을 `corrector.py`에 정의되어 있습니다.

---

## 📁 파일 구성

| 파일명 | 설명 |
|--------|------|
| `corrector.py` | 👉 파인튜닝된 모델을 불러와서 문장을 교정하는 기능을 제공하는 모듈입니다. (`correct_sentence(text: str)` 함수) |
| `gpt_corrected.jsonl` | 👉 파인튜닝에 사용된 데이터셋 (학습용 참고) |
| `llama3_lora_adapter.zip` | ✅ [필수] 파인튜닝된 LoRA 어댑터 파라미터 |
| `llama3_lora_ckpt.zip` | (선택) 전체 학습 체크포인트 (재학습 시 사용 가능) |
| `lora_output.zip` | (선택) tokenizer 저장 폴더 (추론에는 없어도 됨) |
| `LLM_finetuning.ipynb` | 파인튜닝 전체 코드 (학습 구조 참고용) |

---

## ⚙️ 환경 설정

```bash
pip install torch transformers peft accelerate
```

또는 Conda 환경 사용 시:

```bash
conda create -n llama_env python=3.10
conda activate llama_env
pip install torch transformers peft accelerate
```

---

## 📦 압축 해제 (반드시 먼저 수행)

```bash
unzip llama3_lora_adapter.zip
```

> 이 디렉토리가 `corrector.py`와 같은 위치에 있어야 합니다.

---

## ✅ 사용법 예시

### 1. 파이썬 코드에서 사용

```python
from corrector import correct_sentence

text = "this are the example sentense i want to see if it correct."
result = correct_sentence(text)
print(result)
```

### 2. Flask 백엔드 연동 예시

```python
from flask import Flask, request, jsonify
from corrector import correct_sentence

app = Flask(__name__)

@app.route("/correct", methods=["POST"])
def correct_api():
    input_text = request.json.get("text", "")
    output = correct_sentence(input_text)
    return jsonify({"corrected": output})

if __name__ == "__main__":
    app.run()
```

---

## ❗ 주의 사항

- 모델은 `"meta-llama/Llama-3.1-8B-Instruct"` 기반으로 작동합니다.
- GPU 환경에서 실행해야 빠르게 작동합니다. (최소 16GB VRAM 권장)
- `llama3_lora_adapter/` 폴더는 반드시 `corrector.py`와 같은 위치에 있어야 합니다.
- `corrector.py` 내부에서 모델 로딩은 최초 1회만 수행됩니다.

---

## 📬 문의

모듈 관련 문의는 [준서]에게 해 주세요.
