import whisper
from record import record



model = whisper.load_model("tiny")
# record("record.wav", time=3)
result = model.transcribe("record.wav")
print(result["text"])