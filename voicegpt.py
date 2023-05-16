"VoiceGPT – A minimal audio interface to ChatGPT"

import io
import os
import wave

import dotenv
import elevenlabs
import keyboard
import openai

import numpy as np

from pvrecorder import PvRecorder


class NamedBytesIO(io.BytesIO):
    def __init__(self, name):
        self.name = name


def get_wave_file_object_from_array(audio):
    f = NamedBytesIO("recording.wav")
    with wave.open(f, 'wb') as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(16000)  # used by pvrecorder
        wavfile.writeframes(audio.tobytes())
    f.seek(0)
    return f


def record_audiofile():
    recorder = PvRecorder(device_index=-1, frame_length=512)
    audio = []
    print("Hold the right shift key to record a message:")

    while True:
        if keyboard.is_pressed('shift'):
            recorder.start()
            print("Recording started")

            while keyboard.is_pressed('shift'):
                frame = recorder.read()
                audio.extend(frame)
            recorder.stop()
            recorder.delete()
            print("Recording stopped")

            audio = np.array(audio).astype(np.int16)
            return get_wave_file_object_from_array(audio)
        else:
            pass


# Setup
dotenv.load_dotenv()
openai.api_key = os.getenv("OPEN_API_NEW_KEY")
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:

    # 1. Record audio while user holds spacebar
    f_audio = record_audiofile()

    # 2. Transcribe audio
    transcript = openai.Audio.transcribe("whisper-1", f_audio)["text"]
    messages.append({"role": "user", "content": transcript})
    print("User:")
    print('\x1b[1;32;40m' + transcript + '\x1b[0m')

    # 3. OpenAI API completion call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    response_content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": response_content})
    print("Assistant:")
    print('\x1b[1;32;34m' + response_content + '\x1b[0m')

    # 4. Text to speech using Eleven Labs API
    audio = elevenlabs.generate(
        text=response_content,
        voice="Bella",
        model="eleven_monolingual_v1"
    )
    elevenlabs.play(audio)
