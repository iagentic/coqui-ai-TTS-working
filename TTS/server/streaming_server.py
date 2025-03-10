import requests
import simpleaudio as sa
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# API Endpoint
API_URL = "http://localhost:5000/v1/audio/speech"

# OpenAI-compatible JSON request
DATA = {
    "model": "tts-1",  # Placeholder for compatibility
    "input": "This is a real-time streaming test using Coqui XTTS2.",
    "voice": "alloy",  # Placeholder, Coqui doesn't use predefined voices
    "stream": True  # Enable streaming
}

# Send POST request with streaming enabled
response = requests.post(API_URL, json=DATA, stream=True)

if response.status_code == 200:
    audio_buffer = BytesIO()
    
    # Read streamed chunks and write to buffer
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            audio_buffer.write(chunk)
    
    # Convert buffer to an audio segment
    audio_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(audio_buffer)
    
    # Play the streamed audio
    print("Playing streamed XTTS2 audio...")
    play(audio_segment)

else:
    print("Error:", response.status_code, response.text)
