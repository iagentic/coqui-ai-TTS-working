from flask import Flask, Response, request, stream_with_context, jsonify
from TTS.api import TTS
import io
import os
import torch
from pydub import AudioSegment
app = Flask(__name__)

# Load XTTS2 model (ensure it's installed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
SUPPORTED_FORMATS = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'opus': 'audio/ogg'
}
@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible TTS API using Coqui XTTS2
    """
    # Parse request data
    data = request.json
    print("Request ::", data)
    text = data.get("input", "Hello, this is a streaming text-to-speech service.")
    response_format = data.get('response_format', 'mp3').lower()
    voice = data.get("voice", "default")  # XTTS2 doesn't have predefined voices
    model = data.get("model", "tts-1")  # Placeholder for compatibility
    stream = data.get("stream", False)  # Boolean flag for streaming

    # Validate input
    if not text:
        return jsonify({"error": "Missing 'input' parameter"}), 400

    if response_format not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported 'response_format': {response_format}"}), 400

    # Generate speech to a WAV buffer
    wav_buffer = io.BytesIO()
    style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
    tts.tts_to_file(text=text, speaker="Barbora MacLean", language="en", style_wav=style_wav, speaker_wav=None, split_sentences=True, file_path=wav_buffer)
    wav_buffer.seek(0)

    # Convert WAV to desired format
    audio = AudioSegment.from_wav(wav_buffer)
    audio_buffer = io.BytesIO()
    audio.export(audio_buffer, format=response_format)
    audio_buffer.seek(0)

    def audio_stream():
        """
        Stream the generated audio file in chunks
        """
        while chunk := audio_buffer.read(1024):  # Read in 1 KB chunks
            yield chunk

    mime_type = SUPPORTED_FORMATS[response_format]

    # If stream=True, stream the response; otherwise, return full file
    if stream:
        return Response(stream_with_context(audio_stream()), content_type=mime_type)
    else:
        return Response(audio_buffer.read(), content_type=mime_type)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
