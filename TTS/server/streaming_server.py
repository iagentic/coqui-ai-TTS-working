from flask import Flask, Response, request, stream_with_context, jsonify
from TTS.api import TTS
import io
import os
import torch
app = Flask(__name__)

# Load XTTS2 model (ensure it's installed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible TTS API using Coqui XTTS2
    """
    # Parse request data
    data = request.json
    text = data.get("input", "Hello, this is a streaming text-to-speech service.")
    voice = data.get("voice", "default")  # XTTS2 doesn't have predefined voices
    model = data.get("model", "tts-1")  # Placeholder for compatibility
    stream = data.get("stream", False)  # Boolean flag for streaming
    
    # Validate input
    if not text:
        return jsonify({"error": "Missing 'input' parameter"}), 400

    output_path = "temp_output.wav"

    # Generate speech file
    tts.tts_to_file(text=text, speaker="Kumar Dahl", language="en",speaker_wav=None, split_sentences=True, file_path=output_path)

    def audio_stream():
        """
        Stream the generated audio file in chunks
        """
        with open(output_path, "rb") as audio_file:
            while chunk := audio_file.read(1024):  # Read in 1 KB chunks
                yield chunk

    # If stream=True, stream the response; otherwise, return full file
    if stream:
        return Response(stream_with_context(audio_stream()), content_type="audio/wav")
    else:
        return Response(open(output_path, "rb"), content_type="audio/wav")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
