from flask import Flask, Response, request, stream_with_context, jsonify
from TTS.api import TTS
import io
import os
import torch
from pydub import AudioSegment
import tiktoken
app = Flask(__name__)

# Load XTTS2 model (ensure it's installed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
SUPPORTED_FORMATS = {
    'mp3': {'mime_type': 'audio/mpeg', 'sampling_rate': 48000},
    'wav': {'mime_type': 'audio/wav', 'sampling_rate': 48000},
    'opus': {'mime_type': 'audio/ogg', 'sampling_rate': 48000}
}

def split_text_into_chunks(text, max_tokens=400):
    """
    Splits text into chunks, each with a maximum of `max_tokens` tokens.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible TTS API using Coqui XTTS2
    """
    # Parse request data
    data = request.json
    text = data.get("input", "Hello, this is a streaming text-to-speech service.")
    response_format = data.get('response_format', 'mp3').lower()
    stream = data.get("stream", False)  # Boolean flag for streaming

    # Validate input
    if not text:
        return jsonify({"error": "Missing 'input' parameter"}), 400

    if response_format not in SUPPORTED_FORMATS:
        return jsonify({"error": f"Unsupported 'response_format': {response_format}"}), 400

    # Determine the sampling rate based on the requested format
    sampling_rate = SUPPORTED_FORMATS[response_format]['sampling_rate']

    # Split text into chunks to adhere to token limits
    text_chunks = split_text_into_chunks(text, max_tokens=400)
    combined_audio = AudioSegment.silent(duration=0, frame_rate=sampling_rate)

    # Generate speech for each chunk and concatenate
    for chunk in text_chunks:
        wav_buffer = io.BytesIO()
        tts.tts_to_file(text=chunk,speaker_wav="/home/ubuntu/coqui-ai-TTS-working/TTS/voices/small.mp3", language="en", split_sentences=True, file_path=wav_buffer)
        wav_buffer.seek(0)
        audio_segment = AudioSegment.from_wav(wav_buffer)
        # Resample to the desired sampling rate if necessary
        if audio_segment.frame_rate != sampling_rate:
            audio_segment = audio_segment.set_frame_rate(sampling_rate)
        combined_audio += audio_segment

    # Convert combined audio to the desired format
    audio_buffer = io.BytesIO()
    combined_audio.export(audio_buffer, format=response_format)
    audio_buffer.seek(0)

    def audio_stream():
        """
        Stream the generated audio file in chunks
        """
        while chunk := audio_buffer.read(1024):  # Read in 1 KB chunks
            yield chunk

    mime_type = SUPPORTED_FORMATS[response_format]['mime_type']

    # If stream=True, stream the response; otherwise, return full file
    if stream:
        return Response(stream_with_context(audio_stream()), content_type=mime_type)
    else:
        return Response(audio_buffer.read(), content_type=mime_type)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
