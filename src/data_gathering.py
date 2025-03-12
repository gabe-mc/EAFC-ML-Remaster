"""data_gathring.py is used to gather and process input data for training our models."""

import whisper

def transcribe_audio(input_file: str, output_file: str) -> None:
    """Transcribes an MP3 file using Whisper and saves the transcription to a text file.

    Args:
        input_file (str): Path to the MP3 file.
        output_file (str): Path to the output text file.
    """
    model = whisper.load_model("base")
    result = model.transcribe(input_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])