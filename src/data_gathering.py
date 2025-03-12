"""data_gathring.py is used to gather and process input data for training our models."""

import whisper
import json

def transcribe_audio(input_file: str, output_file: str) -> None:
    """Transcribes an MP3 file using Whisper and saves the transcription to a JSON file.

    Args:
        input_file (str): Path to the MP3 file.
        output_file (str): Path to the output JSON file.
    """
    model = whisper.load_model("small")
    result = model.transcribe(input_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"PLACEHOLDER": result["text"]}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    transcribe_audio(input_file="/Users/gabriel/Downloads/beginning_commentary.wav", output_file="/Users/gabriel/Documents/GitHub/EAFC-ML-Remaster/data/output.json")   
    print("Finsihed!")