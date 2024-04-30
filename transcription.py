import argparse
import logging
import os
import re
import sys
from collections import namedtuple
from file_utils import *
from typing import Any
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

logging.basicConfig(level=logging.DEBUG)
Segment = namedtuple('Segment', 'start end speaker')

def transcribe_without_diarization(audio_path: str, output_prefix: str, output_dir: str):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    transcription_file = os.path.join(output_dir, f"{output_prefix}-transcription.txt")

    with open(transcription_file, "w") as f:
        f.write(result["text"])

def transcribe_with_diarization(audio_path: str, output_prefix: str, output_dir: str, diarization_path: str):
    diarization = _load_diarization(diarization_path)
    model = whisper.load_model("base")
    transcription_file = os.path.join(output_dir, f"{output_prefix}-diarized-transcription.txt")
    _export_speaker_segments(audio_path, output_prefix, output_dir, diarization)

    transcriptions = {}
    for segment in diarization:
        segment_label = _get_speaker_segment_label(segment)
        audio_segment_path = os.path.join(output_dir, f"{output_prefix}-speaker-segment-{segment_label}.wav")
        transcription = model.transcribe(audio_segment_path)["text"]
        transcriptions[segment_label] = transcription

    
    with open(transcription_file, "w") as f:
        for label, text in transcriptions.items():
            f.write(f"Speaker {label}: {text}\n")

def _get_speaker_segment_label(segment: Any) -> str:
    return f"{segment.start}-{segment.end}-{segment.speaker}"

def _export_speaker_segments(audio_path: str, output_prefix: str, output_dir: str, diarization: Any):
    audio = AudioSegment.from_wav(audio_path)

    for segment in diarization:
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        speaker_segment = audio[start_ms:end_ms]
        segment_label = _get_speaker_segment_label(segment)
        speaker_segment_file = os.path.join(output_dir, f"{output_prefix}-speaker-segment-{segment_label}.wav")
        speaker_segment.export(speaker_segment_file, format="wav")

def _diarize(audio_path: str, output_prefix: str, output_dir: str) -> str:
    hf_access_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_access_token)
    diarization_result = diarization_pipeline({'audio': audio_path})
    
    diarization_file = os.path.join(output_dir, f"{output_prefix}-diarization.txt")

    # Consolidate speaker segments
    current_speaker = None
    current_start = None
    current_end = None
    time_threshold_secs = 10

    with open(diarization_file, "w") as f:
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if speaker == current_speaker and turn.start - current_end < time_threshold_secs:
                current_end = turn.end
            else:
                if current_speaker is not None:
                    f.write(f"{current_start:.3f} --> {current_end:.3f}: Speaker {current_speaker}\n")
                current_speaker = speaker
                current_start = turn.start
                current_end = turn.end

        # Write the last segment if it exists
        if current_speaker is not None:
            f.write(f"{current_start:.3f} --> {current_end:.3f}: Speaker {current_speaker}\n")

    return diarization_file

def _load_diarization(diarization_path:str):
    segments = []
    with open(diarization_path, 'r') as file:
        for line in file:
            start, end, speaker = re.match(r'(\d+\.\d+) --> (\d+\.\d+): Speaker ([a-zA-Z]+_\d+)', line).groups()
            # segments.append( { "start": float(start), "end": float(end), "speaker": speaker })
            segments.append(Segment(float(start), float(end), speaker))
    return segments

def _load_transcription(transcription_path: str):
    with open(transcription_path, 'r') as file:
        return file.read()

def transcribe(input_path: str, run_diarization: bool):
    output_dir = "./output"
    output_prefix = get_file_name_without_extention(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if is_video(input_path):
        audio_path = video2audio(input_path, output_prefix, output_dir)
    elif is_audio(input_path):
        audio_path = input_path
    else: 
        raise ValueError('Not a video or audio file!')
    
    if run_diarization:
        diarization_file = _diarize(audio_path, output_prefix, output_dir) 
        transcribe_with_diarization(audio_path, output_prefix, output_dir, diarization_file)
    else:
        transcribe_without_diarization(audio_path, output_prefix, output_dir)

    print("Transcription completed. Results are saved in the 'output' directory.")

def _exit_on_invalid_command():
     print("Invalid arguments. Usage: python transcribe.py --file <video-or-audio-file-path> [--diarization true/false]")
     sys.exit(1)

def _create_arg_parser():
    parser = argparse.ArgumentParser(prog='transcription.py', description='Transcribe video or audio file')
    parser.add_argument('--file', type = str, required = True, help = 'a path to audio or video file')
    parser.add_argument('--diarization', type = bool, required = False, default=False, help = 'true or false to indicate whether to diarize the transcription')
    return parser
     
if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()

    input_path = args.file
    run_diarization = args.diarization

    transcribe(input_path, run_diarization)
