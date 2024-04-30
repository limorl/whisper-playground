import os

def is_video(file_path: str) -> bool:
    extension = _get_file_extension(file_path)
    return extension in ['.mp4', '.mpeg-4', '.mov', '.wmv' '.mkv', '.avi']

def is_audio(file_path: str) -> bool:
    extension = _get_file_extension(file_path)
    return extension in ['.mp3', '.wav', '.m4a']

def get_file_name_without_extention(file_path: str) -> str:
    extension = _get_file_extension(file_path)
    base_name = os.path.basename(file_path)
    return base_name.removesuffix(extension)

def video2audio(file_path: str, output_prefix:str, output_dir: str) -> str:
     audio_path = os.path.join(output_dir, f"{output_prefix}.wav")
     os.system(f"ffmpeg -i '{file_path}' -ac 1 -ar 16000 '{audio_path}' -y")
     return audio_path

def _get_file_extension(file_path:str) -> str:
    return os.path.splitext(file_path)[1].lower()