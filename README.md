# whisper-playground

A playground to use whisper python package for transcription.
A [dev container](https://code.visualstudio.com/docs/devcontainers/containers) is used to set up all that is needed included whisper, pyannote, ffmpeg and pydub.

Requirements:
* Docker installed locally
* If you want to use diarization, `pyannote` package reuires Hugging Face access token  

1. Open folder in VS Code
2. copy .devcontainer/env.sample into .devcontainer/.env file and update environment variables as needed.
3. Run 'Reopen in Container' from the command pallete

## Transcription & Diarization
To transcribe a video or audio file, run:
```shell
python transcription.py --file "path/to/audio/or/video/file"
```

If you also want to add diarization and align the transcription with each speaker segment run"
```shell
python transcription.py --file "path/to/audio/or/video/file" --diarization True
```