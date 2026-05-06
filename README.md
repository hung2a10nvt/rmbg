# Description: CLI tool to detect and mute specific audio tracks in videos.

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate  # Note: Use `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
chmod +x rmbg.py
```

## Syntax: 
```bash
./rmbg.py <input_video> <input_audio_background> <output_video>
```

## Example:
```bash
./rmbg.py input_podcast.mp4 jvke_her.wav output_clean.mp4
```
