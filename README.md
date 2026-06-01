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

<img width="742" height="122" alt="test" src="https://github.com/user-attachments/assets/32b3eb47-cad4-46df-ab6f-e17f993d5347" />


## Testing
The algorithm's robustness and accuracy were evaluated under various acoustic conditions, specifically focusing on field recordings with speech overlap and heavy background noise down to a Signal-to-Noise Ratio (SNR) of **-10 dB**.

<img width="6654" height="1907" alt="Stress-test" src="https://github.com/user-attachments/assets/2e5e6f8c-cacb-48a4-9d30-650c0fe1d632" />
