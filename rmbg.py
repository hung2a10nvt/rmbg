#! /usr/bin/env python3
import argparse
import sys
import os
import numpy as np
import subprocess
from pydub import AudioSegment
import librosa
import random

class Fingerprint:
    """Base class for audio fingerprinting algorithms."""
    pass

class PhillipsGuanYang(Fingerprint):
    SAMPLE_RATE = 5000
    FRAME_SIZE = 0.37
    FRAME_STEP = 11.6e-3
    FREQ_MIN = 300
    FREQ_MAX = 2000
    NUM_BANDS = 33
    BITS_PER_FINGERPRINT = 32

    sr = 5000

    def __init__(self, sr_target=5000):
        self.sr_target = sr_target

    @staticmethod
    def _calculate_ber(fp1, fp2):
        "Calculates the Bit Error Rate between two fingerprints."
        min_len = min(len(fp1), len(fp2))
        fp1_cut = fp1[:min_len]
        fp2_cut = fp2[:min_len]

        # XOR to find differing bits
        xor_result = np.bitwise_xor(fp1_cut, fp2_cut)
        total_bit_errors = np.sum(xor_result)
        total_bits = min_len * 32

        return total_bit_errors / total_bits

    @staticmethod
    def _extract_fingerprints(y, sr_target=5000):
        frame_length = int(PhillipsGuanYang.FRAME_SIZE * sr_target)
        hop_length = int(PhillipsGuanYang.FRAME_STEP * sr_target)

        # STFT and Energy matrix
        D = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length,
                               window='hann', center=False))
        S = D.T ** 2

        # Filterbank group energy calculation
        freq_bins = np.fft.rfftfreq(frame_length, 1/sr_target)
        freq_edges = np.logspace(np.log10(PhillipsGuanYang.FREQ_MIN),
                                 np.log10(PhillipsGuanYang.FREQ_MAX),
                                 PhillipsGuanYang.NUM_BANDS + 1)

        fb_matrix = np.zeros((len(freq_bins), PhillipsGuanYang.NUM_BANDS))
        for i in range(PhillipsGuanYang.NUM_BANDS):
            mask = (freq_bins >= freq_edges[i]) & (freq_bins < freq_edges[i+1])
            fb_matrix[mask, i] = 1.0

        energy_matrix = np.dot(S, fb_matrix)

        # Binary fingerprint generation
        diff_bands = energy_matrix[:, :32] - energy_matrix[:, 1:33]
        final_diff = diff_bands[1:, :] - diff_bands[:-1, :]

        return (final_diff > 0).astype(np.uint8)

    def find_best_anchor(self, fp_music):
        window_size = int(5 / self.FRAME_STEP)
        num_frames = len(fp_music)

        if num_frames <= window_size:
            return 0

        # Выбор кандидатов с шагом 5 секунд
        step = int(5 / self.FRAME_STEP)
        candidates = list(range(0, num_frames - window_size, step))

        if not candidates:
            return 0

        # VP-TREE ДЛЯ ПОИСКА СУБ-ОПТИМАЛЬНОГО МЕДОИДА

        # 1. Выбор случайной опорной точки (Vantage Point v1)
        v1 = random.choice(candidates)
        fp_v1 = fp_music[v1 : v1 + window_size]

        # 2. Поиск самой удаленной точки от v1 (точка v2)
        max_dist = -1
        v2 = v1
        for cand in candidates:
            fp_cand = fp_music[cand : cand + window_size]
            dist = self._calculate_ber(fp_v1, fp_cand)
            if dist > max_dist:
                max_dist = dist
                v2 = cand

        # 3. Теперь v2 находится на краю "облака" данных.
        # Суб-оптимальный медоид (якорь) обычно находится в области с высокой
        # плотностью, которую легче найти, анализируя окружение v2.

        fp_v2 = fp_music[v2 : v2 + window_size]

        best_offset = v2
        min_total_dist = float('inf')

        # Берем выборку для локального поиска медоида вокруг v2
        sample_size = min(30, len(candidates))
        test_samples = random.sample(candidates, sample_size)

        for cand in test_samples:
            cand_fp = fp_music[cand : cand + window_size]
            current_total_dist = 0
            for ref in test_samples:
                ref_fp = fp_music[ref : ref + window_size]
                current_total_dist += self._calculate_ber(cand_fp, ref_fp)

            if current_total_dist < min_total_dist:
                min_total_dist = current_total_dist
                best_offset = cand

        return best_offset

    def find_segment(self, fp_music, fp_video, block_seconds=1.0, anchor_offset=None):
        search_window_frames = int(5 / self.FRAME_STEP)

        if len(fp_music) < 100 + search_window_frames or len(fp_video) < search_window_frames:
            return None

        # COARSE ANCHOR SEARCH
        if anchor_offset is not None:
            start_offset_music = anchor_offset
        else:
            start_offset_music = int(len(fp_music) * 0.30)

        if start_offset_music + search_window_frames > len(fp_music):
            start_offset_music = max(0, len(fp_music) - search_window_frames)

        query_block = fp_music[start_offset_music : start_offset_music + search_window_frames]

        min_ber = 1.0
        best_offset_video = -1
        scan_step = 2

        for i in range(0, len(fp_video) - len(query_block), scan_step):
            # Calculate BER
            xor_result = np.bitwise_xor(query_block, fp_video[i : i + len(query_block)])
            ber = np.sum(xor_result) / (len(query_block) * 32.0)

            if ber < min_ber:
                min_ber = ber
                best_offset_video = i

        if min_ber > 0.38:
            return None

        # Calculate BER for each frame (VECTORIZATION)
        # align_offset = position in video of music's frame 0
        align_offset = best_offset_video - start_offset_music

        start_v = max(0, align_offset)
        start_m = max(0, -align_offset)

        match_length = min(len(fp_video) - start_v, len(fp_music) - start_m)

        if match_length <= 0:
            return None

        aligned_v = fp_video[start_v : start_v + match_length]
        aligned_m = fp_music[start_m : start_m + match_length]

        # Handle matrix
        xor_bits = np.bitwise_xor(aligned_v, aligned_m)
        ber_per_frame = np.sum(xor_bits, axis=1) / 32.0

        # Smooth BER with 1.5s-sliding window
        smooth_window = int(1.5 / self.FRAME_STEP)
        kernel = np.ones(smooth_window) / smooth_window
        smoothed_ber = np.convolve(ber_per_frame, kernel, mode='same')

        # 2-ways scan on smoothed BER array
        # Position of anchor
        anchor_idx_aligned = start_offset_music - start_m

        tolerance_ber = 0.39
        max_bad_frames = int(2.5 / self.FRAME_STEP) # Max 2.5s continous noise

        # Forward to find Endtime
        end_idx = match_length
        bad_count = 0
        for i in range(anchor_idx_aligned, match_length):
            if smoothed_ber[i] > tolerance_ber:
                bad_count += 1
                if bad_count > max_bad_frames:
                    end_idx = max(anchor_idx_aligned, i - bad_count)
                    break
            else:
                bad_count = 0

        # Backward to find Start Time
        start_idx = 0
        bad_count = 0
        for i in range(anchor_idx_aligned, -1, -1):
            if smoothed_ber[i] > tolerance_ber:
                bad_count += 1
                if bad_count > max_bad_frames:
                    start_idx = min(anchor_idx_aligned, i + bad_count)
                    break
            else:
                bad_count = 0

        start_time = (start_v + start_idx) * self.FRAME_STEP
        end_time = (start_v + end_idx) * self.FRAME_STEP

        return start_time, end_time, min_ber

    def identify_music_in_video(self, music_path, y_source_array):
        try:
            # Load the reference music
            # extract its fingerprints
            y_music = load_audio(music_path, target_sr=self.sr_target)
            fp_music = self._extract_fingerprints(y_music, self.sr_target)

            # Find the best anchor
            best_anchor = self.find_best_anchor(fp_music)

            if y_source_array is None or len(y_source_array) == 0:
                return None

            # Extract fingerprints for the source (the video/podcast audio)
            fp_source = self._extract_fingerprints(y_source_array, self.sr_target)

            # Matching logic
            result = self.find_segment(fp_music, fp_source, anchor_offset=best_anchor)

            return result

        except Exception as e:
            print(f"Error in identification process: {e}")
            return None

    def get(self, audio):
        assert isinstance(audio, AudioSegment)

        # 1. Ensure audio matches target sample rate and mono channel
        audio = audio.set_frame_rate(self.sr_target).set_channels(1)

        # 2. Convert AudioSegment to NumPy float array (In-memory Pipe logic)
        y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        # 3. Extract and return the fingerprint (hash value)
        fingerprint_matrix = self._extract_fingerprints(y, self.sr_target)
        return fingerprint_matrix
    
def load_audio(path, target_sr=5000):
    y, sr_orig = librosa.load(path, sr=None)

    y_resampled = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)

    return y_resampled

def get_audio_from_video(input_path, target_sr=5000):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-f', 's16le',      # raw PCM 16-bit
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr), # Sample rate
        '-ac', '1',
        '-'
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        if process.returncode != 0:
            print(f"FFmpeg Error: {err.decode()}")
            return None

        # bytes -> NumPy
        audio_int16 = np.frombuffer(out, dtype=np.int16)

        # To float32 for fingerprint extracting
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        return audio_float32

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def mute_video_segment(input_video_path, t_start, t_end, output_video_path):
    # -c:v copy -> Keep source video
    # -af "volume=0:enable='between(t,start,end)'" -> Set volume = 0 during t_start - t_end
    # -c:a aac -> Encode audio flow to AAC
    command = [
        'ffmpeg',
        '-y',  
        '-i', input_video_path,
        '-c:v', 'copy',
        '-af', f"volume=0:enable='between(t,{t_start},{t_end})'",
        '-c:a', 'aac',
        output_video_path
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if process.returncode != 0:
        print(f"FFmpeg error: {process.stderr.decode('utf-8')}")
    else:
        print(f"Success! Clean video saved at: {output_video_path}")
        
def remove_copyright_from_video(infected_audio_path, clean_video_path, music_path, pgy_instance, output_path):
    print(f"Scanning for '{os.path.basename(music_path)}'...")
    
    audio_seg = AudioSegment.from_file(infected_audio_path).set_frame_rate(pgy_instance.sr_target).set_channels(1)
    y_infected_array = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
    
    result = pgy_instance.identify_music_in_video(music_path, y_infected_array)
    
    if result:
        pred_start, pred_end, ber = result
        print(f"Music detected: {pred_start:.2f}s - {pred_end:.2f}s (BER: {ber:.4f})")
        
        mute_video_segment(
            input_video_path=clean_video_path, 
            t_start=pred_start, 
            t_end=pred_end, 
            output_video_path=output_path
        )
    else:
        print("Safe: music not found.")
        
## CLI
def main():
    parser = argparse.ArgumentParser(description="Copyrighted music remover (Mute).")
    parser.add_argument("input_video", help="Path to the input mp4 video file")
    parser.add_argument("input_audio_background", help="Path to the background audio file to search for (wav/mp3)")
    parser.add_argument("output_video", help="Path to save the output mp4 video file")

    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        print(f"Error: Video not found '{args.input_video}'")
        sys.exit(1)
    if not os.path.exists(args.input_audio_background):
        print(f"Error: Audio file not found '{args.input_audio_background}'")
        sys.exit(1)
        
    pgy = PhillipsGuanYang(sr_target=5000)

    print(f"Extracting audio from video...")
    y_infected = get_audio_from_video(args.input_video, target_sr=5000)
    if y_infected is None:
        print("Error: Could not extract audio. Exiting.")
        sys.exit(1)

    print(f"Scanning for '{os.path.basename(args.input_audio_background)}'...")
    result = pgy.identify_music_in_video(args.input_audio_background, y_infected)

    if result:
        pred_start, pred_end, ber = result
        print(f"Violation detected: {pred_start:.2f}s - {pred_end:.2f}s (BER: {ber:.4f})")
        print(f"Cutting audio from the original video...")
        
        mute_video_segment(args.input_video, pred_start, pred_end, args.output_video)
        print(f"[Succeeded] Video saved at: {args.output_video}")
    else:
        print(f"[Safe] No violations found. File unchanged.")

if __name__ == "__main__":
    main()
