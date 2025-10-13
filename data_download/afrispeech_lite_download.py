
import os, csv, uuid, math, random, collections
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import torchaudio
from datasets import load_dataset
import argparse

# ----------------------------
# I/O
# ----------------------------
def save_wav(split, accent_lower, ex, audio_dir, save_audio):
    """Saves an audio example as a WAV file."""
    audio = ex.get("audio")
    if audio is None:
        return False
    wav = audio["array"]
    sr = audio["sampling_rate"]
    out_dir = audio_dir / split / accent_lower
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid4())
    out_path = out_dir / f"{uid}.wav"
    if save_audio:
        sf.write(out_path.as_posix(), wav, sr)
    return True

def collect_split(split_name, ds_stream, per_accent_target, top5, audio_dir, save_audio):
    """Collects a specified number of audio examples per accent for a given split."""
    # fill targets by taking first matching items in order, no shuffle
    need = {a: per_accent_target for a in top5}
    got = {a: 0 for a in top5}
    pbar = tqdm(total=per_accent_target * len(top5), desc=f"{split_name}: saving")
    for ex in ds_stream:
        acc = (ex.get("accent") or "").strip().lower()
        if acc in need and need[acc] > 0:
            if save_wav(split_name, acc, ex, audio_dir, save_audio):
                need[acc] -= 1
                got[acc] += 1
                pbar.update(1)
                if all(v == 0 for v in need.values()):
                    break
    pbar.close()
    print(split_name, "completed:", got)
    # Warn if a split lacks enough items for any accent
    for a in top5:
        if need[a] > 0:
            print(f"[WARN] {split_name}: missing {need[a]} for accent '{a}'")
    return got

def main():
    """Main function to process and save audio data."""
    parser = argparse.ArgumentParser(description="Download and process Afrispeech-200 dataset.")
    parser.add_argument("--use_drive", action="store_true", help="Save under Google Drive")
    parser.add_argument("--top5", nargs="+", default=["yoruba", "igbo", "swahili", "hausa", "ijaw"],
                        help="Space-separated list of top 5 accents (lowercase)")
    parser.add_argument("--n_train", type=int, default=400, help="Number of examples per accent for the training split")
    parser.add_argument("--n_dev", type=int, default=80, help="Number of examples per accent for the development split")
    parser.add_argument("--n_test", type=int, default=120, help="Number of examples per accent for the test split")
    parser.add_argument("--save_audio", action="store_true", default=True, help="Set False to dry-run without saving audio files")

    args = parser.parse_args()

    USE_DRIVE = args.use_drive
    TOP5 = args.top5
    N_TRAIN = args.n_train
    N_DEV = args.n_dev
    N_TEST = args.n_test
    SAVE_AUDIO = args.save_audio

    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = Path("/content/drive/MyDrive/afrispeech_lite_top5_sequential")
    else:
        BASE = Path("/content/afrispeech_lite_top5_sequential")

    AUDIO_DIR = BASE / "audio"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load streams (no shuffle)
    train_stream = load_dataset("intronhealth/afrispeech-200", "all", split="train", streaming=True)
    dev_stream   = load_dataset("intronhealth/afrispeech-200", "all", split="validation", streaming=True)
    test_stream  = load_dataset("intronhealth/afrispeech-200", "all", split="test", streaming=True)

    # Run collection with fixed quotas
    collect_split("train", train_stream, N_TRAIN, TOP5, AUDIO_DIR, SAVE_AUDIO)
    collect_split("validation", dev_stream, N_DEV, TOP5, AUDIO_DIR, SAVE_AUDIO)
    collect_split("test", test_stream, N_TEST, TOP5, AUDIO_DIR, SAVE_AUDIO)

    print("Done. Files saved under:", AUDIO_DIR)

if __name__ == "__main__":
    main()
