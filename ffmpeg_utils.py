import subprocess
import os

def convert_to_wav(input_path, output_path, sample_rate=44100, channels=2):
    """
    Converts an audio/video file to WAV format using ffmpeg.

    Parameters:
    - input_path (str): Path to the input audio/video file.
    - output_path (str): Path where the output WAV file will be saved.
    - sample_rate (int): Audio sample rate in Hz (default: 44100).
    - channels (int): Number of audio channels (default: 2 for stereo).
    """
    try:
        # Ensure the input file exists
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i", input_path,  # Input file
            "-vn",  # Skip the video part if present
            "-acodec", "pcm_s16le",  # WAV format codec
            "-ar", str(sample_rate),  # Sample rate
            "-ac", str(channels),  # Channels (1=mono, 2=stereo)
            output_path
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Conversion successful: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e}")
        return False
    except Exception as ex:
        print(f"Error: {ex}")
        return False
