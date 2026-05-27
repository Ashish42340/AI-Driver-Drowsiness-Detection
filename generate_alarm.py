"""
generate_alarm.py
─────────────────
Run this script ONCE before starting main.py.
It creates alarm.wav in the same folder using only NumPy and wave.
No internet or external file is needed!

Usage:
    python generate_alarm.py
"""

import numpy as np
import wave
import struct
import os

def generate_alarm_wav(filename="alarm.wav", duration=2.0, sample_rate=44100):
    """
    Generate a beeping alarm sound and save it as a .wav file.

    Parameters
    ----------
    filename    : str   – Output file name
    duration    : float – Total duration in seconds
    sample_rate : int   – Audio quality (44100 = CD quality)
    """

    print(f"[INFO] Generating alarm sound → {filename}")

    # Create a series of beep-beep-beep tones
    # Each beep = 0.2 sec ON + 0.1 sec silence
    beep_freq   = 880    # Hz – High-pitched alert tone
    beep_on     = 0.20   # seconds of tone
    beep_off    = 0.10   # seconds of silence
    amplitude   = 28000  # 0-32767 range for 16-bit audio

    samples = []
    t       = 0.0
    dt      = 1.0 / sample_rate

    while t < duration:
        # Beep ON phase
        for _ in range(int(beep_on * sample_rate)):
            val = amplitude * np.sin(2 * np.pi * beep_freq * t)
            samples.append(int(val))
            t += dt

        # Beep OFF phase (silence)
        for _ in range(int(beep_off * sample_rate)):
            samples.append(0)
            t += dt

    # Write to a 16-bit WAV file
    output_path = os.path.join(os.path.dirname(__file__), filename)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)                      # Mono
        wf.setsampwidth(2)                      # 16-bit samples
        wf.setframerate(sample_rate)
        for s in samples:
            s = max(-32768, min(32767, s))      # Clamp to valid range
            wf.writeframes(struct.pack('<h', s))

    print(f"[SUCCESS] alarm.wav created at: {output_path}")
    print("[INFO]    Duration :", round(duration, 2), "seconds")
    print("[INFO]    Sample rate:", sample_rate, "Hz")


if __name__ == "__main__":
    generate_alarm_wav()
    print("\n✅ Done! Now run: python main.py")
