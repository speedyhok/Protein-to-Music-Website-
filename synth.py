import numpy as np
from scipy.io import wavfile
import os

SAMPLE_RATE = 22050  # samples per second (lower = smaller files, still good quality)


def note_to_freq(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def make_tone(freq, duration, velocity, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Richer harmonic stack + subtle detuned layer for warmth (chorus effect)
    wave = (
        1.00 * np.sin(2 * np.pi * freq * t) +
        0.40 * np.sin(2 * np.pi * freq * 2 * t) +
        0.25 * np.sin(2 * np.pi * freq * 3 * t) +
        0.15 * np.sin(2 * np.pi * freq * 4 * t) +
        0.50 * np.sin(2 * np.pi * (freq * 1.005) * t)  # detuned layer
    )

    # ADSR-style envelope: quick attack, gentle decay, sustain, release
    n = len(wave)
    attack  = int(sample_rate * min(0.01, duration / 6))
    release = int(sample_rate * min(0.08, duration / 3))
    envelope = np.ones(n)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1, 0, release)
    # slight decay across the whole note so it doesn't feel flat
    decay_curve = np.linspace(1.0, 0.7, n)
    envelope = envelope * decay_curve

    wave = wave * envelope
    amplitude = (velocity / 127.0) * 0.25
    return wave * amplitude



def render_track(notes, total_duration, sample_rate=SAMPLE_RATE):
    """
    notes: list of dicts with keys: pitch (MIDI note), start (sec), end (sec), velocity
    Returns a numpy array of the mixed audio.
    """
    num_samples = int(sample_rate * total_duration) + sample_rate  # +1s padding
    audio = np.zeros(num_samples)

    for note in notes:
        freq = note_to_freq(note["pitch"])
        dur  = note["end"] - note["start"]
        if dur <= 0:
            continue
        tone = make_tone(freq, dur, note["velocity"], sample_rate)

        start_sample = int(note["start"] * sample_rate)
        end_sample   = start_sample + len(tone)
        if end_sample > len(audio):
            tone = tone[:len(audio) - start_sample]
            end_sample = len(audio)

        audio[start_sample:end_sample] += tone

    # Normalize to avoid clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    return audio


def save_wav(audio, path, sample_rate=SAMPLE_RATE):
    """Save float audio array as a 16-bit WAV file."""
    audio_int16 = np.int16(audio * 32767)
    wavfile.write(path, sample_rate, audio_int16)
    print(f"  Wrote: {path}")