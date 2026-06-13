import numpy as np
import os
from synth import render_track, save_wav

def compose(music_params, output_dir, name, mutation_position, bpm=140):
    os.makedirs(output_dir, exist_ok=True)
    mut_idx = mutation_position - 1
    paths   = {}

    for track_type in ("wt", "mutant"):
        params    = music_params[track_type]
        is_mutant = (track_type == "mutant")
        label     = "wildtype" if not is_mutant else "mutant"

        melody_notes  = []   # bright/main melody
        dark_notes    = []   # post-mutation dark instrument (mutant only)
        harmony_notes = []
        bass_notes    = []

        pitch      = params["pitch"]
        velocity   = params["velocity"]
        consonance = params["consonance"]
        seq_len    = len(pitch)

        current_time = 0.0
        beat         = 60.0 / bpm
        phrase_size  = 8

        for i in range(seq_len):
            p        = int(np.clip(pitch[i], 48, 84))
            vel      = int(velocity[i])
            note_dur = beat / 2

            if i > 0 and i % phrase_size == 0:
                current_time += beat * 0.25

            # ── Mutation site ─────────────────────────────────
            if i == mut_idx:
                if is_mutant:
                    current_time += beat * 0.5          # silence before

                    clash_dur = beat * 2
                    for semitone in [0, 1, 6]:          # minor 2nd + tritone
                        clash_p = int(np.clip(p - 6 + semitone, 0, 127))
                        dark_notes.append({
                            "pitch": clash_p, "velocity": 110,
                            "start": current_time, "end": current_time + clash_dur
                        })

                    bass_notes.append({
                        "pitch": int(np.clip(p - 24, 21, 60)), "velocity": 100,
                        "start": current_time, "end": current_time + beat
                    })

                    current_time += clash_dur + beat * 0.75   # silence after
                    continue
                else:
                    note_dur = beat
                    vel      = min(127, vel + 15)

            # ── Tension arc near mutation ─────────────────────
            dist = abs(i - mut_idx)
            if dist <= 4:
                note_dur = note_dur * (1 - dist * 0.05)
                vel      = min(127, vel + (4 - dist) * 5)

            # ── Post-mutation: mutant sounds broken ───────────
            if is_mutant and i > mut_idx:
                note_dur = note_dur * 1.1
                vel      = max(30, vel - 10)
                dark_notes.append({
                    "pitch": p, "velocity": vel,
                    "start": current_time, "end": current_time + note_dur * 0.85
                })
                current_time += note_dur
                continue

            melody_notes.append({
                "pitch": p, "velocity": vel,
                "start": current_time, "end": current_time + note_dur * 0.85
            })
            current_time += note_dur

        # ── Harmony: one chord per phrase ────────────────────
        phrase_time = 0.0
        for ps in range(0, seq_len, phrase_size):
            pe = min(ps + phrase_size, seq_len)
            t  = phrase_time
            for i in range(ps, pe):
                dur = beat / 2 * (1 - abs(i - mut_idx) * 0.05 if abs(i - mut_idx) <= 4 else 1)
                if i > 0 and i % phrase_size == 0:
                    t += beat * 0.25
                t += dur
            phrase_dur = t - phrase_time
            root       = int(np.clip(pitch[ps] - 12, 24, 72))
            avg_cons   = float(consonance[ps:pe].mean())
            in_mut     = ps <= mut_idx < pe

            if is_mutant and in_mut:
                intervals, hvel = [0, 3, 6, 9], 55    # diminished 7th
            elif avg_cons > 0.6:
                intervals, hvel = [0, 4, 7], 38       # major
            else:
                intervals, hvel = [0, 3, 7], 38       # minor

            for iv in intervals:
                harmony_notes.append({
                    "pitch": int(np.clip(root + iv, 21, 100)), "velocity": hvel,
                    "start": phrase_time, "end": phrase_time + phrase_dur * 0.92
                })
            phrase_time = t

        total_duration = current_time

        # ── Render all tracks and mix them ───────────────────
        all_notes = melody_notes + dark_notes + harmony_notes + bass_notes
        audio = render_track(all_notes, total_duration)

        out_path = os.path.join(output_dir, f"{name}_{label}.wav")
        save_wav(audio, out_path)
        paths[track_type] = out_path

        total_notes = len(melody_notes) + len(dark_notes)
        print(f"  {label}: {total_duration:.1f}s, {total_notes} melody notes")

    return {"wt_path": paths["wt"], "mutant_path": paths["mutant"],
            "duration": total_duration}