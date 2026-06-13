from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import numpy as np

from fetcher  import get_pair, PRESET_PAIRS
from embedder import load_model, embed_sequence, build_music_params, compare_embeddings, get_magnitude_arrays
from composer import compose

import requests as req_lib
import time as time_lib

def uniprot_get(url, params=None, max_retries=3, delay=1.5):
    """GET request to UniProt with automatic retries on 503/timeout."""
    for attempt in range(max_retries):
        try:
            resp = req_lib.get(url, params=params, timeout=15)
            if resp.status_code == 503:
                if attempt < max_retries - 1:
                    time_lib.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except (req_lib.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time_lib.sleep(delay)
                continue
            raise
    return resp

app = Flask(__name__)

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-2 once at startup — reused for every request
print("Loading ESM-2 model (this happens once)...")
tokenizer, model = load_model()
print("Server ready.")


@app.route("/")
def home():
    return render_template("index.html", presets=PRESET_PAIRS)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    preset = data.get("preset")

    if preset not in PRESET_PAIRS:
        return jsonify({"error": "Unknown preset"}), 400

    t0 = time.time()

    # Stage 1: fetch
    pair    = get_pair(preset)
    config  = pair["config"]
    wt_seq  = pair["wt"]["sequence"]
    mut_seq = pair["mutant"]["sequence"]
    mut_pos = config["position"]

    # Stage 2: embed (full sequence needed for accurate embeddings)
    wt_emb  = embed_sequence(wt_seq,  tokenizer, model)
    mut_emb = embed_sequence(mut_seq, tokenizer, model)
    delta   = compare_embeddings(wt_emb, mut_emb)

    # ── Trim to a window around the mutation for the music ──
    WINDOW = 20  # residues before/after mutation
    win_start = max(0, mut_pos - 1 - WINDOW)
    win_end   = min(len(wt_seq), mut_pos - 1 + WINDOW + 1)

    wt_seq_win  = wt_seq[win_start:win_end]
    mut_seq_win = mut_seq[win_start:win_end]
    wt_emb_win  = wt_emb[win_start:win_end]
    mut_emb_win = mut_emb[win_start:win_end]
    mut_pos_win = mut_pos - win_start  # adjusted position within the window

    # Stage 3: music params (using the windowed sequence)
    music_params = build_music_params(wt_seq_win, mut_seq_win, wt_emb_win, mut_emb_win, mut_pos_win)
    # Stage 4: compose audio
    result = compose(music_params, output_dir=OUTPUT_DIR, name=preset,
                      mutation_position=mut_pos_win, bpm=140)

    # Build chart data: delta values around the mutation site
    window_start = max(0, mut_pos - 4)
    window_end   = min(len(delta), mut_pos + 4)

    wt_mag, mut_mag = get_magnitude_arrays(wt_emb_win, mut_emb_win)

    chart_data = {
        "positions": list(range(win_start + 1, win_end + 1)),
        "deltas": [round(float(d), 3) for d in delta[win_start:win_end]],
        "wt_magnitude": [round(float(m), 3) for m in wt_mag],
        "mut_magnitude": [round(float(m), 3) for m in mut_mag],
        "mutation_position": mut_pos,
    }

    return jsonify({
        "protein_name": pair["wt"]["name"],
        "organism": pair["wt"]["organism"],
        "mutation": f"{config['original']}{config['position']}{config['mutant']}",
        "disease": config["disease"],
        "note": config["note"],
        "chart": chart_data,
        "wt_audio": f"/audio/{preset}_wildtype.wav",
        "mutant_audio": f"/audio/{preset}_mutant.wav",
        "time_taken": round(time.time() - t0, 1),
    })


@app.route("/audio/<filename>")
def audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)

import re

DISEASE_KEYWORDS = [
    "syndrome", "disease", "anemia", "deficiency", "cancer", "carcinoma",
    "dystrophy", "fibrosis", "Parkinson", "Alzheimer", "Huntington",
    "diabetes", "epilepsy", "ataxia", "neuropathy"
]

@app.route("/variants/<uniprot_id>")
def variants(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    resp = uniprot_get(url)
    data = resp.json()
    
    results = []
    for feature in data.get("features", []):
        if feature.get("type") != "Natural variant":
            continue

        desc = feature.get("description", "")
        alt_seq = feature.get("alternativeSequence", {})
        original = alt_seq.get("originalSequence")
        alternatives = alt_seq.get("alternativeSequences", [])

        if not original or not alternatives:
            continue

        position = feature["location"]["start"]["value"]
        if feature["location"]["end"]["value"] != position:
            continue

        desc_lower = desc.lower()

        # Tier 1: named syndrome/disease (not just "cancer")
        named_match = re.search(
            r"in ([A-Za-z][A-Za-z0-9\-,' ]*(?:syndrome|disease|anemia|deficiency|dystrophy|fibrosis|disorder|atrophy))",
            desc, re.IGNORECASE
        )
        # Tier 2: generic cancer mention
        cancer_match = "cancer" in desc_lower or "carcinoma" in desc_lower or "tumor" in desc_lower

        if named_match:
            tier = 1
            disease_name = named_match.group(1).strip()
        elif cancer_match:
            tier = 2
            disease_name = "cancer-associated variant"
        else:
            continue  # skip variants with no disease association at all

        results.append({
            "position": position,
            "original": original,
            "mutant": alternatives[0],
            "label": f"{original}{position}{alternatives[0]} — {disease_name}",
            "disease": disease_name,
            "description": desc,
            "tier": tier,
        })

    # Sort: named diseases first, then by description length (more detail = more notable)
    results.sort(key=lambda r: (r["tier"], -len(r["description"])))

    # Deduplicate
    seen = set()
    unique_results = []
    for r in results:
        key = (r["position"], r["mutant"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    return jsonify({
        "protein_id": uniprot_id,
        "protein_name": data["proteinDescription"]["recommendedName"]["fullName"]["value"],
        "variants": unique_results[:30],  # cap at 30 for dropdown sanity
    })


@app.route("/search_proteins")
def search_proteins():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{query} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,protein_name,gene_names",
        "format": "json",
        "size": 10,
    }
    resp = uniprot_get(url, params=params)
    data = resp.json()

    results = []
    for entry in data.get("results", []):
        try:
            name = entry["proteinDescription"]["recommendedName"]["fullName"]["value"]
        except KeyError:
            continue
        gene = entry.get("genes", [{}])[0].get("geneName", {}).get("value", "")
        results.append({
            "id": entry["primaryAccession"],
            "name": name,
            "gene": gene,
        })

    return jsonify({"results": results})


@app.route("/generate_custom", methods=["POST"])
def generate_custom():
    data = request.get_json()
    uniprot_id = data.get("uniprot_id")
    position   = int(data.get("position"))
    mutant_aa  = data.get("mutant")

    t0 = time.time()

    # Fetch and mutate
    from fetcher import fetch_sequence, make_mutant_sequence
    wt_data = fetch_sequence(uniprot_id)
    wt_seq  = wt_data["sequence"]
    mut_seq, original_aa = make_mutant_sequence(wt_seq, position, mutant_aa)

    # Embed
    wt_emb  = embed_sequence(wt_seq,  tokenizer, model)
    mut_emb = embed_sequence(mut_seq, tokenizer, model)
    delta   = compare_embeddings(wt_emb, mut_emb)

    # Window around mutation
    WINDOW = 20
    win_start = max(0, position - 1 - WINDOW)
    win_end   = min(len(wt_seq), position - 1 + WINDOW + 1)

    wt_seq_win  = wt_seq[win_start:win_end]
    mut_seq_win = mut_seq[win_start:win_end]
    wt_emb_win  = wt_emb[win_start:win_end]
    mut_emb_win = mut_emb[win_start:win_end]
    mut_pos_win = position - win_start

    music_params = build_music_params(wt_seq_win, mut_seq_win, wt_emb_win, mut_emb_win, mut_pos_win)

    safe_name = f"custom_{uniprot_id}_{original_aa}{position}{mutant_aa}"
    result = compose(music_params, output_dir=OUTPUT_DIR, name=safe_name,
                      mutation_position=mut_pos_win, bpm=140)

    window_start_delta = max(0, mut_pos_win - 4)
    window_end_delta   = min(len(delta[win_start:win_end]), mut_pos_win + 4)
    delta_window = delta[win_start:win_end]

    wt_mag, mut_mag = get_magnitude_arrays(wt_emb_win, mut_emb_win)

    chart_data = {
        "positions": list(range(win_start + 1, win_end + 1)),
        "deltas": [round(float(d), 3) for d in delta_window],
        "wt_magnitude": [round(float(m), 3) for m in wt_mag],
        "mut_magnitude": [round(float(m), 3) for m in mut_mag],
        "mutation_position": position,
    }

    return jsonify({
        "protein_name": wt_data["name"],
        "organism": wt_data["organism"],
        "mutation": f"{original_aa}{position}{mutant_aa}",
        "disease": data.get("disease", "Custom variant"),
        "note": data.get("description", ""),
        "chart": chart_data,
        "wt_audio": f"/audio/{safe_name}_wildtype.wav",
        "mutant_audio": f"/audio/{safe_name}_mutant.wav",
        "time_taken": round(time.time() - t0, 1),
    })

    
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)