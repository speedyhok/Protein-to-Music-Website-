import requests

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"

PRESET_PAIRS = {
    "sickle_cell": {
        "name": "Hemoglobin beta — sickle cell disease",
        "wt_id": "P68871",
        "position": 6,
        "original": "E",
        "mutant": "V",
        "disease": "Sickle cell anemia",
        "blosum_score": -2,
        "note": "Most famous point mutation in medicine. One letter causes lifelong disease."
    },
    "tp53_cancer": {
        "name": "Tumour suppressor p53 — cancer",
        "wt_id": "P04637",
        "position": 175,
        "original": "R",
        "mutant": "H",
        "disease": "Most common mutation across all human cancers",
        "blosum_score": -1,
        "note": "R175H — found in breast, lung, colon, and virtually every other cancer type."
    },
    "brca1": {
        "name": "BRCA1 — hereditary breast cancer",
        "wt_id": "P38398",
        "position": 1699,
        "original": "R",
        "mutant": "W",
        "disease": "Hereditary breast and ovarian cancer",
        "blosum_score": -3,
        "note": "R1699W — disrupts the BRCT domain critical for DNA damage response."
    },
    "parkinsons": {
        "name": "Alpha-synuclein — Parkinson's disease",
        "wt_id": "P37840",
        "position": 53,
        "original": "A",
        "mutant": "T",
        "disease": "Parkinson's disease",
        "blosum_score": 0,
        "note": "A53T — causes the protein to misfold and clump into toxic aggregates in neurons."
    },
    "alzheimers": {
        "name": "Amyloid precursor protein — Alzheimer's disease",
        "wt_id": "P05067",
        "position": 717,
        "original": "V",
        "mutant": "I",
        "disease": "Familial Alzheimer's disease",
        "blosum_score": 3,
        "note": "V717I (London mutation) — increases production of toxic amyloid-beta plaques."
    },
    "cystic_fibrosis": {
        "name": "CFTR — cystic fibrosis",
        "wt_id": "P13569",
        "position": 551,
        "original": "G",
        "mutant": "D",
        "disease": "Cystic fibrosis",
        "blosum_score": -2,
        "note": "G551D — locks the chloride channel shut, one of the most severe CF mutations."
    },
    "huntingtons": {
        "name": "Huntingtin — Huntington's disease",
        "wt_id": "P42858",
        "position": 17,
        "original": "P",
        "mutant": "S",
        "disease": "Huntington's disease",
        "blosum_score": -1,
        "note": "Near the polyQ expansion region — disrupts protein clearance, causing neurodegeneration."
    },
    "kras_cancer": {
        "name": "KRAS — lung and pancreatic cancer",
        "wt_id": "P01116",
        "position": 12,
        "original": "G",
        "mutant": "D",
        "disease": "Lung and pancreatic cancer",
        "blosum_score": -1,
        "note": "G12D — locks KRAS in its 'on' state, driving uncontrolled cell growth. One of the most common cancer mutations."
    },
}




def fetch_sequence(uniprot_id):
    url = f"{UNIPROT_BASE}/{uniprot_id}.json"
    print(f"  Fetching {uniprot_id} from UniProt...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    sequence = data["sequence"]["value"]
    try:
        name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except (KeyError, IndexError):
        name = uniprot_id
    try:
        organism = data["organism"]["scientificName"]
    except KeyError:
        organism = "Unknown"
    return {"id": uniprot_id, "name": name, "organism": organism,
            "sequence": sequence, "length": len(sequence)}


def make_mutant_sequence(sequence, position, new_aa):
    idx = position - 1
    mutant = list(sequence)
    original = mutant[idx]
    mutant[idx] = new_aa
    return "".join(mutant), original


def get_pair(preset_key):
    config = PRESET_PAIRS[preset_key]
    print(f"\n[Fetcher] Loading: {config['name']}")
    print(f"  Disease  : {config['disease']}")
    print(f"  Mutation : {config['original']}{config['position']}{config['mutant']}")
    wt_data = fetch_sequence(config["wt_id"])
    mutant_seq, confirmed_original = make_mutant_sequence(
        wt_data["sequence"], config["position"], config["mutant"]
    )
    print(f"  Length   : {wt_data['length']} residues")
    return {
        "config": config,
        "wt": wt_data,
        "mutant": {**wt_data, "sequence": mutant_seq,
                   "name": wt_data["name"] + f" [{confirmed_original}{config['position']}{config['mutant']}]"},
    }