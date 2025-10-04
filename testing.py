import os
import json
import re
import time
import math
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from openai import OpenAI

try:
    import tiktoken
except Exception:
    tiktoken = None

from dsl import objects  # keep using your DSL for object extraction

COLOR_NAMES = {
    0: "black (background)",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "magenta",
    7: "orange",
    8: "teal",
    9: "brown"
}

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ------------------ Utilities ------------------
def extract_json_from_text(text: str) -> Tuple[Optional[Any], str]:
    if not text:
        return None, ""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text), text
        except Exception:
            pass
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate), candidate
        except Exception:
            pass
    return None, text

def prompt_hash_key(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

class DiskCache:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        p = self.folder / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return None
        return None

    def set(self, key: str, obj: Any):
        p = self.folder / f"{key}.json"
        try:
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

# ------------------ LLM wrapper ------------------
class LLMClient:
    def __init__(self, api_key: str, model: str = 'gpt-4', cache_folder: str = '.llm_cache'):
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY environment variable is required')
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = DiskCache(cache_folder)

    def call(self, messages: List[Dict[str, str]], temperature: float = 0.7,
             max_tokens: int = 1500, use_cache: bool = True) -> str:

        prompt_repr = json.dumps({
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }, sort_keys=True)
        key = prompt_hash_key(prompt_repr)

        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug('Cache hit')
                return cached.get('content', '')

        response_text = None
        try:
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens
                )
                choice = resp.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    response_text = choice.message.content
                else:
                    response_text = getattr(choice, 'text', str(choice))
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'create'):
                resp = self.client.chat.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens
                )
                choice = resp.choices[0]
                if isinstance(choice.get('message'), dict) and 'content' in choice['message']:
                    response_text = choice['message']['content']
                else:
                    response_text = choice.get('text', '')
            else:
                resp = self.client.completions.create(
                    model=self.model, prompt=messages[-1]['content'],
                    max_tokens=max_tokens, temperature=temperature
                )
                response_text = getattr(resp.choices[0], 'text', '')
        except Exception as e:
            logger.error('LLM call failed: %s', e)
            raise

        response_text = str(response_text).strip()
        self.cache.set(key, {'content': response_text, 'timestamp': time.time()})
        return response_text
    
        # ------------------ Rich object serialization ------------------
import json
import time
from typing import List, Dict

# --- Enhanced object serialization with relations ---
def serialize_objects_for_gpt(obj_list, start_id=0):
    """
    Serializes objects for GPT, assigning unique IDs and providing detailed attributes.
    Returns a list of dictionaries representing each object.
    """
    result = []
    for idx, obj in enumerate(obj_list, start=start_id):
        obj_id = idx + 1  # Unique ID
        obj_cells = [
            {"cell": f"{v} ({COLOR_NAMES.get(v, f'color{v}')})", "position": [r, c]}
            for v, (r, c) in obj
        ]
        rows = [r for _, (r, _) in obj]
        cols = [c for _, (_, c) in obj]
        bbox = [[min(rows), min(cols)], [max(rows), max(cols)]]
        shape = [len(set(rows)), len(set(cols))]
        colors = sorted({f"{v} ({COLOR_NAMES.get(v, f'color{v}')})" for v, _ in obj})

        result.append({
            "id": obj_id,
            "cells": obj_cells,
            "bbox": bbox,
            "shape": shape,
            "position": [min(rows), min(cols)],
            "colors": colors,
            "size": len(obj_cells)
        })
    return result

def serialize_objects_with_relations(obj_list):
    """
    Adds relative position relations between objects.
    """
    objs = serialize_objects_for_gpt(obj_list)
    for o in objs:
        o['relations'] = []
        for other in objs:
            if o['id'] == other['id']:
                continue
            delta_row = other['position'][0] - o['position'][0]
            delta_col = other['position'][1] - o['position'][1]
            o['relations'].append({'to_id': other['id'], 'delta': [delta_row, delta_col]})
    return objs


def _normalize_text_for_dup_check(s: str) -> str:
    # simple normalization: lowercase, remove punctuation, collapse whitespace
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0

def generate_hypotheses_gpt(
    llm,  # LLMClient
    problem: Dict[str, any],
    n: int = 10,
    retries: int = 6   # bump retries
) -> List[Dict[str, any]]:
    
    serialized_examples = []
    for ex in problem.get('train', []):
        input_objs = serialize_objects_with_relations(
            objects(ex['input'], univalued=True, diagonal=False, without_bg=True)
        )
        output_objs = serialize_objects_with_relations(
            objects(ex['output'], univalued=True, diagonal=False, without_bg=True)
        )
        object_pairs = []
        for inp_obj in input_objs:
            matched = None
            for out_obj in output_objs:
                if out_obj['id'] in [p['output_id'] for p in object_pairs]:
                    continue
                if set(inp_obj['colors']) & set(out_obj['colors']):
                    matched = out_obj
                    break
            object_pairs.append({
                "input_id": inp_obj['id'],
                "input": inp_obj,
                "output_id": matched['id'] if matched else None,
                "output": matched
            })
        serialized_examples.append({
            'input': ex['input'],
            'output': ex['output'],
            'objects': object_pairs
        })

    base_messages = [{'role': 'system', 'content': 'You are a JSON-only assistant.'}]
    for idx, ex in enumerate(serialized_examples, start=1):
        prompt = (
            f"Training example {idx} of {len(serialized_examples)}:\n"
            f"Input grid:\n{ex['input']}\n"
            f"Output grid:\n{ex['output']}\n"
            f"Objects:\n{json.dumps(ex['objects'])}\n\n"
        )
        base_messages.append({'role': 'user', 'content': prompt})
        base_messages.append({'role': 'assistant', 'content': 'I have analyzed this example and updated my understanding.'})

    hypotheses = []
    seen_hypotheses = []  # store normalized strings instead of raw set

    # Prepare a temperature schedule that stays inside [0.3, 0.95] and increases gradually
    def temp_for_index(i):
        # spread temperatures across a reasonable band, deterministic
        return max(0.3, min(0.95, 0.5 + 0.4 * (i / max(1, n-1))))

    for i in range(n):
        messages = base_messages.copy()

        # small, harmless variation token appended to user prompt — deterministic but different per i
        variation_token = f"VAR-{i}-{hashlib.sha1(str(i).encode()).hexdigest()[:6]}"

        final_prompt = (
            "Based on all training examples provided, generate exactly 1 general hypothesis "
            "describing how the input grid changes to the output grid. DO NOT MENTION ROTATIONS UNLESS THERE IS SIGNIFICANT DATA TO SUPPORT IT."
            "Consider the objects in the grid and their relationships: "
            "relative positions, alignment, clustering, or other emergent patterns. "
            "Your sub-hypotheses should be hierarchical:\n"
            "  Level 1: Object-level changes (local modifications to individual objects)\n"
            "  Level 2: Group-level patterns (interactions or relationships between objects)\n"
            "  Level 3: Grid-level emergent behavior (overall patterns visible in the full grid)\n"
            f"{'IMPORTANT: Generate a hypothesis that is meaningfully different from previously generated ones. ' if hypotheses else ''}"
            "Avoid repeating simple strategies that failed previously. "
            f"Include this short variation token (ignore it for logic): {variation_token} "
            "Output JSON only with keys {'hypothesis': '...', 'sub_hypotheses': ['...']}."
        )
        messages.append({'role': 'user', 'content': final_prompt})

        hypothesis_generated = False
        last_err = None

        # We'll attempt multiple times with small deterministic temp jitter and disable cache for generation
        for attempt in range(retries):
            try:
                # deterministic jitter so every call is slightly different
                base_temp = temp_for_index(i)
                jitter = ((attempt % 3) - 1) * 0.03  # -0.03, 0.0, +0.03 cycle
                temp = max(0.0, min(1.0, base_temp + jitter))

                # Force no cache when generating new hypotheses to avoid cache collisions
                raw = llm.call(messages, temperature=temp, max_tokens=900, use_cache=False)
                parsed, _ = extract_json_from_text(raw)

                if isinstance(parsed, dict):
                    hyp = str(parsed.get('hypothesis', '')).strip()
                    subs = parsed.get('sub_hypotheses', [])
                    if not hyp:
                        last_err = 'Empty hypothesis'
                        continue
                    if not isinstance(subs, list):
                        subs = []

                    # Simple semantic uniqueness via Jaccard on normalized text
                    text_repr = _normalize_text_for_dup_check(hyp + ' ' + ' '.join(map(str, subs)))
                    is_dup = False
                    for seen in seen_hypotheses:
                        if _jaccard_similarity(seen, text_repr) > 0.88:
                            is_dup = True
                            last_err = 'Semantic duplicate (jaccard > 0.88)'
                            break
                    if is_dup:
                        # If duplicate, attempt another try (don't consume this hypothesis index)
                        continue

                    seen_hypotheses.append(text_repr)
                    hypotheses.append({'hypothesis': hyp, 'sub_hypotheses': [str(s).strip() for s in subs]})
                    hypothesis_generated = True
                    break
                else:
                    last_err = f'Parsed not a dict: {type(parsed)}'
            except Exception as e:
                last_err = str(e)
            # small backoff
            time.sleep(0.25 * (attempt + 1))

        if not hypothesis_generated:
            print(f'Failed to generate hypothesis {i+1} after {retries} retries: {last_err}')
            # continue to next i; we still try to reach n outputs

    return hypotheses


# ------------------ Apply hypothesis with context using objects ------------------
APPLY_WITH_CONTEXT_PROMPT_TEMPLATE = (
    "You are given training examples from an ARC problem and a set of generated hypotheses.\n"
    "Now, given a new input grid (held-out example), apply each hypothesis to produce a predicted output.\n"
    "IN ALL ANSWERS USE THE FOLLOWING COLOUR MAPPING: {colour_dict}"
    "**CONSTRAINTS**\n"
    "- Preserve grid dimensions.\n"
    "- Use the transformations described in the hypothesis, informed by the training examples.\n"
    "- If uncertain, pick the most consistent transformation seen in training.\n"
    "- WHEN GENERATING THE OUTPUT GRID, ONLY APPLY THE TRANSFORMATION TO OBJECTS SEEN IN THE INPUT GRID."
    "\n"
    "Hypothesis: {hypothesis}\n"
    "Sub-hypotheses: {substeps_json}\n"
    "Held-out input grid: {input_json}\n"
    "Held-out input grids objects: {heldout_objects}\n"
    "- Output exactly as JSON 2D array of integers, where each integer corresponds to a color ID (see mapping below).\n"
    "- Never output color names, only integers.\n"
    "- Color mapping: {heldout_objects}\n"
)

def apply_hypothesis_with_context(llm: LLMClient, hypothesis: Dict[str, Any],
                                  train_examples: List[Dict[str, Any]],
                                  input_grid: List[List[int]], retries: int = 3) -> Optional[List[List[int]]]:
    # Use objects() to extract objects from held-out input
    heldout_objs = objects(input_grid, univalued=True, diagonal=False, without_bg=True)
    heldout_objects = serialize_objects_for_gpt(heldout_objs)

    prompt = APPLY_WITH_CONTEXT_PROMPT_TEMPLATE.format(
        hypothesis=hypothesis.get('hypothesis', ''),
        substeps_json=json.dumps(hypothesis.get('sub_hypotheses', [])),
        input_json=json.dumps(input_grid),
        heldout_objects=json.dumps(heldout_objects),
        colour_dict = json.dumps(COLOR_NAMES)
    )

    messages = [
        {'role': 'system', 'content': 'Return ONLY a JSON array of integers, no code fences, no text.'},
        {'role': 'user', 'content': prompt}
    ]

    last_err = None
    R_in = len(input_grid)
    C_in = len(input_grid[0]) if R_in > 0 else 0

    for attempt in range(retries):
        try:
            raw = llm.call(messages, temperature=0.0, max_tokens=1200)
            if not raw:
                last_err = "Empty response from LLM"
                continue

            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()

            parsed, _ = extract_json_from_text(raw)
            if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                # Coerce all cells to integers (default to 0 if invalid)
                parsed_int = []
                for row in parsed:
                    new_row = []
                    for cell in row:
                        try:
                            new_row.append(int(cell))
                        except Exception:
                            new_row.append(0)
                    parsed_int.append(new_row)

                R_out = len(parsed_int)
                C_out = len(parsed_int[0]) if R_out > 0 else 0

                if R_out != R_in or C_out != C_in:
                    # Auto-correct dimensions by cropping/padding
                    fixed = [r[:C_in] + [0] * (C_in - len(r)) for r in parsed_int[:R_in]]
                    while len(fixed) < R_in:
                        fixed.append([0] * C_in)
                    parsed_int = fixed

                return parsed_int
            else:
                last_err = 'Parsed output invalid or not a 2D list'
        except Exception as e:
            last_err = str(e)
        if last_err:
            logger.debug('Attempt %d failed: %s', attempt + 1, last_err)
            time.sleep(0.5 * (attempt + 1))
    logger.warning('Failed to apply hypothesis with context: %s', last_err)
    return None


# ------------------ Evaluation and scoring ------------------
def normalize_objects(obj_list) -> List[set]:
    result = []
    try:
        for element in obj_list:
            if isinstance(element, set):
                result.append(element)
            elif isinstance(element, list):
                s = set()
                for p in element:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        s.add((int(p[0]), int(p[1])))
                if s:
                    result.append(s)
    except Exception:
        return []
    return result

def evaluate_hypothesis_on_problem(llm: LLMClient, hypothesis: Dict[str, Any], problem: Dict[str, Any]) -> Dict[str, Any]:
    train = problem.get('train', [])
    total = len(train)
    per_example = []
    grid_accs = []

    context_examples = [{'input': ex['input'], 'output': ex['output']} for ex in train]

    for ex in train:
        input_grid = ex['input']
        expected = ex['output']
        predicted = apply_hypothesis_with_context(llm, hypothesis, context_examples, input_grid, retries=3)

        example_result = {
            'input': input_grid,
            'expected': expected,
            'predicted': predicted,
            'status': 'ok' if predicted is not None else 'invalid_prediction'
        }

        if predicted is None:
            per_example.append(example_result)
            continue

        R = len(expected)
        C = len(expected[0]) if R > 0 else 0

        # ✅ strict cell-by-cell comparison (including zeros)
        total_cells = R * C
        correct_cells = sum(
            1 for r in range(R) for c in range(C)
            if predicted[r][c] == expected[r][c]
        )
        grid_accuracy = correct_cells / max(1, total_cells)

        grid_accs.append(grid_accuracy)
        example_result.update({
            'cell_accuracy': grid_accuracy,
            'equal': (predicted == expected)
        })
        per_example.append(example_result)

    avg_cell_accuracy = sum(grid_accs) / max(1, len(grid_accs))
    return {
        'hypothesis': hypothesis,
        'accuracy': avg_cell_accuracy,
        'per_example': per_example
    }


# ------------------ Simplicity & scoring ------------------
def simplicity_score(hypothesis: Dict[str, Any], encoder, min_tokens=1, max_tokens=200) -> float:
    text = hypothesis.get('hypothesis', '') + ' ' + ' '.join(hypothesis.get('sub_hypotheses', []))
    tokens = encoder.encode(text) if encoder else text.split()
    L = len(tokens)
    L = max(min_tokens, min(L, max_tokens))
    score = 1.0 - (L - min_tokens) / (max_tokens - min_tokens)
    return max(0.0, min(1.0, score))

# ------------------ Weighted matrix ------------------
def build_weighted_matrix(predicted_grids_by_hyp: List[Optional[List[List[int]]]],
                          solomonoff_scores: List[float],
                          colors: List[int] = list(range(10))) -> Dict[str, Any]:
    H = len(predicted_grids_by_hyp)
    grid_shape = None
    for g in predicted_grids_by_hyp:
        if g is not None:
            grid_shape = (len(g), len(g[0]) if g and len(g[0]) > 0 else 0)
            break
    if grid_shape is None:
        raise ValueError("All predicted grids are None; cannot build weighted matrix.")
    R, C = grid_shape
    valid_indices = [i for i, g in enumerate(predicted_grids_by_hyp) if g is not None]

    if not valid_indices:
        per_cell_probs = [[{col: 1.0/len(colors) for col in colors} for _ in range(C)] for _ in range(R)]
        argmax_grid = [[min(colors) for _ in range(C)] for _ in range(R)]
        return {'per_cell_probs': per_cell_probs, 'argmax_grid': argmax_grid, 'colors': colors}

    weights = [solomonoff_scores[i] if i in valid_indices else 0.0 for i in range(H)]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        uniform = 1.0 / len(valid_indices)
        weights = [uniform if i in valid_indices else 0.0 for i in range(H)]
        weight_sum = 1.0

    accum = [[{col: 0.0 for col in colors} for _ in range(C)] for _ in range(R)]
    for h_idx, grid in enumerate(predicted_grids_by_hyp):
        w = weights[h_idx] if h_idx < len(weights) else 0.0
        if w <= 0 or grid is None:
            continue
        for r in range(R):
            for c in range(C):
                val = grid[r][c]
                if val not in colors:
                    val = min(max(val, min(colors)), max(colors))
                accum[r][c][val] += w

    per_cell_probs = [[{} for _ in range(C)] for _ in range(R)]
    argmax_grid = [[0 for _ in range(C)] for _ in range(R)]
    for r in range(R):
        for c in range(C):
            cell_counts = accum[r][c]
            s = sum(cell_counts.values())
            probs = {col: cell_counts[col] / s for col in colors} if s > 0 else {col: 1.0/len(colors) for col in colors}
            per_cell_probs[r][c] = probs
            max_p = max(probs.values())
            best_cols = [col for col, p in probs.items() if p == max_p]
            argmax_grid[r][c] = min(best_cols)

    return {'per_cell_probs': per_cell_probs, 'argmax_grid': argmax_grid, 'colors': colors}

# ------------------ Main driver (modified to do leave-one-out eval) ------------------
def run(problem_file: str, out_folder: str, n_hyp: int = 20, model: str = 'gpt-4', eval_idx: Optional[int] = None) -> None:
    """
    eval_idx: zero-based index of the example to hold out for evaluation.
              If None, defaults to the last training example.
    """
    api_key = 
    llm = LLMClient(api_key=api_key, model=model, cache_folder=os.path.join(out_folder, '.llm_cache'))

    encoder = None
    if tiktoken:
        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                encoder = tiktoken.get_encoding('cl100k_base')
            except Exception:
                encoder = None

    with open(problem_file, 'r', encoding='utf-8') as f:
        problem = json.load(f)

    problem_id = problem.get('problem_id', Path(problem_file).stem)
    save_dir = Path(out_folder) / problem_id
    save_dir.mkdir(parents=True, exist_ok=True)

    full_train = problem.get('train', [])
    if not full_train:
        logger.error('Problem has no training examples. Aborting.')
        return

    if eval_idx is None:
        eval_idx = len(full_train) - 1
    if eval_idx < 0 or eval_idx >= len(full_train):
        logger.error('eval_idx %s is out of range (0..%d). Aborting.', eval_idx, len(full_train)-1)
        return

    # Split train vs eval example (leave-one-out)
    train_examples = [ex for i, ex in enumerate(full_train) if i != eval_idx]
    eval_example = full_train[eval_idx]

    logger.info('Using %d examples for training and example %d for evaluation.', len(train_examples), eval_idx + 1)

    # make a temporary problem object that contains only the reduced training set
    temp_problem = dict(problem)
    temp_problem['train'] = train_examples

    logger.info('Generating %d hypotheses from GPT based on the training set (excluding example %d)...', n_hyp, eval_idx + 1)
    hypotheses = generate_hypotheses_gpt(llm, temp_problem, n=n_hyp, retries=4)
    if not hypotheses:
        logger.error('No hypotheses generated by GPT. Aborting.')
        return

    analyses = []
    predicted_grids_train = []  # predicted grids for each hypothesis on each training example
    total_hypotheses = len(hypotheses)

    # Evaluate hypotheses only on the training set to compute solomonoff scores
    for i, hyp in enumerate(hypotheses):
        logger.info('Evaluating hypothesis %d/%d on training set: %s', i + 1, total_hypotheses, hyp.get('hypothesis', '')[:80])
        analysis = evaluate_hypothesis_on_problem(llm, hyp, temp_problem)
        simp = simplicity_score(hyp, encoder)
        sol_score = simp * analysis['accuracy']
        analysis.update({'simplicity': simp, 'solomonoff_score': sol_score})
        analyses.append(analysis)
        predicted_grids_train.append([ex['predicted'] for ex in analysis['per_example']])

    # Now: apply all generated hypotheses onto the evaluation example input (DO NOT recompute solomonoff scores)
    logger.info('Applying all hypotheses to evaluation example (index %d) input. (No re-scoring.)', eval_idx + 1)
    eval_input = eval_example['input']
    per_hyp_eval_predictions: List[Optional[List[List[int]]]] = []
    for i, hyp in enumerate(hypotheses):
        logger.info('Applying hypothesis %d/%d to eval example', i + 1, total_hypotheses)
        pred = apply_hypothesis_with_context(llm, hyp, train_examples, eval_input, retries=3)
        if pred is None:
            # Mark as None (no prediction). build_weighted_matrix will skip these and renormalize weights.
            logger.debug('Hypothesis %d produced no valid prediction on eval example; marking as None.', i + 1)
            per_hyp_eval_predictions.append(None)
        else:
            per_hyp_eval_predictions.append(pred)

    # Collect solomonoff scores for weighting (from 'analyses' computed on training set)
    solomonoff_scores = [a.get('solomonoff_score', 0.0) for a in analyses]

    # Build weighted matrix of per-cell probabilities using solomonoff scores
    weighted_result = build_weighted_matrix(per_hyp_eval_predictions, solomonoff_scores, colors=list(range(10)))

    # Prepare outputs and save files
    full_json = {
        'problem_id': problem_id,
        'eval_idx': int(eval_idx),
        'n_hypotheses': total_hypotheses,
        'analyses_on_training_set': analyses,
        'eval_example_input': eval_input,
        'eval_example_expected': eval_example.get('output'),
        'per_hyp_eval_predictions': per_hyp_eval_predictions,
        'solomonoff_scores': solomonoff_scores,
        'weighted_matrix': weighted_result
    }

    full_path = save_dir / 'full_leave_one_out_analysis.json'
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)

    logger.info('Saved full leave-one-out analysis to %s', full_path)

    # Save the argmax weighted grid separately
    solomonoff_path = save_dir / 'solomonoff_eval_argmax.json'
    with open(solomonoff_path, 'w', encoding='utf-8') as f:
        json.dump({'problem_id': problem_id, 'argmax_grid': weighted_result['argmax_grid'],
                   'colors': weighted_result['colors']}, f, indent=2, ensure_ascii=False)

    logger.info('Saved argmax weighted predictions to %s', solomonoff_path)

    # -------------------- Markdown summary --------------------
    md_lines = []
    md_lines.append(f"# 🧮 Leave-One-Out Solomonoff Analysis for `{problem_id}`\n")
    md_lines.append(f"**Held-out example index (1-based):** {eval_idx + 1}\n")

    # ------------------------------------------------------------
    # Hypotheses Ranking with normalized Solomonoff Weights
    # ------------------------------------------------------------
    md_lines.append("\n## 📊 Ranked Hypotheses\n")
    md_lines.append("| Rank | Hypothesis | Simplicity | Train Acc. | Solomonoff Score | Solomonoff Weight |")
    md_lines.append("|------|------------|------------|------------|-----------------|-----------------|")

    # Sort hypotheses by Solomonoff Score
    ranked = sorted(enumerate(analyses), key=lambda x: x[1].get("solomonoff_score", 0.0), reverse=True)

    # Compute total for normalization
    total_score = sum(a.get("solomonoff_score", 0.0) for _, a in ranked)
    total_score = total_score if total_score > 0 else 1.0  # avoid division by zero

    for rank, (orig_idx, a) in enumerate(ranked, start=1):
        hyp_text = a.get("hypothesis", {}).get("hypothesis", "(no text)")
        score = a.get("solomonoff_score", 0.0)
        weight = score / total_score
        md_lines.append(
            f"| {rank} | {hyp_text} "
            f"| {a.get('simplicity',0):.3f} "
            f"| {a.get('accuracy',0):.3f} "
            f"| {score:.3f} "
            f"| {weight:.3f} |"
        )

    # ------------------------------------------------------------
    # Held-out Example
    # ------------------------------------------------------------
    def grid_to_str_safe(grid):
        if grid is None:
            return "(no prediction)"
        return "\n".join(" ".join(str(x) for x in row) for row in grid)

    md_lines.append("\n## 🎯 Held-out Example\n")
    md_lines.append("**Input Grid:**")
    md_lines.append("```")
    md_lines.append("\n".join(" ".join(str(x) for x in row) for row in eval_input))
    md_lines.append("```")

    md_lines.append("**Expected Output Grid:**")
    md_lines.append("```")
    if eval_example.get("output") is not None:
        md_lines.append("\n".join(" ".join(str(x) for x in row) for row in eval_example["output"]))
    else:
        md_lines.append("(none provided)")
    md_lines.append("```")

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    md_lines.append("\n## 🤖 Predictions on Held-out Example\n")
    md_lines.append("| Hypothesis # | Weight | Matches Expected? | Predicted Grid |")
    md_lines.append("|--------------|--------|-------------------|----------------|")

    for i, pred in enumerate(per_hyp_eval_predictions):
        expected = eval_example.get("output")
        matches = "✅" if (expected is not None and pred == expected) else "❌"
        pred_str = grid_to_str_safe(pred)
        md_lines.append(
            f"| {i+1} | {solomonoff_scores[i]:.3f} | {matches} |<br>```"
            f"\n{pred_str}\n```|"
        )

    # ------------------------------------------------------------
    # Aggregated Result
    # ------------------------------------------------------------
    md_lines.append("\n## 🏗️ Aggregated (Solomonoff-weighted) Output\n")
    md_lines.append("**Argmax Grid:**")
    md_lines.append("```")
    md_lines.append("\n".join(" ".join(str(x) for x in row) for row in weighted_result["argmax_grid"]))
    md_lines.append("```")

    # ------------------------------------------------------------
    # Per-cell Probabilities (Top 2)
    # ------------------------------------------------------------
    md_lines.append("\n## 🔎 Per-cell Probabilities (Top 2 Colors per Cell)\n")
    md_lines.append("| Row | Col | Top-1 | Top-2 |")
    md_lines.append("|-----|-----|-------|-------|")

    R = len(weighted_result["per_cell_probs"])
    C = len(weighted_result["per_cell_probs"][0]) if R > 0 else 0
    for r in range(R):
        for c in range(C):
            dist = weighted_result["per_cell_probs"][r][c]
            top2 = sorted(dist.items(), key=lambda kv: -kv[1])[:2]
            t1 = f"{top2[0][0]}:{top2[0][1]:.2f}" if len(top2) > 0 else "-"
            t2 = f"{top2[1][0]}:{top2[1][1]:.2f}" if len(top2) > 1 else "-"
            md_lines.append(f"| {r} | {c} | {t1} | {t2} |")

    # Save markdown
    md_path = save_dir / "leave_one_out_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info('Leave-one-out summary saved to %s', md_path)
    logger.info('Analysis complete. Results saved to %s', save_dir)

# ------------------ CLI ------------------
if __name__ == '__main__':
    run(
        problem_file=r'MINI-ARC\data\MiniARC\connect_the_dots_to_make_a_square_l6aescjwslh5v45v3gf.json',
        out_folder='4',
        n_hyp=6,
        model='gpt-4'
    )
