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
import numpy as np

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

# --- Hypothesis generation ---
def generate_hypotheses_gpt(
    llm,  # LLMClient
    problem: Dict[str, any],
    n: int = 10,
    retries: int = 3
) -> List[Dict[str, any]]:
    
    serialized_examples = []

    # --- Serialize all training examples with relations ---
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

    # --- Build base context ---
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

    # --- Generate hypotheses ---
    hypotheses = []
    seen_hypotheses = set()
    
    for i in range(n):
        messages = base_messages.copy()

        final_prompt = (
            "Based on all training examples provided, generate exactly 1 general hypothesis "
            "describing how the input grid changes to the output grid. "
            "Consider the objects in the grid and their relationships: "
            "relative positions, alignment, clustering, or other emergent patterns. "
            "Your sub-hypotheses should be hierarchical:\n"
            "  Level 1: Object-level changes (local modifications to individual objects)\n"
            "  Level 2: Group-level patterns (interactions or relationships between objects)\n"
            "  Level 3: Grid-level emergent behavior (overall patterns visible in the full grid)\n"
            f"{'IMPORTANT: Generate a hypothesis that is meaningfully different from previously generated ones. ' if hypotheses else ''}"
            "Avoid repeating simple strategies that failed previously. "
            "Output JSON only with keys {'hypothesis': '...', 'sub_hypotheses': ['...']}."
        )

        messages.append({'role': 'user', 'content': final_prompt})

        hypothesis_generated = False
        last_err = None

        for attempt in range(retries):
            try:
                raw = llm.call(messages, temperature=0.8 + 0.1 * i, max_tokens=800)
                parsed, _ = extract_json_from_text(raw)

                if isinstance(parsed, dict):
                    hyp = str(parsed.get('hypothesis', '')).strip()
                    subs = parsed.get('sub_hypotheses', [])
                    if not hyp:
                        last_err = 'Empty hypothesis'
                        continue
                    if not isinstance(subs, list):
                        subs = []

                    # Semantic uniqueness (simple heuristic)
                    text_repr = (hyp + ' ' + ' '.join(map(str, subs))).lower()
                    if text_repr in seen_hypotheses:
                        last_err = 'Duplicate hypothesis generated'
                        continue

                    seen_hypotheses.add(text_repr)
                    hypotheses.append({'hypothesis': hyp, 'sub_hypotheses': [str(s).strip() for s in subs]})
                    hypothesis_generated = True
                    break
                else:
                    last_err = f'Parsed not a dict: {type(parsed)}'
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5 * (attempt + 1))

        if not hypothesis_generated:
            # Log warning and continue
            print(f'Failed to generate hypothesis {i+1} after {retries} retries: {last_err}')

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

# ------------------ BMA Likelihood computation ------------------
def compute_likelihood(true_grid, pred_grid, K=10, epsilon=0.1):
    """
    Computes Bayesian likelihood P(D|M) under a per-cell categorical noise model.
    
    Args:
        true_grid (List[List[int]]): Ground-truth grid.
        pred_grid (List[List[int]]): Hypothesis-predicted grid.
        K (int): Number of possible colors.
        epsilon (float): Error probability.
    
    Returns:
        float: Likelihood P(D|M).
    """
    true_arr = np.array(true_grid)
    pred_arr = np.array(pred_grid)
    
    matches = (true_arr == pred_arr)
    N_correct = np.sum(matches)
    N_total = true_arr.size
    N_wrong = N_total - N_correct
    
    # Log-likelihood for numerical stability
    log_likelihood = (
        N_correct * np.log(1 - epsilon) +
        N_wrong * np.log(epsilon / (K - 1))
    )
    
    return np.exp(log_likelihood)

def compute_bma_weights(likelihoods: List[float], priors: Optional[List[float]] = None) -> List[float]:
    """
    Compute BMA weights: w_i = P(M_i | D) ∝ P(D | M_i) * P(M_i)
    where P(D | M_i) is likelihood and P(M_i) is prior
    """
    if priors is None:
        # Uniform priors
        priors = [1.0] * len(likelihoods)
    
    # Compute posterior probabilities (unnormalized)
    posteriors = [l * p for l, p in zip(likelihoods, priors)]
    
    # Normalize
    total = sum(posteriors)
    if total <= 0:
        # Fallback to uniform weights
        return [1.0 / len(likelihoods)] * len(likelihoods)
    
    weights = [p / total for p in posteriors]
    return weights

# ------------------ Evaluation and scoring ------------------
def evaluate_hypothesis_on_problem_bma(llm: LLMClient, hypothesis: Dict[str, Any], 
                                      problem: Dict[str, Any]) -> Dict[str, Any]:
    """Modified evaluation that computes BMA-relevant metrics"""
    train = problem.get('train', [])
    total = len(train)
    per_example = []
    likelihoods = []
    
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
            likelihoods.append(1e-10)  # Very low likelihood
            per_example.append(example_result)
            continue

        R = len(expected)
        C = len(expected[0]) if R > 0 else 0

        # Compute likelihood for BMA
        likelihood = compute_likelihood(predicted, expected)
        likelihoods.append(likelihood)

        # Also compute accuracy for comparison
        total_cells = R * C
        correct_cells = sum(
            1 for r in range(R) for c in range(C)
            if predicted[r][c] == expected[r][c]
        )
        accuracy = correct_cells / max(1, total_cells)

        example_result.update({
            'cell_accuracy': accuracy,
            'likelihood': likelihood,
            'equal': (predicted == expected)
        })
        per_example.append(example_result)

    # Compute average metrics
    avg_likelihood = np.mean(likelihoods) if likelihoods else 1e-10
    avg_accuracy = np.mean([ex.get('cell_accuracy', 0) for ex in per_example if ex.get('cell_accuracy') is not None])
    
    return {
        'hypothesis': hypothesis,
        'avg_likelihood': avg_likelihood,
        'likelihoods': likelihoods,
        'accuracy': avg_accuracy,
        'per_example': per_example
    }

# ------------------ BMA weighted matrix ------------------
def build_bma_weighted_matrix(predicted_grids_by_hyp: List[Optional[List[List[int]]]],
                             bma_weights: List[float],
                             colors: List[int] = list(range(10))) -> Dict[str, Any]:
    """
    Build weighted matrix using BMA weights instead of Solomonoff scores.
    """
    # H = total number of hypotheses
    H = len(predicted_grids_by_hyp)

    # Determine the shape of the grids
    grid_shape = None
    for g in predicted_grids_by_hyp:
        if g is not None:
            grid_shape = (len(g), len(g[0]) if g and len(g[0]) > 0 else 0)
            break
    if grid_shape is None:
        raise ValueError("All predicted grids are None; cannot build weighted matrix.")
    R, C = grid_shape

    # Indices of hypotheses that actually have predictions
    valid_indices = [i for i, g in enumerate(predicted_grids_by_hyp) if g is not None]

    # If no valid hypotheses, return uniform probabilities
    if not valid_indices:
        per_cell_probs = [[{col: 1.0/len(colors) for col in colors} for _ in range(C)] for _ in range(R)]
        argmax_grid = [[min(colors) for _ in range(C)] for _ in range(R)]
        return {'per_cell_probs': per_cell_probs, 'argmax_grid': argmax_grid, 'colors': colors}

    # Use BMA weights (already normalized)
    weights = [bma_weights[i] if i < len(bma_weights) else 0.0 for i in range(H)]

    # Accumulate weighted votes for each cell
    accum = [[{col: 0.0 for col in colors} for _ in range(C)] for _ in range(R)]
    for h_idx, grid in enumerate(predicted_grids_by_hyp):
        w = weights[h_idx] if h_idx < len(weights) else 0.0
        if w <= 0 or grid is None:
            continue
        for r in range(R):
            for c in range(C):
                val = grid[r][c]
                # Clip value to allowed colors if needed
                if val not in colors:
                    val = min(max(val, min(colors)), max(colors))
                accum[r][c][val] += w

    # Convert to probabilities and find argmax
    per_cell_probs = [[{} for _ in range(C)] for _ in range(R)]
    argmax_grid = [[0 for _ in range(C)] for _ in range(R)]
    for r in range(R):
        for c in range(C):
            cell_counts = accum[r][c]
            s = sum(cell_counts.values())
            probs = {col: cell_counts[col] / s for col in colors} if s > 0 else {col: 1.0/len(colors) for col in colors}
            per_cell_probs[r][c] = probs

            # argmax grid: choose color with maximum probability
            max_p = max(probs.values())
            best_cols = [col for col, p in probs.items() if p == max_p]
            argmax_grid[r][c] = min(best_cols)  # break ties by choosing smallest

    return {'per_cell_probs': per_cell_probs, 'argmax_grid': argmax_grid, 'colors': colors}

# ------------------ Main driver (BMA version) ------------------
def run(problem_file: str, out_folder: str, n_hyp: int = 20, model: str = 'gpt-4', eval_idx: Optional[int] = None) -> None:
    """
    BMA version: Uses Bayesian Model Averaging instead of Solomonoff approach
    """
    api_key = 
    llm = LLMClient(api_key=api_key, model=model, cache_folder=os.path.join(out_folder, '.llm_cache'))

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

    # Make temporary problem object for hypothesis generation
    temp_problem = dict(problem)
    temp_problem['train'] = train_examples

    logger.info('Generating %d hypotheses from GPT...', n_hyp)
    hypotheses = generate_hypotheses_gpt(llm, temp_problem, n=n_hyp, retries=4)
    if not hypotheses:
        logger.error('No hypotheses generated by GPT. Aborting.')
        return

    # Evaluate hypotheses on training set to compute BMA weights
    logger.info('Evaluating hypotheses on training set for BMA weight computation...')
    analyses = []
    all_likelihoods = []
    
    for i, hyp in enumerate(hypotheses):
        logger.info('Evaluating hypothesis %d/%d: %s', i + 1, len(hypotheses), hyp.get('hypothesis', '')[:80])
        analysis = evaluate_hypothesis_on_problem_bma(llm, hyp, temp_problem)
        analyses.append(analysis)
        all_likelihoods.append(analysis['avg_likelihood'])

    # Compute BMA weights using likelihoods from training set
    logger.info('Computing BMA weights...')
    bma_weights = compute_bma_weights(all_likelihoods)
    
    # Apply hypotheses to evaluation example
    logger.info('Applying hypotheses to evaluation example...')
    eval_input = eval_example['input']
    per_hyp_eval_predictions: List[Optional[List[List[int]]]] = []
    
    for i, hyp in enumerate(hypotheses):
        logger.info('Applying hypothesis %d/%d to eval example', i + 1, len(hypotheses))
        pred = apply_hypothesis_with_context(llm, hyp, train_examples, eval_input, retries=3)
        per_hyp_eval_predictions.append(pred)

    # Build BMA weighted matrix
    logger.info('Building BMA weighted prediction matrix...')
    weighted_result = build_bma_weighted_matrix(per_hyp_eval_predictions, bma_weights, colors=list(range(10)))

    # Prepare and save results
    full_json = {
        'problem_id': problem_id,
        'eval_idx': int(eval_idx),
        'method': 'BMA',  # Changed from Solomonoff
        'n_hypotheses': len(hypotheses),
        'analyses_on_training_set': analyses,
        'bma_weights': bma_weights,  # Changed from solomonoff_scores
        'eval_example_input': eval_input,
        'eval_example_expected': eval_example.get('output'),
        'per_hyp_eval_predictions': per_hyp_eval_predictions,
        'weighted_matrix': weighted_result
    }

    full_path = save_dir / 'full_bma_leave_one_out_analysis.json'
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    logger.info('Saved full BMA analysis to %s', full_path)

    # Save argmax grid
    bma_path = save_dir / 'bma_eval_argmax.json'
    with open(bma_path, 'w', encoding='utf-8') as f:
        json.dump({'problem_id': problem_id, 'argmax_grid': weighted_result['argmax_grid'],
                   'colors': weighted_result['colors']}, f, indent=2, ensure_ascii=False)
    logger.info('Saved BMA argmax predictions to %s', bma_path)

    # Generate markdown summary
    md_lines = []
    md_lines.append(f"# 🧮 Leave-One-Out BMA Analysis for `{problem_id}`\n")
    md_lines.append(f"**Method:** Bayesian Model Averaging\n")
    md_lines.append(f"**Held-out example index (1-based):** {eval_idx + 1}\n")

    # ------------------------------------------------------------
    # Hypotheses Ranking by BMA weights
    # ------------------------------------------------------------
    md_lines.append("\n## 📊 Ranked Hypotheses (by BMA weight)\n")
    md_lines.append("| Rank | Hypothesis | Avg Likelihood | Train Acc. | BMA Weight |")
    md_lines.append("|------|------------|----------------|------------|------------|")

    ranked = sorted(enumerate(analyses), key=lambda x: bma_weights[x[0]], reverse=True)
    for rank, (orig_idx, a) in enumerate(ranked, start=1):
        hyp = a["hypothesis"]
        md_lines.append(
            f"| {rank} | {hyp.get('hypothesis','(no text)')[:60]}... "
            f"| {a.get('avg_likelihood',0):.6f} "
            f"| {a.get('accuracy',0):.3f} "
            f"| {bma_weights[orig_idx]:.6f} |"
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
    md_lines.append("\n## 🤖 BMA-Weighted Predictions on Held-out Example\n")
    md_lines.append("| Hypothesis # | BMA Weight | Matches Expected? | Predicted Grid |")
    md_lines.append("|--------------|------------|-------------------|----------------|")

    for i, pred in enumerate(per_hyp_eval_predictions):
        expected = eval_example.get("output")
        matches = "✅" if (expected is not None and pred == expected) else "❌"
        pred_str = grid_to_str_safe(pred)
        weight = bma_weights[i] if i < len(bma_weights) else 0.0
        md_lines.append(
            f"| {i+1} | {weight:.6f} | {matches} |<br>```"
            f"\n{pred_str}\n```|"
        )

    # ------------------------------------------------------------
    # Aggregated Result
    # ------------------------------------------------------------
    md_lines.append("\n## 🏗️ Aggregated (BMA-weighted) Output\n")
    md_lines.append("**Argmax Grid:**")
    md_lines.append("```")
    md_lines.append("\n".join(" ".join(str(x) for x in row) for row in weighted_result["argmax_grid"]))
    md_lines.append("```")

    # Check if BMA prediction matches expected output
    expected_output = eval_example.get("output")
    if expected_output is not None:
        bma_matches = weighted_result["argmax_grid"] == expected_output
        md_lines.append(f"**BMA Prediction Accuracy:** {'✅ Perfect match' if bma_matches else '❌ Does not match expected output'}")

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
            t1 = f"{top2[0][0]}:{top2[0][1]:.3f}" if len(top2) > 0 else "-"
            t2 = f"{top2[1][0]}:{top2[1][1]:.3f}" if len(top2) > 1 else "-"
            md_lines.append(f"| {r} | {c} | {t1} | {t2} |")

    # ------------------------------------------------------------
    # BMA Method Details
    # ------------------------------------------------------------
    md_lines.append("\n## 📈 BMA Method Details\n")
    md_lines.append("**Bayesian Model Averaging (BMA)** combines predictions from multiple models by:")
    md_lines.append("1. Computing likelihood P(data|model) for each hypothesis on training data")
    md_lines.append("2. Converting to posterior weights: w_i ∝ P(data|M_i) × P(M_i)")
    md_lines.append("3. Weighting predictions: P(output|input) = Σ w_i × P(output|input,M_i)")
    md_lines.append(f"\n**Total BMA weight sum:** {sum(bma_weights):.6f}")
    md_lines.append(f"**Number of valid predictions:** {len([p for p in per_hyp_eval_predictions if p is not None])}/{len(per_hyp_eval_predictions)}")

    # Save markdown
    md_path = save_dir / "bma_leave_one_out_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info('BMA summary saved to %s', md_path)
    logger.info('BMA analysis complete. Results saved to %s', save_dir)

# ------------------ Additional utility functions for BMA ------------------
def compare_methods(solomonoff_results: Dict, bma_results: Dict) -> Dict[str, Any]:
    """
    Compare Solomonoff vs BMA results if both are available
    """
    comparison = {
        'solomonoff_argmax': solomonoff_results.get('weighted_matrix', {}).get('argmax_grid'),
        'bma_argmax': bma_results.get('weighted_matrix', {}).get('argmax_grid'),
        'methods_agree': None,
        'expected_output': bma_results.get('eval_example_expected')
    }
    
    if comparison['solomonoff_argmax'] and comparison['bma_argmax']:
        comparison['methods_agree'] = comparison['solomonoff_argmax'] == comparison['bma_argmax']
    
    return comparison

def run_from_solomonoff_file(solomonoff_file: str, out_folder: str, noise_std: float = 0.1) -> None:
    """
    Robust BMA-from-file runner.
    - Reads a Solomonoff leave-one-out JSON.
    - Extracts training-set likelihoods (robust to missing/zero fields).
    - Computes BMA weights from training likelihoods.
    - Applies weights to held-out predictions and saves JSON + markdown summary.
    """
    # Load file
    with open(solomonoff_file, "r", encoding="utf-8") as f:
        solomonoff_results = json.load(f)

    problem_id = solomonoff_results.get("problem_id", Path(solomonoff_file).stem)
    save_dir = Path(out_folder) / problem_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Try a few likely keys for stored training analyses
    analyses_train = solomonoff_results.get("analyses_on_training_set") \
                    or solomonoff_results.get("analyses") \
                    or solomonoff_results.get("analyses_train") \
                    or solomonoff_results.get("train_analyses") \
                    or []

    per_hyp_eval_predictions = solomonoff_results.get("per_hyp_eval_predictions", [])
    eval_input = solomonoff_results.get("eval_example_input")
    eval_expected = solomonoff_results.get("eval_example_expected")
    eval_idx = solomonoff_results.get("eval_idx", None)

    if not analyses_train:
        logger.error("No training analyses found in Solomonoff file (checked several keys). Aborting.")
        return
    if eval_expected is None:
        logger.error("No expected output found in Solomonoff file. Aborting.")
        return

    # Compute a robust per-hypothesis training likelihood to use for BMA
    training_likelihoods = []
    enriched_analyses = []  # copies of analyses with an added 'bma_avg_likelihood_used'
    for i, a in enumerate(analyses_train):
        # Helper to safely pull numeric fields
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        avg = None
        # Common places to look
        if isinstance(a, dict):
            avg = _num(a.get("avg_likelihood"))
            if avg is None or avg == 0:
                # Try 'likelihoods' list
                likelihoods_list = a.get("likelihoods") or a.get("likelihood_list") or a.get("train_likelihoods")
                if isinstance(likelihoods_list, (list, tuple)) and len(likelihoods_list) > 0:
                    try:
                        avg = float(np.mean([_num(x) or 0.0 for x in likelihoods_list]))
                    except Exception:
                        avg = None

            if (avg is None or avg == 0) and "likelihood" in a:
                avg = _num(a.get("likelihood"))

            # Some versions may store a single scalar under other names
            if (avg is None or avg == 0):
                for alt in ("avg_lik", "avgLikelihood", "train_avg_likelihood", "train_likelihood"):
                    if alt in a:
                        avg = _num(a.get(alt))
                        if avg is not None:
                            break

            # If we still don't have a positive avg, fall back to accuracy -> convert to a likelihood
            if avg is None or avg <= 0:
                acc = a.get("accuracy") or a.get("train_acc") or a.get("train_accuracy") or a.get("acc") or 0.0
                try:
                    acc = float(acc)
                except Exception:
                    acc = 0.0
                # Use the same transformation as compute_likelihood (approx): exp(accuracy / noise_std)
                avg = float(np.exp(acc / noise_std))
                logger.debug("Hypothesis %d: no avg_likelihood found; using transformed accuracy->likelihood (acc=%.3f -> lik=%.6f).", i+1, acc, avg)
        else:
            # If analysis isn't a dict (unexpected), use default fallback
            avg = 1e-10
            logger.debug("Hypothesis %d: unexpected analysis format, falling back to tiny likelihood.", i+1)

        # Safety floor to avoid zeros
        if avg is None or not np.isfinite(avg) or avg <= 0:
            avg = 1e-10
            logger.debug("Hypothesis %d: forced minimal likelihood fallback.", i+1)

        training_likelihoods.append(avg)

        # enrich the analysis entry for saving & printing
        copy_a = dict(a) if isinstance(a, dict) else {"raw": a}
        copy_a["bma_avg_likelihood_used"] = avg
        # ensure accuracy present for display
        if "accuracy" not in copy_a:
            copy_a["accuracy"] = a.get("accuracy") if isinstance(a, dict) else None
        enriched_analyses.append(copy_a)

    # Compute BMA weights using the robust training likelihoods
    bma_weights = compute_bma_weights(training_likelihoods)

    # Build BMA weighted matrix for the held-out predictions
    weighted_result = build_bma_weighted_matrix(per_hyp_eval_predictions, bma_weights, colors=list(range(10)))

    # Save JSON results (full)
    full_json = {
        "problem_id": problem_id,
        "method": "BMA",
        "eval_idx": int(eval_idx) if eval_idx is not None else None,
        "eval_example_input": eval_input,
        "eval_example_expected": eval_expected,
        "per_hyp_eval_predictions": per_hyp_eval_predictions,
        "bma_weights": bma_weights,
        "weighted_matrix": weighted_result,
        "analyses_on_training_set": enriched_analyses,
        "training_likelihoods_used": training_likelihoods
    }
    full_path = save_dir / "full_bma_leave_one_out_analysis.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    logger.info("Saved full BMA analysis to %s", full_path)

    # Save argmax grid separately
    bma_path = save_dir / "bma_eval_argmax.json"
    with open(bma_path, "w", encoding="utf-8") as f:
        json.dump({
            "problem_id": problem_id,
            "argmax_grid": weighted_result["argmax_grid"],
            "colors": weighted_result["colors"],
        }, f, indent=2, ensure_ascii=False)
    logger.info("Saved BMA argmax predictions to %s", bma_path)

    # --- Markdown summary (formatted like your example) ---
    md_lines = []
    md_lines.append(f"# 🧮 Leave-One-Out BMA Analysis for `{problem_id}`\n")
    md_lines.append("**Method:** Bayesian Model Averaging\n")
    if eval_idx is not None:
        md_lines.append(f"**Held-out example index (1-based):** {int(eval_idx) + 1}\n")
    else:
        md_lines.append("\n")

    # Ranked hypotheses table
    md_lines.append("\n## 📊 Ranked Hypotheses (by BMA weight)\n")
    md_lines.append("| Rank | Hypothesis | Avg Likelihood | Train Acc. | BMA Weight |")
    md_lines.append("|------|------------|----------------|------------|------------|")
    ranked = sorted(enumerate(enriched_analyses), key=lambda x: bma_weights[x[0]], reverse=True)
    for rank, (orig_idx, a) in enumerate(ranked, start=1):
        # extract readable hypothesis text
        hyp_val = a.get("hypothesis", None)
        if isinstance(hyp_val, dict):
            hyp_text = hyp_val.get("hypothesis") or hyp_val.get("text") or str(hyp_val)
        else:
            hyp_text = str(hyp_val) if hyp_val is not None else f"Hypothesis {orig_idx+1}"
        avg_lik = a.get("bma_avg_likelihood_used", 0.0)
        acc = a.get("accuracy", a.get("train_acc", 0.0)) or 0.0
        md_lines.append(
            f"| {rank} | {hyp_text[:60]}{'...' if len(hyp_text) > 60 else ''} "
            f"| {avg_lik:.6f} "
            f"| {float(acc):.3f} "
            f"| {bma_weights[orig_idx]:.6f} |"
        )

    # Held-out input & expected
    def grid_to_str_safe(grid):
        if grid is None:
            return "(no prediction)"
        return "\n".join(" ".join(str(x) for x in row) for row in grid)

    md_lines.append("\n## 🎯 Held-out Example\n")
    md_lines.append("**Input Grid:**")
    md_lines.append("```")
    md_lines.append(grid_to_str_safe(eval_input))
    md_lines.append("```")

    md_lines.append("**Expected Output Grid:**")
    md_lines.append("```")
    md_lines.append(grid_to_str_safe(eval_expected))
    md_lines.append("```")

    # Predictions table (held-out)
    md_lines.append("\n## 🤖 BMA-Weighted Predictions on Held-out Example\n")
    md_lines.append("| Hypothesis # | BMA Weight | Matches Expected? | Predicted Grid |")
    md_lines.append("|--------------|------------|-------------------|----------------|")
    for i, pred in enumerate(per_hyp_eval_predictions):
        matches = "✅" if pred == eval_expected else "❌"
        pred_str = grid_to_str_safe(pred)
        md_lines.append(
            f"| {i+1} | {bma_weights[i]:.6f} | {matches} |<br>```"
            f"\n{pred_str}\n```|"
        )

    # Aggregated argmax
    md_lines.append("\n## 🏗️ Aggregated (BMA-weighted) Output\n")
    md_lines.append("**Argmax Grid:**")
    md_lines.append("```")
    md_lines.append(grid_to_str_safe(weighted_result["argmax_grid"]))
    md_lines.append("```")
    matches = weighted_result["argmax_grid"] == eval_expected
    md_lines.append(f"**BMA Prediction Accuracy:** {'✅ Perfect match' if matches else '❌ Does not match expected output'}")

    # Per-cell probabilities (Top 2)
    md_lines.append("\n## 🔎 Per-cell Probabilities (Top 2 Colors per Cell)\n")
    md_lines.append("| Row | Col | Top-1 | Top-2 |")
    md_lines.append("|-----|-----|-------|-------|")
    R = len(weighted_result["per_cell_probs"])
    C = len(weighted_result["per_cell_probs"][0]) if R > 0 else 0
    for r in range(R):
        for c in range(C):
            dist = weighted_result["per_cell_probs"][r][c]
            top2 = sorted(dist.items(), key=lambda kv: -kv[1])[:2]
            t1 = f"{top2[0][0]}:{top2[0][1]:.3f}" if len(top2) > 0 else "-"
            t2 = f"{top2[1][0]}:{top2[1][1]:.3f}" if len(top2) > 1 else "-"
            md_lines.append(f"| {r} | {c} | {t1} | {t2} |")

    # Method details summary
    md_lines.append("\n## 📈 BMA Method Details\n")
    md_lines.append("**Bayesian Model Averaging (BMA)** combines predictions from multiple models by:")
    md_lines.append("1. Computing likelihood P(data|model) for each hypothesis on training data")
    md_lines.append("2. Converting to posterior weights: w_i ∝ P(data|M_i) × P(M_i)")
    md_lines.append("3. Weighting predictions: P(output|input) = Σ w_i × P(output|input,M_i)")
    md_lines.append(f"\n**Total BMA weight sum:** {sum(bma_weights):.6f}")
    md_lines.append(f"**Number of valid predictions:** {len([p for p in per_hyp_eval_predictions if p is not None])}/{len(per_hyp_eval_predictions)}")

    md_path = save_dir / "bma_leave_one_out_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("BMA summary saved to %s", md_path)
    logger.info("BMA analysis complete. Results saved to %s", save_dir)


def analyze_weight_distribution(bma_weights: List[float], solomonoff_scores: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Analyze the distribution of weights between methods
    """
    analysis = {
        'bma_weights': {
            'max': max(bma_weights) if bma_weights else 0,
            'min': min(bma_weights) if bma_weights else 0,
            'mean': np.mean(bma_weights) if bma_weights else 0,
            'std': np.std(bma_weights) if bma_weights else 0,
            'entropy': -sum(w * np.log(w + 1e-10) for w in bma_weights if w > 0)
        }
    }
    
    if solomonoff_scores:
        # Normalize solomonoff scores for comparison
        total_sol = sum(solomonoff_scores)
        norm_sol = [s/total_sol for s in solomonoff_scores] if total_sol > 0 else solomonoff_scores
        
        analysis['solomonoff_weights'] = {
            'max': max(norm_sol),
            'min': min(norm_sol),
            'mean': np.mean(norm_sol),
            'std': np.std(norm_sol),
            'entropy': -sum(w * np.log(w + 1e-10) for w in norm_sol if w > 0)
        }
        
        # Compute correlation between weight assignments
        if len(bma_weights) == len(norm_sol):
            correlation = np.corrcoef(bma_weights, norm_sol)[0, 1] if len(bma_weights) > 1 else 0
            analysis['weight_correlation'] = correlation
    
    return analysis

# ------------------ CLI ------------------
if __name__ == '__main__':
    #run(
    #    problem_file=r'.\MINI-ARC\data\MiniARC\1_3_5th_go_down_2_4th_go_up_l6abuzscmjgg12f0mdc.json',
    #    out_folder=r'bma_results\1',
    #    n_hyp=7,
    #    model='gpt-4'
    #)

    run_from_solomonoff_file(
            solomonoff_file=r"4\connect_the_dots_to_make_a_square_l6aescjwslh5v45v3gf\full_leave_one_out_analysis.json",
            out_folder=r"bma_results\4-1"
        )