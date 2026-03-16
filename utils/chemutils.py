# --------------------------------------------------------------------
# Original code Copyright (c) 2018 Wengong Jin, Regina Barzilay, Tommi Jaakkola
# - Wengong Jin, Regina Barzilay, Tommi Jaakkola. Junction Tree Variational Autoencoder for Molecular Graph Generation. arXiv, 2019.
# Covered by original MIT license
##
# Modifications copyright Nicolas Nemeth 2025 licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

# Please keep at the top
import os
import re
import math
import random
import itertools
import subprocess
import sys
from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import rdFingerprintGenerator, DataStructs, AllChem, Descriptors, QED

from utils.mod_rule import ModRule

# Add Contrib to path for SAScore
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError:
    print("Warning: Could not import sascorer. SAScore will be 10.0 (bad).")
    sascorer = None

RDLogger.DisableLog("rdApp.warning")
from functools import lru_cache


import torch
import mod
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.data import Data
from pretraingnn.loader import mol_to_graph_data_obj_simple
from pretraingnn.model import GNN_graphpred
from IPython.display import display, SVG
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

## Constants
MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000
WILD_STAR = '*'
X_VAR = "X"
MOD_POST_CMD: str = "/mydata/mod-repo/libs/post_mod/bin/mod_post"
##

### METRICS SECTION

def calculate_strain_energy(mol: Chem.Mol) -> float:
    """
    Estimates strain energy using MMFF94.
    Returns: Energy (kcal/mol). Lower is better. 
    Returns 1000.0 if embedding fails (impossible geometry).
    """
    if mol is None: return 1000.0
    mol_h = Chem.AddHs(mol)
    # Try to embed in 3D
    res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if res == -1:
        return 1000.0  # Geometry impossible
    
    # Optimize geometry
    res = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    if res == -1:
        return 1000.0 # Did not converge (highly strained)
        
    # Get Energy
    props = AllChem.MMFFGetMoleculeProperties(mol_h)
    if props is None:
        return 1000.0
    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    return ff.CalcEnergy()

def calculate_lipinski_violations(mol: Chem.Mol, include_rotatable_bonds: bool = True) -> int:
    """
    Calculates Lipinski Rule of 5 violations.
    Note: The 'Rule of 5' refers to the cutoffs (500, 5, 5, 10) being multiples of 5, 
    not the number of rules (which is 4).
    
    1. MW <= 500
    2. LogP <= 5
    3. H-Bond Donors <= 5
    4. H-Bond Acceptors <= 10
    5. Num rotatable bonds <= 10 (Veber's rule)
    
    Optionally includes Veber's rule (Rotatable Bonds <= 10) as a 5th condition.
    """
    if mol is None: return 0
    
    try:
        mol.UpdatePropertyCache(strict=False)
    except:
        pass

    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if Descriptors.NumHDonors(mol) > 5: violations += 1
    if Descriptors.NumHAcceptors(mol) > 10: violations += 1
    
    if include_rotatable_bonds:
        if Descriptors.NumRotatableBonds(mol) > 10: violations += 1
        
    return violations

def calculate_sascore(mol: Chem.Mol) -> float:
    """
    Calculates Synthetic Accessibility Score (SAScore).
    Range: 1 (easy to synthesize) to 10 (very difficult).
    """
    if mol is None: return 10.0
    if sascorer is None: return 10.0
    
    try:
        score = sascorer.calculateScore(mol)
        return score
    except Exception as e:
        print(f"Error calculating SAScore: {e}")
        return 10.0

def calculate_complexity(mol: Chem.Mol) -> float:
    """
    Calculates molecular complexity using BertzCT.
    Lower is generally better for lead-likeness, though it depends on the target.
    """
    if mol is None: return 10000.0
    return Descriptors.BertzCT(mol)

def evaluate_molecule_quality(smiles: str) -> float:
    """
    Composite Score: Higher is Better.
    Combines QED, Strain Energy, Lipinski Violations, SAScore, and Complexity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return -100.0
    
    try:
        Chem.SanitizeMol(mol)
    except:
        return -100.0

    # 1. QED (0 to 1) - Drug-likeness. Higher is better.
    qed_score = QED.qed(mol)
    
    # 2. Strain Energy (Lower is better)
    # We normalize: < 50 kcal/mol is okay. > 100 is bad.
    energy = calculate_strain_energy(mol)
    energy_penalty = min(energy, 200.0) / 200.0 # 0.0 to 1.0 scale roughly
    
    # 3. Rule of 5 (0 to 5). Lower is better.
    ro5_violations = calculate_lipinski_violations(mol)
    ro5_penalty = ro5_violations * 0.2
    
    # 4. SAScore (1 to 10). Lower is better.
    sa_score = calculate_sascore(mol)
    # Normalize 1-10 to roughly 0-1 penalty
    sa_penalty = (sa_score - 1.0) / 9.0 
    
    # 5. Complexity (BertzCT). Lower is better (usually).
    # Typical drug-like molecules have BertzCT < 800-1000.
    complexity = calculate_complexity(mol)
    # Normalize: 0 to 1500 -> 0 to 1
    complexity_penalty = min(complexity, 1500.0) / 1500.0

    # Composite Formula
    # We want to maximize QED and minimize the penalties.
    # Weights can be tuned.
    # Score = QED - w1*Strain - w2*Ro5 - w3*SA - w4*Complexity
    
    final_score = (1.0 * qed_score) \
                  - (1.0 * energy_penalty) \
                  - (0.5 * ro5_penalty) \
                  - (0.5 * sa_penalty) \
                  - (0.2 * complexity_penalty)
    
    return final_score

def np_bits_to_explicit_bv(arr: np.ndarray) -> DataStructs.ExplicitBitVect:
    a = np.asarray(arr).astype(np.bool_).ravel()  # shape (n,)
    n = a.size
    bv = DataStructs.ExplicitBitVect(n)
    # Set on-bits
    on = np.nonzero(a)[0]
    for i in on:
        bv.SetBit(int(i))
    return bv

def tanimoto_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> int:
    if fp_a is None or fp_b is None:
        return 1.0
    # Accept either RDKit bitvectors or numpy arrays
    if isinstance(fp_a, np.ndarray):
        fp_a = np_bits_to_explicit_bv(fp_a)
    if isinstance(fp_b, np.ndarray):
        fp_b = np_bits_to_explicit_bv(fp_b)
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)  # third arg is optional, bool
    return 1.0 - sim

###

### ML SECTION

def ensure_batch(data: Data) -> Data:
    if getattr(data, 'batch', None) is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data

@torch.no_grad()
def get_molecule_embeddings(model: GNN_graphpred, mol: Chem.Mol, device: str='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes as input the GNN model and an rdkit molecule and returns a vector embedding for 
    the whole graph and each node's embeddings, one vector per node, 
    as a node embeddings matrix.
    """
    data = mol_to_graph_data_obj_simple(mol)
    data = ensure_batch(data).to(device)
    out = model(data)
    graph_emb = out["graph"]
    node_emb  = out["node"]
    return graph_emb, node_emb

def _bitvect_to_numpy(bv: Chem.DataStructs.ExplicitBitVect) -> np.ndarray:
    arr = np.zeros((bv.GetNumBits(),), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def _counts_fp_to_numpy(fp_counts: Chem.DataStructs.UIntSparseIntVect, n_bits: int) -> np.ndarray:
    out = np.zeros((n_bits,), dtype=np.float32)
    for idx, val in fp_counts.GetNonzeroElements().items():
        if idx < n_bits:
            out[idx] = float(val)
        # RDKit guarantees indices within range for Morgan, but guard anyway.
    return out

@lru_cache(maxsize=None)
def _cached_morgan_generator(
    radius: int,
    fpSize: int,
    includeChirality: bool,
    useBondTypes: bool,
):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fpSize,
        includeChirality=includeChirality,
        useBondTypes=useBondTypes,
        includeRingMembership=False,
    )


def morgan_counts_from_external_ids(
    mol: Chem.Mol,
    ext_ids: Set[int],
    ext2rd: Dict[int, int],
    radius: int,
    fpSize: int,
    useChirality: bool = False,
    useBondTypes: bool = True,
    return_bits: bool = True,
) -> np.ndarray:
    """
    Centers-restricted Morgan fingerprint (counts by default).
    Returns np.float32 vector of length fpSize (counts or bits).
    """
    if mol is None:
        raise ValueError("mol is None")
    if not ext_ids:
        return np.zeros((fpSize,), dtype=np.float32)

    # Map external -> RDKit atom indices; silently skip missing
    ext2rd_get = ext2rd.get
    centers_set = {ext2rd_get(e) for e in ext_ids}
    centers_set.discard(None)
    if not centers_set:
        return np.zeros((fpSize,), dtype=np.float32)

    # Deduplicate and validate as pure Python ints
    centers = sorted(int(c) for c in centers_set)
    n_atoms = mol.GetNumAtoms()
    for c in centers:
        if not isinstance(c, int):
            raise TypeError(f"Atom index {c} has type {type(c)}; must be built-in int.")
        if c < 0 or c >= n_atoms:
            raise ValueError(f"Atom index {c} out of range (0..{n_atoms-1})")

    gen = _cached_morgan_generator(radius, fpSize, useChirality, useBondTypes)

    if return_bits:
        bv = gen.GetFingerprint(mol, fromAtoms=centers)
        return _bitvect_to_numpy(bv).astype(np.float32, copy=False)

    counts_vect = gen.GetCountFingerprint(mol, fromAtoms=centers)
    return _counts_fp_to_numpy(counts_vect, fpSize)

###

def reconstruct_dataset_from_rules(
        smiles: List[str],
        rules: List[ModRule],
        create_summary: bool=False,
        apply_all_targets: bool=False,
        alpha: Optional[float]=0.15,
        max_iter_space_exploration: int=50,
        max_rule_application_execute_strat: int=10,
        max_heavy_atom_count: Optional[int]=None,
        # replaces all WILDSTARS in non-terminal mols with a heavy atoms
        # randomly sampled from the vocabulary containing all heavy atoms occurring in the input dataset
        make_all_mols_terminal: bool=False,
        original_smiles: Optional[List[str]]=None,
        rule_reapplication_decay: Optional[float]=0.15,
        verbose: bool=False,
        enable_beam_search: bool=False,
        pruning_threshold: float=0.0,
    ) -> Tuple[List[str], int]:
        """
        Traverses through derivation graph holding applied reaction rules and
        products in a breadth-first-search (BFS) manner using `collection.deque` (a queue).

        Reverse each induced rule (X -> substructure) iteratively on the split molecules
        to obtain final SMILES dataset. May require multi-pass expansion if nested X.

        - apply_all_targets (bool): when True enqueue every product generated by a rule application;
            otherwise only the first target is considered for expansion.

        Current placeholder: returns the *split-state* SMILES (keys) without reconstruction.
        """
        inversed_rules: List[mod.Rule] = []
        for rule in rules:
            cached_inverse = getattr(rule, "_cached_inverse_rule", None)
            if cached_inverse is None:
                cached_inverse = rule.modRule.makeInverse()
                setattr(rule, "_cached_inverse_rule", cached_inverse)
            inversed_rules.append(cached_inverse)

        # --- Precompute Vocabulary and Frequencies (Moved up for Beam Search) ---
        vocab: Set[str] = set()
        for r in inversed_rules:
            for v in r.right.vertices:
                lbl = v.stringLabel
                if lbl not in (X_VAR, WILD_STAR, 'H'):
                    vocab.add(lbl)
        
        atom_freqs = {}
        if original_smiles is not None:
            counts = defaultdict(int)
            total_count = 0
            for s in original_smiles:
                mol_tmp = Chem.MolFromSmiles(s)
                if mol_tmp:
                    for a in mol_tmp.GetAtoms():
                        sym = a.GetSymbol()
                        if sym in ( WILD_STAR, X_VAR, 'H' ): 
                            continue
                        counts[sym] += 1
                        total_count += 1
            
            if total_count > 0:
                for sym, c in counts.items():
                    atom_freqs[sym] = c / total_count
                    vocab.add(sym)

        vocab_list: List[str] = list(vocab) if vocab else ['C']
        # -----------------------------------------------------------------------

        # Precompute terminal status for rules
        rule_is_terminal = {}
        for r in inversed_rules:
            is_term = True
            # Check if any vertex in RHS has label X_VAR except context vertices
            context_vertex_ids = { v.id for v in r.context.vertices }
            for v in r.right.vertices:
                if v.id in context_vertex_ids:
                    continue
                if v.stringLabel in (X_VAR, WILD_STAR):
                    is_term = False
                    break
            rule_is_terminal[r] = is_term

        # Precompute heavy atom delta for rules
        rule_heavy_delta = {}
        for r in inversed_rules:
            # Vertices in context are preserved.
            # We assume vertices in context share IDs with left and right.
            context_ids = {v.id for v in r.context.vertices}
            
            removed_heavy = 0
            for v in r.left.vertices:
                if v.id in context_ids:
                    continue
                # Count heavy atoms (not H)
                # X and * are considered heavy
                if v.stringLabel not in ('H',):
                    removed_heavy += 1
            
            added_heavy = 0
            for v in r.right.vertices:
                if v.id in context_ids:
                    continue
                if v.stringLabel not in ('H',):
                    added_heavy += 1
            
            rule_heavy_delta[r] = added_heavy - removed_heavy

        p = mod.DGPrinter()
        p.graphvizPrefix = 'layout = "dot";'
        # initialize all_mols with the starting molecule
        all_mols_redundant: Set[mod.Graph] = { mod.Graph.fromSMILES(smi.replace(WILD_STAR, X_VAR), allowAbstract=True, add=False, printStereoWarnings=False) for smi in smiles }
        all_mols: Set[mod.Graph] = dedup_by_dfs( all_mols_redundant )
        mod.post.summarySection("Build space via apply framework")
        dg = mod.DG(graphDatabase=list(all_mols), labelSettings=mod.LabelSettings(mod.LabelType.Term, mod.LabelRelation.Unification))

        max_rule_applications = dict()
        for r in inversed_rules:
            # num must be the same as `max_steps` in JunctionTree.apply_rule_until_done function
            max_rule_applications[r] = max_rule_application_execute_strat
        currMols: Set[Tuple[mod.Graph, Tuple]] = set()
        for mol in all_mols:
            currMols |= { ( mol, tuple() ) }
        # Use list for random selection to avoid BFS bias and reduce variance
        currMols_list: List[Tuple[mod.Graph, Tuple]] = list(currMols)
        seen = set()
        cnt = 0

        pruned_graphs = set()

        with dg.build() as b:
            while currMols_list and cnt < max_iter_space_exploration:
                # Randomly pick next molecule to process
                # Optimization: Swap with last element and pop to avoid O(N) shift
                rand_idx = random.randrange(len(currMols_list))
                if rand_idx != len(currMols_list) - 1:
                    currMols_list[rand_idx], currMols_list[-1] = currMols_list[-1], currMols_list[rand_idx]
                
                m, rule_app_counts_tuple = currMols_list.pop()
                
                # ignore all nodes in dg where depth reached > 'depth_limit' = 'number of rules applied in total'
                # if dg_depth_limit is not None and sum([ elem[1] for elem in rule_app_counts_tuple ]) > dg_depth_limit:
                #     continue
                rule_app_counts = dict(rule_app_counts_tuple)
                key = (m, rule_app_counts_tuple)
                if key in seen:
                    continue
                seen.add(key)

                # Check heavy atom count constraint
                if max_heavy_atom_count is not None:
                    n_heavy = 0
                    for v in m.vertices:
                        lbl = v.stringLabel
                        if lbl in (X_VAR, WILD_STAR):
                            n_heavy += 1
                        elif lbl != 'H':
                            n_heavy += 1
                    
                    if n_heavy > max_heavy_atom_count:
                        continue

                # Find all applicable rules first
                candidates = []
                for r in inversed_rules:
                    # Check size constraint
                    if max_heavy_atom_count is not None:
                        if n_heavy + rule_heavy_delta[r] > max_heavy_atom_count:
                            continue

                    applied_count = rule_app_counts.get(r, 0)
                    rule_max = max_rule_applications.get(r, 1)
                    if applied_count >= rule_max:
                        continue
                    
                    # Check applicability
                    res = b.apply([ m ], r, verbosity=0)
                    if res:
                        candidates.append((r, res))
                
                if not candidates:
                    cnt += 1
                    continue

                # Calculate probabilities
                weights = []
                if alpha is not None:
                    for r, _ in candidates:
                        xr = 1 if rule_is_terminal[r] else 0
                        w = math.exp(alpha * cnt * xr)
                        weights.append(w)
                else:
                    weights = [1.0] * len(candidates)
                # Sample one rule
                chosen_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                r, res = candidates[chosen_idx]
                
                # Apply the chosen rule
                if not apply_all_targets:
                    # Determine number of repetitions
                    applied_count = rule_app_counts.get(r, 0)
                    rule_max = max_rule_applications.get(r, 1)
                    available = max(1, rule_max - applied_count)
                    
                    # Linear decay weights: 1.0, 0.95, 0.90, ...
                    if rule_reapplication_decay is not None:
                        rep_weights = [max(0.05, 1.0 - rule_reapplication_decay * i) for i in range(available)]
                        num_repeats = random.choices(range(1, available + 1), weights=rep_weights, k=1)[0]
                    else:
                        num_repeats = 1
                    
                    # Randomly select one valid match (derivation) from the possible matches
                    d = random.choice(res)
                    
                    current_graph = None
                    if d.targets:
                        current_graph = list(d.targets)[0].graph
                    
                    # If we successfully applied once, and we want more repeats
                    steps_done = 1
                    
                    current_n_heavy = 0
                    if max_heavy_atom_count is not None:
                        current_n_heavy = n_heavy + rule_heavy_delta[r]

                    # Continue applying if needed
                    while steps_done < num_repeats and current_graph is not None:
                        if max_heavy_atom_count is not None:
                            if current_n_heavy + rule_heavy_delta[r] > max_heavy_atom_count:
                                break

                        res_next = b.apply([current_graph], r, verbosity=0)
                        if not res_next:
                            break
                        # Randomly select one match for the next step as well
                        d_next = random.choice(res_next)
                        if not d_next.targets:
                            break
                        current_graph = list(d_next.targets)[0].graph
                        steps_done += 1
                        
                        if max_heavy_atom_count is not None:
                            current_n_heavy += rule_heavy_delta[r]
                    
                    if current_graph is not None:
                        x_graph = current_graph
                        
                        if enable_beam_search:
                            # Heuristic check using vocabulary filling
                            t_smi = x_graph.smiles.replace(X_VAR, WILD_STAR)
                            t_mol = Chem.MolFromSmiles(t_smi, sanitize=False)
                            
                            should_prune = False
                            if t_mol:
                                # Fill wildcards with realistic atoms
                                t_mol_filled = fill_wildcards_with_vocab(t_mol, vocab_list, atom_freqs)
                                if t_mol_filled:
                                    f_score = calculate_fast_quality_score(t_mol_filled)
                                    if f_score < pruning_threshold:
                                        should_prune = True
                                else:
                                    should_prune = True
                            else:
                                should_prune = True

                            if should_prune:
                                pruned_graphs.add(x_graph)
                                x_graph = None # Prune

                        if x_graph is not None:
                            pruned_graphs.discard(x_graph)
                            updated_rule_app_counts = dict(rule_app_counts)
                            updated_rule_app_counts[r] = rule_app_counts.get(r, 0) + steps_done
                            hashable_rule_app_counts = tuple(sorted(updated_rule_app_counts.items()))
                            item = (x_graph, hashable_rule_app_counts)
                            all_mols.add(x_graph)
                            currMols_list.append(item)
                else:
                    # Randomly select one valid match (derivation) from the possible matches
                    d = random.choice(res)
                    if d.targets:
                        targets = list(d.targets)
                        for target in targets:
                            x_graph = target.graph
                            
                            if enable_beam_search:
                                # Heuristic check using vocabulary filling
                                t_smi = x_graph.smiles.replace(X_VAR, WILD_STAR)
                                t_mol = Chem.MolFromSmiles(t_smi, sanitize=False)
                                
                                should_prune = False
                                if t_mol:
                                    t_mol_filled = fill_wildcards_with_vocab(t_mol, vocab_list, atom_freqs)
                                    if t_mol_filled:
                                        f_score = calculate_fast_quality_score(t_mol_filled)
                                        if f_score < pruning_threshold:
                                            should_prune = True
                                    else:
                                        should_prune = True
                                else:
                                    should_prune = True

                                if should_prune:
                                    pruned_graphs.add(x_graph)
                                    continue

                            pruned_graphs.discard(x_graph)
                            updated_rule_app_counts = dict(rule_app_counts)
                            updated_rule_app_counts[r] = rule_app_counts.get(r, 0) + 1
                            hashable_rule_app_counts = tuple(sorted(updated_rule_app_counts.items()))
                            item = (x_graph, hashable_rule_app_counts)
                            all_mols.add(x_graph)
                            currMols_list.append(item)
                cnt += 1
                if verbose and cnt % 10 == 0:
                    print(f"iter {cnt}")

        if create_summary:
            dg.print(p)
            mod.post.enableInvokeMake()
            mod.post.enableCompileSummary()
            mod.post.flushCommands()
            ### Generate summary.pdf
            os.chdir("/MasterThesis")
            subprocess.run( [ MOD_POST_CMD ] )
        ###
        derived_terminal_graphs: Set[mod.Graph] = { v.graph for v in dg.vertices if v.outDegree == 0 and v.inDegree > 0 }
        derived_terminal_graphs = derived_terminal_graphs - pruned_graphs
        smiles_of_derived_terminal_graphs: List[str] = [ g.smiles.replace(X_VAR, WILD_STAR) for g in derived_terminal_graphs ]

        if not make_all_mols_terminal:
            return smiles_of_derived_terminal_graphs, cnt
        
        # Replace wildcards with random atoms using helper
        for i in range(len(smiles_of_derived_terminal_graphs)):
            smi = smiles_of_derived_terminal_graphs[i]
            if WILD_STAR not in smi:
                continue
            
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                continue
            
            filled_mol = fill_wildcards_with_vocab(mol, vocab_list, atom_freqs)
            if filled_mol:
                try:
                    smiles_of_derived_terminal_graphs[i] = Chem.MolToSmiles(filled_mol)
                except:
                    pass

        return smiles_of_derived_terminal_graphs, cnt


def dedup_by_dfs(graphs: Set[mod.Graph]) -> Set[mod.Graph]:
    """Deduplicates the graphs set such that there are no isomorphic"""
    unique = {}
    for g in graphs:
        key = g.graphDFS
        if key not in unique:
            unique[key] = g
    return set( unique.values() )


def _as_modrule(obj: Any) -> Optional[ModRule]:
    """
    Helper: initiate_random_grammar_rule_induction may return either:
      - a single ModRule (when only_do_once=True and a split happened)
      - a list of ModRule (all accumulated) or empty list
    This normalizes to a single latest ModRule or None.
    """
    if obj is None:
        return None
    if isinstance(obj, ModRule):
        return obj
    if isinstance(obj, (list, tuple)):
        if not obj:
            return None
        last = obj[-1]
        return last if isinstance(last, ModRule) else None
    return None


def modify_smiles(smiles: str, kekulize: bool, obtain2D: bool) -> str:
    """
    Returns the kekulized version of the smiles string.
    Smiles string will remain unchanged if already kekulized.
    And turns it into a 2D smiles string if it has 3D chemistry.
    """
    smiles = smiles.replace(X_VAR, WILD_STAR)
    if not any([kekulize, obtain2D]):
        return smiles
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    if obtain2D:
        Chem.RemoveStereochemistry(mol)
    modified_smiles = Chem.MolToSmiles(mol)
    modified_smiles = modified_smiles.replace(WILD_STAR, X_VAR)
    return modified_smiles

def draw_junction_graph(cliques_dict: Dict[int, List[List[int]]], edges_dict: Dict[int, List[Tuple[int, int]]], fig_size: Tuple[int, int]=(7, 7)) -> None:
    G = nx.Graph()
    cliques_keys = cliques_dict.keys()
    G.add_nodes_from(cliques_keys)
    valid_edges = [(u, v) for u, v in edges_dict.values() if u in cliques_keys and v in cliques_keys]
    G.add_edges_from(valid_edges)
    labels = {i: f"{i}: {str(clique)}" for i, clique in cliques_dict.items()}
    plt.figure(figsize=fig_size)
    try:
        pos = nx.kamada_kawai_layout(G)
    except nx.NetworkXError as e:
        print(f"'kamada_kawai_layout' did not work: {e}. 'spring_layout' used instead.")
        pos = nx.spring_layout(G, seed=42) 
    nx.draw(
        G,
        pos,
        labels = labels,
        node_color='lightgreen', 
        node_size=500,         
        edge_color='black',       
        font_size=6,           
        width=1               
    )
    plt.margins(0.1)
    plt.show()

def create_modsubgraph_original(
    mg: mod.Graph,
    vertex_idxs_to_keep: Dict[int, List[int]],
    internal_to_external_id_map: Dict[int, int],
    intersection: Optional[Set[int]] = None,
) -> Optional[mod.Graph]:
    """
    Creates a new mod graph representing the subgraph induced by the given set of vertex IDs,
    along with a mapping of new indices back to original indices.

    Args:
        mg (mod.Graph): The original mod graph object.
        vertex_idxs_to_keep: dict whose values are external atom IDs to include.
        internal_to_external_id_map: map from internal -> external vertex id.
        intersection: if provided, any cluster that intersects it is entirely excluded.

    Returns:
        mod.Graph for the substructure/fragment, or None on invalid input.
    """
    if not mg or not vertex_idxs_to_keep:
        return None

    # Collect external vertex IDs to keep, excluding clusters that intersect `intersection`.
    if intersection:
        intersection = set(intersection)  # ensure O(1) membership
        ext_ids = [vid for cluster in vertex_idxs_to_keep.values()
                   if set(cluster).isdisjoint(intersection)
                   for vid in cluster]
    else:
        ext_ids = [vid for cluster in vertex_idxs_to_keep.values() for vid in cluster]

    ext_ids_set: Set[int] = set(ext_ids)
    if not ext_ids_set:
        return None

    int2ext = internal_to_external_id_map  # local alias for speed

    # Collect H (internal ids) attached to kept heavy atoms.
    h_internal_ids_to_keep: Set[int] = set()
    for e in mg.edges:
        s = e.source
        t = e.target

        s_label = s.stringLabel
        t_label = t.stringLabel
        s_is_wild = (s_label == WILD_STAR or s_label == X_VAR)
        t_is_wild = (t_label == WILD_STAR or t_label == X_VAR)

        s_is_H = (not s_is_wild) and (s.atomId.symbol == "H")
        t_is_H = (not t_is_wild) and (t.atomId.symbol == "H")

        if s_is_H and t_is_H:
            raise Exception(
                "Edge-case: bond between two H-atoms occurred, case not handled yet! "
                "see chemutils.create_modsubgraph_original"
            )

        if s_is_H:
            if int2ext[t.id] in ext_ids_set:
                h_internal_ids_to_keep.add(s.id)
        elif t_is_H:
            if int2ext[s.id] in ext_ids_set:
                h_internal_ids_to_keep.add(t.id)

    # Determine starting external id for new hydrogens (avoid clashes with kept external ids).
    max_ext = getattr(mg, "maxExternalId", 0)
    start_ext = max_ext if max_ext != 0 else (max(ext_ids_set) if ext_ids_set else 0)
    next_h_ext = start_ext + 1

    # Map included internal ids -> external ids for node emission.
    included_int_to_ext: Dict[int, int] = {}
    gml_nodes: List[str] = []

    append_node = gml_nodes.append
    for v in mg.vertices:
        v_label = v.stringLabel
        is_wild = (v_label == WILD_STAR or v_label == X_VAR)
        # Only non-wildstar vertices have atomId; wildstar is not H
        is_H = (not is_wild) and (v.atomId.symbol == "H")

        if is_H:
            if v.id in h_internal_ids_to_keep:
                included_int_to_ext[v.id] = next_h_ext
                append_node(f'  node [ id {next_h_ext} label "H" ]')
                next_h_ext += 1
        else:
            mapped_ext = int2ext[v.id]
            if mapped_ext in ext_ids_set:
                included_int_to_ext[v.id] = mapped_ext
                append_node(f'  node [ id {mapped_ext} label "{v_label}" ]')

    # Maintain original behavior: assert at least one vertex collected.
    assert len(gml_nodes) != 0

    # Emit edges only when both endpoints were included.
    gml_edges: List[str] = []
    append_edge = gml_edges.append
    for e in mg.edges:
        s_ext = included_int_to_ext.get(e.source.id)
        t_ext = included_int_to_ext.get(e.target.id)
        if s_ext is not None and t_ext is not None:
            if s_ext <= t_ext:
                a, b = s_ext, t_ext
            else:
                a, b = t_ext, s_ext
            append_edge(f'  edge [ source {a} target {b} label "{e.stringLabel}" ]')

    # Build GML
    if gml_edges:
        gml = "graph [\n" + "\n".join(gml_nodes) + "\n" + "\n".join(gml_edges) + "\n]"
    else:
        gml = "graph [\n" + "\n".join(gml_nodes) + "\n]"

    try:
        return mod.Graph.fromGMLString(gml, add=False, printStereoWarnings=False)
    except Exception as ex:
        # print(f"Error creating graph from GML: {ex}")
        # print(f"Generated GML:\n{gml}")
        return None

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def sanitize(mol, kekulize=True):
    try:
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def create_modsubgraph(
    mg: mod.Graph,
    vertex_idxs_to_keep: Dict[int, List[int]],
    internal_to_external_id_map: Dict[int, int],
    intersection: Set[int],
    anchor_vertex_ids: Set[int],
    break_bonds_after_ring_split: bool = False,
    real_intersection: Optional[Set[int]] = None,
) -> mod.Graph:
    """
    Create a mod graph representing the subgraph induced by the given external vertex IDs.
    Includes hydrogens attached to kept heavy atoms (but not to anchors), and relabels anchors
    to WILD_STAR. Optionally breaks bonds whose both endpoints are in `real_intersection`.
    
    Args:
        mg (mod.Graph): The original mod graph object.
        vertex_idxs_to_keep: A dictionary with key:int and value:atom indices from the original
            molecule to include in the substructure/fragment.
        internal_to_external_id_map (Dict[int, int]): map from internal to external vertex id
        intersection (Set[int]): ids of vertices part of the interface that is split
        anchor_vertex_ids (Set[int]): ids of all anchor nodes
        break_bonds_after_ring_split (bool): whether we need to break the bond after the right ring
            has been split off of left ring with is_left True, then we need to break the bond, because
            now it belongs to right ring and not left ring. 
            (This can maybe (need to check yet) happen with one ringe being split
            off of two or more rings it is connected to via an edge.)

    Returns:
        The substructure/fragment as a mod.Graph (or None on failure, for backward-compat).
    """
    if not mg or not vertex_idxs_to_keep:
        return None

    # 1) Collect external vertex IDs to keep (exclude clusters intersecting `intersection`)
    inter = set(intersection) if intersection else None
    ext_ids: List[int] = []
    if inter:
        for cluster in vertex_idxs_to_keep.values():
            keep = True
            for x in cluster:
                if x in inter:
                    keep = False
                    break
            if keep:
                ext_ids.extend(cluster)
    else:
        for cluster in vertex_idxs_to_keep.values():
            ext_ids.extend(cluster)

    ext_ids_set: Set[int] = set(ext_ids)
    if not ext_ids_set and not anchor_vertex_ids:
        return None

    # Always include anchors (external IDs) in the vertex set
    anchors = set(anchor_vertex_ids)
    ext_ids_set |= anchors

    int2ext = internal_to_external_id_map  # local alias for speed
    edges = mg.edges
    verts = mg.vertices

    # 2) Determine hydrogens (by INTERNAL id) to include:
    #    include H if attached to a kept heavy atom that is not an anchor and not wild-star.
    h_internal_ids_to_keep: Set[int] = set()
    for e in edges:
        s = e.source
        t = e.target

        s_label = s.stringLabel
        t_label = t.stringLabel

        s_is_wild = (s_label == WILD_STAR or s_label == X_VAR)
        t_is_wild = (t_label == WILD_STAR or t_label == X_VAR)

        s_is_H = (not s_is_wild) and (s.atomId.symbol == "H")
        t_is_H = (not t_is_wild) and (t.atomId.symbol == "H")

        if s_is_H and t_is_H:
            raise Exception(
                "Edge-case: bond between two H-atoms occurred, case not handled yet! "
                "see chemutils.create_modsubgraph"
            )

        if s_is_H:
            neigh_ext = int2ext[t.id]
            if (neigh_ext in ext_ids_set) and (neigh_ext not in anchors) :#and (not t_is_wild): # required because H-atoms can also be on Xs
                h_internal_ids_to_keep.add(s.id)
        elif t_is_H:
            neigh_ext = int2ext[s.id]
            if (neigh_ext in ext_ids_set) and (neigh_ext not in anchors) :#and (not s_is_wild): # required because H-atoms can also be on Xs
                h_internal_ids_to_keep.add(t.id)

    # 3) Emit nodes:
    #    - Heavy/anchor nodes keep their external IDs; anchors get WILD_STAR label.
    #    - Hydrogens get fresh external IDs after mg.maxExternalId (or max(ext_ids_set)).
    max_ext = getattr(mg, "maxExternalId", 0)
    start_ext = max_ext if max_ext != 0 else (max(ext_ids_set) if ext_ids_set else 0)
    next_h_ext = start_ext + 1

    included_int_to_ext: Dict[int, int] = {}  # internal id -> external id used in GML
    gml_nodes: List[str] = []
    append_node = gml_nodes.append

    for v in verts:
        v_label = v.stringLabel
        is_wild = (v_label == WILD_STAR or v_label == X_VAR)
        is_H = (not is_wild) and (v.atomId.symbol == "H")

        if is_H:
            if v.id in h_internal_ids_to_keep:
                included_int_to_ext[v.id] = next_h_ext
                append_node(f'  node [ id {next_h_ext} label "H" ]')
                next_h_ext += 1
        else:
            mapped_ext = int2ext[v.id]
            if mapped_ext in ext_ids_set:
                out_label = WILD_STAR if mapped_ext in anchors else v_label
                included_int_to_ext[v.id] = mapped_ext
                append_node(f'  node [ id {mapped_ext} label "{out_label}" ]')

    # Maintain original behavior: must have at least one vertex
    assert len(gml_nodes) != 0

    # 4) Emit edges for which both endpoints were included
    gml_edges: List[str] = []
    append_edge = gml_edges.append
    do_break = break_bonds_after_ring_split and bool(real_intersection)
    real_inter = real_intersection if real_intersection else set()

    for e in edges:
        s_ext = included_int_to_ext.get(e.source.id)
        if s_ext is None:
            continue
        t_ext = included_int_to_ext.get(e.target.id)
        if t_ext is None:
            continue

        # If requested, drop bonds whose endpoints are both in real_intersection
        if do_break and (s_ext in real_inter) and (t_ext in real_inter):
            continue

        a, b = (s_ext, t_ext) if s_ext <= t_ext else (t_ext, s_ext)
        append_edge(f'  edge [ source {a} target {b} label "{e.stringLabel}" ]')

    # 5) Assemble GML and construct graph
    if gml_edges:
        gml = "graph [\n" + "\n".join(gml_nodes) + "\n" + "\n".join(gml_edges) + "\n]"
    else:
        gml = "graph [\n" + "\n".join(gml_nodes) + "\n]"

    try:
        return mod.Graph.fromGMLString(gml, add=False, printStereoWarnings=False)
    except Exception as ex:
        print(f"Error creating graph from GML: {ex}")
        print(f"Generated GML:\n{gml}")
        return None

def get_num_wildstars(mg: mod.Graph, left_right_union: Set[int], internal_external_map: Dict[int, int]) -> int:
    """Gets number of vertices with '*' / wildstar label on mod.Graph"""
    count: int = 0
    for v in mg.vertices:
        mapped_v_id: int = internal_external_map[v.id]
        if (v.stringLabel == WILD_STAR or v.stringLabel == X_VAR) and mapped_v_id not in left_right_union:
            count += 1
    return count

def get_internal_to_external_id_map(mg: mod.Graph, includeHs: bool = True) -> Dict[int, int]:
    """
    Return a mapping from internal -> external vertex IDs for a mod.Graph.

    Fast path prefers per-vertex externalId if available; otherwise falls back
    to getVertexFromExternalId over the external ID range.
    """
    if mg is None:
        return {}

    min_ext = getattr(mg, "minExternalId", 0)
    max_ext = getattr(mg, "maxExternalId", 0)

    # Common case: identity mapping (no external IDs defined)
    if min_ext == 0 and max_ext == 0:
        if includeHs:
            return {v.id: v.id for v in mg.vertices}
        return {
            v.id: v.id
            for v in mg.vertices
            if (v.stringLabel == WILD_STAR or v.stringLabel == X_VAR) or v.atomId.symbol != "H"
        }

    # Helper for filtering out hydrogens (unless includeHs)
    def keep(v) -> bool:
        return includeHs or (v.stringLabel == WILD_STAR or v.stringLabel == X_VAR) or v.atomId.symbol != "H"

    # Try fast path: use vertex.externalId if exposed by bindings
    it = iter(mg.vertices)
    try:
        first = next(it)
    except StopIteration:
        return {}

    if hasattr(first, "externalId"):
        result: Dict[int, int] = {}
        if keep(first):
            result[first.id] = first.externalId  # type: ignore[attr-defined]
        for v in it:
            if keep(v):
                result[v.id] = v.externalId  # type: ignore[attr-defined]
        return result

    # Fallback: use getVertexFromExternalId over [min_ext, max_ext]
    get_by_ext = mg.getVertexFromExternalId
    result: Dict[int, int] = {}
    for ext_id in range(min_ext, max_ext + 1):
        v = get_by_ext(ext_id)
        if v.isNull():
            continue
        if keep(v):
            result[v.id] = ext_id
    return result
        
def find_vertex_by_id(mg: mod.Graph, vid: int) -> Optional[mod.Graph.Vertex]:
    """
    Finds a vertex object in a mød graph by its internal ID.

    Args:
      mg: The mød graph object (instance of mod.Graph).
      vid: The integer internal ID of the vertex to find.

    Returns:
      The mod.Graph.Vertex object if found, otherwise None.
    """
    for vertex in mg.vertices:
        if vertex.id == vid:
            return vertex
    return None

def find_edge(mg: mod.Graph, edge: Tuple[int, int]) -> Optional[mod.Graph.Edge]:
    """
    Finds a vertex object in a mød graph by its internal ID.

    Args:
      mg: The mød graph object (instance of mod.Graph).
      edge: Tuple of (source vertex id, target vertex id)

    Returns:
      The mod.Graph.Edge object if found, otherwise None.
    """
    for edge in mg.edges:
        if edge == edge:
            return edge
    return None


def get_junction_tree(mg: mod.Graph, internal_to_external_id_map: Optional[Dict[int, int]] = None) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
    """
    Decomposes a mod graph into a tree of chemical substructures (cliques).

    Identifies cliques (rings, non-ring bonds, key junction atoms),
    builds a graph connecting cliques based on shared atoms, and
    computes the Maximum Spanning Edges (MSE) of this graph.

    Args:
      mg (mod.Graph): The input mod graph object.
      internal_to_external_id_map (Dict[int, int]): map for mapping from mod internal id to external id set in smiles string

    Returns:
      A tuple containing:
        - cliques: A dict where each key is the original list position
            and each value is a list of atom indices belonging to a clique.
        - edges: A dict where each key is the original edge position in the list and
            the value is a list of tuples representing the edges (connections) between cliques
            in the MST, specified by clique indices.
    """
    nxg = get_nxgraph_from_mod_graph(mg)
    n_vertices = mg.numVertices

    # Special case: single vertex
    if n_vertices == 1:
        v0 = next(iter(mg.vertices))
        return {0: [internal_to_external_id_map[v0.id]]}, {}

    cliques: List[List[int]] = []

    # Cache locals for speed
    edges_iter = mg.edges
    edge_in_graph = is_edge_in_nxgraph

    # Add non-ring bond cliques (between non-H atoms)
    for e in edges_iter:
        s = e.source
        t = e.target

        s_is_wild = (s.stringLabel == WILD_STAR or s.stringLabel == X_VAR)
        t_is_wild = (t.stringLabel == WILD_STAR or t.stringLabel == X_VAR)

        # Skip bonds involving H (unless wildstar or 'X', which is never H)
        if (not s_is_wild and s.atomId.symbol == "H") or (not t_is_wild and t.atomId.symbol == "H"):
            continue

        v1_id = s.id
        v2_id = t.id

        # Preserve original behavior via helper
        if not edge_in_graph(nxg, (v1_id, v2_id)):
            cliques.append([v1_id, v2_id])

    # Add ring cliques via minimum cycle basis (skip call if graph is acyclic)
    # Cyclomatic number > 0 implies cycles exist
    if nxg.number_of_edges() - nxg.number_of_nodes() + nx.number_connected_components(nxg) > 0:
        cliques.extend(nx.minimum_cycle_basis(nxg))

    # Build per-vertex clique neighbors
    neighbors: List[List[int]] = [[] for _ in range(n_vertices)]
    for idx, clique in enumerate(cliques):
        for v in clique:
            neighbors[v].append(idx)

    # Merge rings with > 2-atom intersections
    cliques_sets: List[set] = [set(c) for c in cliques]
    cliques_len: List[int] = [len(c) for c in cliques]

    for i in range(len(cliques)):
        if cliques_len[i] <= 2:
            continue
        ci_set = cliques_sets[i]
        for atom in cliques[i]:
            for j in neighbors[atom]:
                if i >= j or cliques_len[j] <= 2:
                    continue
                inter_size = len(ci_set & cliques_sets[j])
                if inter_size > 2:
                    # Merge j into i
                    ci_set |= cliques_sets[j]
                    cliques_sets[i] = ci_set
                    cliques[i] = list(ci_set)
                    cliques_len[i] = len(ci_set)

                    # Clear j
                    cliques_sets[j].clear()
                    cliques[j] = []
                    cliques_len[j] = 0

    # Build clique graph edges and add singleton cliques where needed
    edges_score = defaultdict(int)

    for v in range(n_vertices):
        cnei = neighbors[v]
        if len(cnei) <= 1:
            continue

        bonds = [c for c in cnei if cliques_len[c] == 2]
        rings = [c for c in cnei if cliques_len[c] > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([v])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges_score[(c1, c2)] = 1
        elif len(rings) > 2:
            cliques.append([v])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges_score[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            # Connect all neighbor cliques; weight by intersection size
            for a in range(len(cnei)):
                c1 = cnei[a]
                if cliques_len[c1] == 0:
                    continue
                s1 = cliques_sets[c1]
                for b in range(a + 1, len(cnei)):
                    c2 = cnei[b]
                    if cliques_len[c2] == 0:
                        continue
                    inter_size = len(s1 & cliques_sets[c2])
                    if edges_score[(c1, c2)] < inter_size:
                        edges_score[(c1, c2)] = inter_size

    dict_cliques: Dict[int, List[int]] = {i: c for i, c in enumerate(cliques)}
    dict_edges: Dict[int, Tuple[int, int]] = {i: e for i, e in enumerate(edges_score.keys())}

    # If no edges, return early
    if not edges_score:
        if internal_to_external_id_map:
            m = internal_to_external_id_map
            for kc in dict_cliques:
                dict_cliques[kc] = [m[v] for v in dict_cliques[kc]]
        dict_cliques, dict_edges = sanitize_junction_tree_if_necessary(dict_cliques, dict_edges)
        # sort clique atom id list and edge tuples
        dict_cliques, dict_edges = { k: sorted(v) for k,v in dict_cliques.items() }, { k: tuple(sorted(v)) for k,v in dict_edges.items() }
        return dict_cliques, dict_edges

    # Build clique graph with transformed weights for minimum spanning edges
    edges_list = [u + (MST_MAX_WEIGHT - w,) for u, w in edges_score.items()]
    clique_graph = nx.Graph()
    clique_graph.add_weighted_edges_from(edges_list)

    mse = nx.minimum_spanning_edges(clique_graph, weight="weight", data=False)
    edges_from_mse = list(mse)

    dict_edges = {i: e for i, e in enumerate(edges_from_mse)}

    # Optional mapping to external IDs
    if internal_to_external_id_map:
        m = internal_to_external_id_map
        for kc in dict_cliques:
            dict_cliques[kc] = [m[v] for v in dict_cliques[kc]]

    # Final sanitization step
    dict_cliques, dict_edges = sanitize_junction_tree_if_necessary(dict_cliques, dict_edges)
    # sort clique atom id list and edge tuples
    dict_cliques, dict_edges = { k: sorted(v) for k,v in dict_cliques.items() }, { k: tuple(sorted(v)) for k,v in dict_edges.items() }
    return dict_cliques, dict_edges

def sanitize_junction_tree_if_necessary(
    cliques_dict: Dict[int, List[int]],
    edges_dict: Dict[int, Tuple[int, int]],
    verbose: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[int, int]]]:
    """
    Checks if there are any empty cliques. If True then it removes the empty cliques
    and updates all cliques connected to empty clique to point point to the right clique.
    """
    # 1) Collect empty cliques
    empty_clique_ids: Set[int] = {cid for cid, c in cliques_dict.items() if len(c) == 0}
    if not empty_clique_ids:
        return cliques_dict, edges_dict

    # 2) Identify edges touching empty cliques, and record non-empty neighbors
    eids_to_delete: Set[int] = set()
    ids_of_empty_clique_neighbors: Set[int] = set()

    for eid, (a, b) in edges_dict.items():
        a_empty = a in empty_clique_ids
        b_empty = b in empty_clique_ids
        if a_empty or b_empty:
            eids_to_delete.add(eid)
            # If exactly one endpoint is empty, record the other (non-empty) neighbor
            if a_empty ^ b_empty:
                ids_of_empty_clique_neighbors.add(b if a_empty else a)

    # 3) Delete edges between empty cliques and their neighbors (in-place, as original)
    for _id in eids_to_delete:
        del edges_dict[_id]

    # 4) Build clique adjacency (neighbors among cliques)
    neighbors: Dict[int, Set[int]] = defaultdict(set)
    for u, v in edges_dict.values():
        neighbors[u].add(v)
        neighbors[v].add(u)
    # Ensure all clique ids exist in neighbors (even isolated ones)
    for cid in cliques_dict.keys():
        neighbors.setdefault(cid, set())

    # 5) If a single connected component (considering only non-empty cliques), just drop empty cliques
    nxgraph: nx.Graph = nx.Graph()
    nxgraph.add_edges_from(edges_dict.values())
    nxgraph.add_nodes_from([k for k, v in cliques_dict.items() if len(v) != 0])

    num_components: int = nx.number_connected_components(nxgraph)
    if num_components == 1:
        for _i in empty_clique_ids:
            del cliques_dict[_i]
        return cliques_dict, edges_dict

    # 6) Reconnect cliques that used to point to empty cliques, to correct non-empty cliques
    #    Precompute sets for cliques to speed up intersections
    clique_sets: Dict[int, Set[int]] = {cid: set(members) for cid, members in cliques_dict.items()}
    # Precompute union of neighbor atoms for each clique id (based on current neighbors)
    neighbor_atoms: Dict[int, Set[int]] = {}
    for cid, nbs in neighbors.items():
        if not nbs:
            neighbor_atoms[cid] = set()
        else:
            # Union of all atoms in neighbor cliques
            acc: Set[int] = set()
            for nb in nbs:
                acc |= clique_sets.get(nb, set())
            neighbor_atoms[cid] = acc

    # Maintain an incrementing edge id counter equal to "max existing key"
    next_edge_id = max(edges_dict.keys(), default=0)

    for neigh_cid in ids_of_empty_clique_neighbors:
        # Components reflect current nxgraph (updated as we add edges)
        components: List[Set[int]] = list(nx.connected_components(nxgraph))
        cluster_set = clique_sets.get(neigh_cid, set())

        # Candidates: cliques that share atoms with neigh_cid after excluding neighbor atoms of candidate,
        # and are not the same clique.
        # Keep logic identical while avoiding repeated set construction.
        neigh_candidates = []
        for cand_id, cand_members in cliques_dict.items():
            if neigh_cid == cand_id:
                continue
            if not cand_members:
                continue
            cand_set = clique_sets[cand_id]
            # Same expression as original: set(cluster).intersection(c) - set([y for x in neighbors[id] for y in cliques_dict[x]])
            if (cluster_set & cand_set) - neighbor_atoms.get(cand_id, set()):
                neigh_candidates.append(cand_id)

        # Remove candidates that are in the same component (preserve original helper call)
        if neigh_candidates:
            components_list = components  # naming as in original
            neigh_candidates = [
                i for i in neigh_candidates if check_different_components(components_list, neigh_cid, i)
            ]

        if not neigh_candidates:
            continue

        # Select candidate with smallest clique size (same criterion as original)
        selected_candidate_id: int = min(neigh_candidates, key=lambda nc: len(cliques_dict[nc]))

        # Add new connection to edges_dict and nxgraph
        next_edge_id += 1
        a, b = (neigh_cid, selected_candidate_id)
        new_edge = (min(a, b), max(a, b))
        edges_dict[next_edge_id] = new_edge
        nxgraph.add_edge(*new_edge)

    # 7) Remove empty cliques
    for _i in empty_clique_ids:
        del cliques_dict[_i]

    # 8) Merge disconnected components into one connected component if needed
    components: List[Set[int]] = list(nx.connected_components(nxgraph))
    if len(components) > 1 and verbose:
        print("Disconnected components occurred, will be merged into one connected component.")

    while len(components) > 1:
        dict1: Dict[int, List[int]] = {k: cliques_dict[k] for k in components[0]}
        best_pair: Optional[Tuple[int, int]] = None

        for comp in components[1:]:
            dict2: Dict[int, List[int]] = {k: cliques_dict[k] for k in comp}
            best_pair = max_intersecting_lists(dict1, dict2)
            if best_pair is not None:
                e1, e2 = best_pair
                next_edge_id += 1
                new_edge = (min(e1, e2), max(e1, e2))
                edges_dict[next_edge_id] = new_edge
                nxgraph.add_edge(*new_edge)
                break

        # Recompute components on the updated graph
        components = list(nx.connected_components(nxgraph))

    return cliques_dict, edges_dict

def load_rules_from_folder(folder_path: str) -> List[mod.Rule]:
    """Loads all rules from folder and parses them into ModRule object"""
    rules: List[mod.Rule] = []
    if not os.path.exists(folder_path):
        return rules
    file_names: List[str] = os.listdir(folder_path)
    file_names.sort(key=lambda x: int(re.search(r"[-+]?\d*\.\d+|\d+", x).group(0)))
    for fname in file_names:
        with open(f"{folder_path}/{fname}", 'r') as ifile:
            ruleGmlString: str = ifile.read().strip()
        rule: mod.Rule = mod.Rule.fromGMLString(ruleGmlString)
        rules.append(rule)
    return rules

def max_intersecting_lists(dict1: Dict[int, List[int]], dict2: Dict[int, List[int]]) -> Optional[Tuple[int, int]]:
    """
    Finds the pair of sublists (one from each input list) with the highest number of intersecting integers.
    
    In case of ties in intersection count, prefers the pair with fewer total elements.
    If still tied, selects the first such pair encountered.

    Args:
        dict1 (Dict[int, List[int]]): First dict of lists of integers.
        dict2 (Dict[int, List[int]]): Second dict of lists of integers.

    Returns:
        Optional[Tuple[int, int]]: The pair of dict keys of lists with the highest intersection, 
        or None if either input list is empty.
    """
    max_intersection = -1
    best_pair: Optional[Tuple[int, int]] = None
    min_total_length = float('inf')

    for key1, sublist1 in dict1.items():
        set1 = set(sublist1)
        for key2, sublist2 in dict2.items():
            set2 = set(sublist2)
            intersection_count = len(set1 & set2)
            total_length = len(sublist1) + len(sublist2)

            if (intersection_count > max_intersection 
                or (intersection_count == max_intersection and total_length < min_total_length) 
                or (intersection_count == max_intersection and total_length == min_total_length and best_pair is None)):
                max_intersection = intersection_count
                min_total_length = total_length
                best_pair = (key1, key2)

    return best_pair

def check_different_components(components: List[Set[int]], cid1: int, cid2: int) -> bool:
    """
    Checks if cid1 and cid2 are in different components.

    Args:
        components (List[Set[int]]): A list of disjoint sets, where each set represents a component.
        cid1 (int): The first element to check.
        cid2 (int): The second element to check.

    Returns:
        bool: True if cid1 and cid2 are in different sets (i.e., components), False if they are in the same one.
    """
    for component in components:
        if cid1 in component and cid2 in component:
            return False  
    return True 

def is_edge_in_modgraph(mg: mod.Graph, edge: Tuple[int, int]) -> bool:
    """
    Checks if an edge is part of any simple cycle in a mød graph using NetworkX.

    Args:
      mg: The mød graph object.
      edge: The tuple of vertex id (source, target) representing the edge.

    Returns:
      True if the edge is in a cycle, False otherwise.
    """
    nxg = get_nxgraph_from_mod_graph(mg)
    is_in_cycle = is_edge_in_nxgraph(nxg, edge)
    return is_in_cycle

def is_edge_in_nxgraph(nxg: nx.Graph, edge: Tuple[int, int]) -> bool:
    """
    Checks if an edge is part of any simple cycle in a mød graph using NetworkX.

    Args:
      mg: The mød graph object.
      edge: The tuple of vertex id (source, target) representing the edge.

    Returns:
      True if the edge is in a cycle, False otherwise.
    """
    v1, v2 = edge
    edge_rev = (edge[1], edge[0])
    try:
        if v1 not in nxg or v2 not in nxg:
            return False
        for cycle_nodes in nx.simple_cycles(nxg):
            if len(cycle_nodes) < 2:
                continue
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                current_edge = (u, v)
                if current_edge == edge or current_edge == edge_rev:
                    return True # edge found in cylce
        return False # no edge found
    except Exception as e:
        print(f"An error occurred during cycle detection: {e}")
        return False
    

def is_vertex_in_modgraph(mg: mod.Graph, vid: int) -> bool:
    """
    Checks if a vertex is part of any simple cycle in a mød graph.

    Args:
      mg: The mød graph object.
      vid: The integer ID of the vertex to check.

    Returns:
      True if the vertex is in a cycle, False otherwise.
    """
    nxg = get_nxgraph_from_mod_graph(mg)
    is_in_cyle = is_vertex_in_nxgraph(nxg, vid)
    return is_in_cyle


def is_vertex_in_nxgraph(nxg: nx.Graph, vid: int) -> bool:
    """
    Checks if a vertex is part of any simple cycle in a nx digraph.

    Args:
      nxg (nx.Digraph): the digraph
      vid (int): id of the vertex

    Returns:
      True if the vertex is in a cycle, False otherwise.
    """
    try:
        if vid not in nxg:
            return False
        return any(vid in cycle for cycle in nx.simple_cycles(nxg)) 
    except Exception as e:
        print(f"An error occurred during cycle detection: {e}")
        return False


def get_nxgraph_from_mod_graph(mg: mod.Graph) -> nx.DiGraph:
    """
    Convert mod graph to nx graph.
    
    Args:
      mg (mod.Graph): the mod graph
    Returns:
      the nx graph.
    """
    edges = []
    for e in mg.edges: 
      edges.append((e.source.id, e.target.id)) 
    nxg = nx.Graph(edges) 
    return nxg

def find_connected_component_split_subsets(original_graph: nx.Graph, candidates: Set[int], internal_to_external_id_map: Dict[int, int]) -> List[Set[int]]:
    """
    Find all subsets of candidate atom external ids of intersection 
    whose removal splits the graph into exactly two connected components.
    
    Args:
        original_graph: graph to evaluate results on 
        candidates (Set[int]):
            A list of external ids of the atoms considered for removal.
        internal_to_external_id_map: map from interal to external atom id
    
    Returns:
    List[Set[int]]
        A list of subsets (as sets) from the candidate nodes. Each subset, if removed along with its edges,
        causes the graph to split into exactly two connected components.
        
    Notes:
        - The function does not modify the original graph; it works on copies.
        - Runtime is exponential in the number of candidate nodes, as it checks all possible subsets.
    """
    result = []
    nxgraph_no_h_atoms: nx.Graph = get_nxgraph_from_mod_graph(original_graph, internal_to_external_id_map, True)
    
    for r in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, r):
            nxgraph_no_h_atoms_copy = nxgraph_no_h_atoms.copy()
            nxgraph_no_h_atoms_copy.remove_nodes_from(subset)
            num_components = nx.number_connected_components(nxgraph_no_h_atoms_copy)
            if num_components == 1:
                result.append(set(subset))
    return result

def find_connected_component_split_subsets(original_graph: nx.Graph, candidates: Set[int], 
                                           left: Set[int], right: Set[int], 
                                           internal_to_external_id_map: Dict[int, int]) -> List[Set[int]]:
    """
    Find all subsets of candidate atom external ids whose removal splits the graph into exactly two connected components,
    with one component being a subset of 'left' and the other a subset of 'right'.
    
    Args:
        original_graph: graph to evaluate results on 
        candidates (Set[int]): A set of external ids of the atoms considered for removal.
        left (Set[int]): A set of external ids representing the left partition.
        right (Set[int]): A set of external ids representing the right partition.
        internal_to_external_id_map: map from internal to external atom id
    
    Returns:
    List[Set[int]]
        A list of subsets (as sets) from the candidate nodes. Each subset, if removed along with its edges,
        causes the graph to split into exactly two connected components that are subsets of left and right.
        
    Notes:
        - The function does not modify the original graph; it works on copies.
        - Runtime is exponential in the number of candidate nodes, as it checks all possible subsets.
    """
    result = []
    nxgraph_no_h_atoms: nx.Graph = get_nxgraph_from_mod_graph(original_graph, internal_to_external_id_map, True)
    
    for r in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, r):
            nxgraph_no_h_atoms_copy = nxgraph_no_h_atoms.copy()
            nxgraph_no_h_atoms_copy.remove_nodes_from(subset)
            
            components = list(nx.connected_components(nxgraph_no_h_atoms_copy))
            if len(components) == 2:
                comp1, comp2 = components
                
                if (comp1.issubset(left) and comp2.issubset(right)) or (comp1.issubset(right) and comp2.issubset(left)):
                    result.append(set(subset))
    return result

# Check for Completeness
def check_completeness(generated: Set[mod.Graph], targets: List[mod.Graph]) -> bool:
    return all(t in generated for t in targets)

### RDKIT section all functions that use rdkit come here
# Measure Structural Diversity
def to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_matrix(graphs: List[mod.Graph]) -> List[List[float]]:
    fps = [to_fp(g.smiles) for g in graphs]
    matrix = []
    for i in range(len(fps)):
        row = []
        for j in range(len(fps)):
            sim = Chem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            row.append(sim)
        matrix.append(row)
    return matrix

def compute_diversity(graphs: List[mod.Graph]) -> float:
    mat = tanimoto_matrix(graphs)
    # Average pairwise distance (1 - similarity)
    return sum(1 - mat[i][j] for i in range(len(mat)) for j in range(i+1, len(mat))) / (len(mat)*(len(mat)-1)/2)

def mod_to_rdkit_bond_map(mod_bond_type: mod.libpymod.BondType) -> Chem.BondType:
    """
    Maps from mod bond type to rdkit chem bond type.
    """
    mod_to_rdkit_bond_map = {
        mod.libpymod.BondType.Single: Chem.BondType.SINGLE,
        mod.libpymod.BondType.Double: Chem.BondType.DOUBLE,
        mod.libpymod.BondType.Triple: Chem.BondType.TRIPLE,
        mod.libpymod.BondType.Aromatic: Chem.BondType.AROMATIC,
        mod.libpymod.BondType.Invalid: Chem.BondType.UNSPECIFIED, 
    }
    return mod_to_rdkit_bond_map.get(mod_bond_type, Chem.BondType.UNSPECIFIED)

def display_mol(mol: Chem.Mol, addAtomIndices: bool=True, fixedFontSize: float=16, annotationFontScale: float=0.5, size: Tuple[int, int]=(300,300), removeHs: bool=True) -> None:
    if mol is None:
        print("Mol must not be None.")
        return
    if removeHs:
        mol = Chem.RemoveHs(mol)
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) 
    drawOptions = drawer.drawOptions()
    drawOptions.addAtomIndices = addAtomIndices
    drawOptions.fixedFontSize = fixedFontSize
    drawOptions.annotationFontScale = annotationFontScale
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace('svg:', '')))      
    
def display_smiles(smiles: str, addAtomIndices: bool=True, size: Tuple[int, int]=(300,300), removeHs: bool=True, sanitize: bool=False) -> None:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        print(f"Could not create molecule from SMILES: {smiles}")
        return
    if removeHs:
        mol = Chem.RemoveHs(mol)
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) 
    drawOptions = drawer.drawOptions()
    drawOptions.addAtomIndices = addAtomIndices 
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace('svg:', '')))

###

def calculate_fast_quality_score(mol: Chem.Mol) -> float:
    """
    Calculates a heuristic quality score using only fast metrics (QED, Ro5, SAScore, Complexity).
    Skips Strain Energy (MMFF).
    """
    if mol is None: return -100.0
    
    qed_score = QED.qed(mol)
    ro5 = calculate_lipinski_violations(mol)
    sa = calculate_sascore(mol)
    comp = calculate_complexity(mol)
    
    ro5_penalty = ro5 * 0.2
    sa_penalty = (sa - 1.0) / 9.0
    complexity_penalty = min(comp, 1500.0) / 1500.0
    
    # Weights similar to evaluate_molecule_quality
    return (1.0 * qed_score) - (0.5 * ro5_penalty) - (0.5 * sa_penalty) - (0.2 * complexity_penalty)

def fill_wildcards_with_vocab(
    mol: Chem.Mol, 
    vocab_list: List[str], 
    atom_freqs: Dict[str, float]
) -> Optional[Chem.Mol]:
    """
    Replaces all dummy atoms (atomic num 0) in the molecule with atoms from vocab_list,
    respecting valency constraints and sampling based on atom_freqs.
    """
    if mol is None: return None
    
    try:
        # Optimization: Modify in-place (mol is already a copy from MolFromSmiles)
        # instead of creating RWMol.
        
        atoms_to_replace = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        
        if not atoms_to_replace:
            try:
                Chem.SanitizeMol(mol)
            except:
                return None
            return mol
        
        # Randomize order
        random.shuffle(atoms_to_replace)
        
        pt = Chem.GetPeriodicTable()
        
        # Cache valences for this vocab list to avoid repeated GetDefaultValence calls
        # We use a static cache or just compute it once per call if vocab is small.
        # Since vocab is small (usually < 20 atoms), we just compute it.
        vocab_valences = []
        for symbol in vocab_list:
            if symbol == 'C': v = 4
            elif symbol == 'N': v = 3
            elif symbol == 'O': v = 2
            elif symbol == 'S': v = 6
            elif symbol == 'P': v = 5
            elif symbol in ['F', 'Cl', 'Br', 'I', 'H']: v = 1
            elif symbol == 'B': v = 3
            else:
                try:
                    v = pt.GetDefaultValence(pt.GetAtomicNumber(symbol))
                except:
                    v = 4
            
            weight = atom_freqs.get(symbol, 0.0) if atom_freqs else 1.0
            vocab_valences.append((symbol, v, weight))

        for atom in atoms_to_replace:
            # Calculate required valence
            required_valence = 0.0
            for bond in atom.GetBonds():
                b_type = bond.GetBondTypeAsDouble()
                if b_type == 1.5:
                    required_valence += 1.0
                else:
                    required_valence += b_type
            
            valid_atoms = []
            valid_weights = []

            for symbol, max_val, weight in vocab_valences:
                if max_val >= required_valence:
                    valid_atoms.append(symbol)
                    valid_weights.append(weight)
            
            if not valid_atoms:
                chosen = 'C' # Fallback
            else:
                # Optimization: Check if we really need random.choices (expensive)
                # If only one option, pick it.
                if len(valid_atoms) == 1:
                    chosen = valid_atoms[0]
                else:
                    sum_w = sum(valid_weights)
                    if sum_w > 0:
                        # random.choices is relatively fast, but we can optimize if needed
                        chosen = random.choices(valid_atoms, weights=valid_weights, k=1)[0]
                    else:
                        chosen = random.choice(valid_atoms)

            atom.SetAtomicNum(pt.GetAtomicNumber(chosen))
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
        
        # Sanitize to ensure validity
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None