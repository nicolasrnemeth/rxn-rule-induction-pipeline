from __future__ import annotations

from utils.chemutils import tanimoto_distance, evaluate_molecule_quality
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")
import numpy as np
import random

from dataclasses import dataclass
from typing import List, Iterable, Optional, Dict, Callable, Set, Tuple, Any

## CONST
WILD_STAR: str = "*"
X_VAR: str = "X"
##


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Returns RDKit canonical SMILES or None if invalid.
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _to_mol(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles) if smiles else None


def compute_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
    use_chirality: bool = False,
    use_bondtypes: bool = True,
    include_ring_membership: bool = False,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    """
    Compute Morgan fingerprints for each SMILES.

    Returns list of tuples:
      (canonical_smiles or None, Mol or None, fingerprint or None)

    Invalid molecules have (None, None, None).
    """
    out = []
    for s in smiles_list:
        mol = _to_mol(s)
        if mol is None:
            out.append((None, None, None))
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
            useBondTypes=use_bondtypes,
            includeRingMembership=include_ring_membership,
        )
        bv = gen.GetFingerprint(mol, fromAtoms=[])
        fp = _bitvect_to_numpy(bv).astype(np.float32, copy=False)
        out.append((can, mol, fp))
    return out

def _bitvect_to_numpy(bv: Chem.DataStructs.ExplicitBitVect) -> np.ndarray:
    arr = np.zeros((bv.GetNumBits(),), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


# ------------------------------------------------------------------------------------
# Metric Implementations
# ------------------------------------------------------------------------------------

def metric_validity_uniqueness_novelty(
    gen_fps: List[Tuple[str, Optional[Any], Optional[Any]]],
    initial_canonical_smiles: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Computes:
      validity:   (#valid_gen / #total_gen)
      uniqueness: (#unique_valid_gen / #valid_gen)     (0 if no valid)
      novelty:    (#valid_gen_not_in_train / #valid_gen) if training set provided else None

    gen_fps: output from compute_fingerprints on generated list
    """
    valid_cans = [can for (can, mol, fp) in gen_fps if can is not None]
    valid_cans_length = len(valid_cans)
    real_cans = [can for (can, mol, fp) in gen_fps if WILD_STAR not in can and X_VAR not in can]
    real_cans_length = len(real_cans)

    validity = real_cans_length / valid_cans_length if valid_cans_length > 0 else 0.0

    unique_valid = set(valid_cans)
    uniqueness = len(unique_valid) / valid_cans_length if valid_cans_length > 0 else 0.0

    if initial_canonical_smiles is not None and valid_cans:
        novel = [c for c in real_cans if c not in initial_canonical_smiles]
        novelty = len(novel) / real_cans_length if real_cans_length > 0 else 0.0
    else:
        novelty = float("nan")  # or None

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty
    }


def metric_diversity(fps: List[Tuple[str, Optional[Any], Optional[Any]]]) -> float:
    """
    Average pairwise Tanimoto distance among valid generated molecules.
    O(n^2); consider subsampling for very large sets.
    """
    n = len(fps)
    if n < 2:
        return 0.0
    acc: float = 0.0
    count: int = 0
    for i in range(n):
        fpi = fps[i]
        for j in range(i + 1, n):
            acc += tanimoto_distance(fpi, fps[j])
            count += 1
    return acc / count if count else 0.0

def metric_rule_count(num_rules: int) -> float:
    """
    Returns the raw count of rules.
    """
    return float(num_rules)

def metric_rule_score(num_rules: int, alpha: float = 0.1) -> float:
    """
    Returns a normalized score for the number of rules, where smaller count -> closer to 1.
    Formula: 1.0 / (1.0 + alpha * num_rules)
    """
    return 1.0 / (1.0 + alpha * float(num_rules))

def metric_chamfer_distance(
    gen_fps: List[Tuple[str, Optional[Any], Optional[Any]]],
    ref_fps: List[Tuple[str, Optional[Any], Optional[Any]]]
) -> float:
    """
    Symmetric Chamfer distance between generated set G and reference set R:
      CD(G,R) = (1/|G|) Σ_{g∈G} min_{r∈R} d(g,r) + (1/|R|) Σ_{r∈R} min_{g∈G} d(r,g)
    where d = Tanimoto distance = 1 - similarity.
    Returns 0.0 if either set has no valid fingerprints.
    NOTE: Not averaged (some works divide by 2).
    """
    # Filter to valid fingerprints (non-None)
    gen_valid = [ fp for (can, mol, fp) in gen_fps ]
    ref_valid = [ fp for (can, mol, fp) in ref_fps ]

    if not gen_valid or not ref_valid:
        return 0.0 

    # G -> R term
    sum_gr = 0.0
    for g in gen_valid:
        best = 1.0
        for r in ref_valid:
            d = tanimoto_distance(g, r)
            # Clamp & validate
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_gr += best
    term1 = sum_gr / len(gen_valid)

    # R -> G term
    sum_rg = 0.0
    for r in ref_valid:
        best = 1.0
        for g in gen_valid:
            d = tanimoto_distance(r, g)
            if d < 0.0: d = 0.0
            if d > 1.0: d = 1.0
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_rg += best
    term2 = sum_rg / len(ref_valid)

    return term1 + term2


# ------------------------------------------------------------------------------------
# Aggregator / Public API
# ------------------------------------------------------------------------------------

@dataclass
class MetricCalculator:
    training_smiles: Iterable[str]
    retro_success_fn: Optional[Callable[[str], bool]] = None
    membership_fn: Optional[Callable[[str], bool]] = None
    fp_radius: int = 3
    fp_nbits: int = 1024
    fp_chirality: bool = False
    diversity_subsample: Optional[int] = None  # optional subsample size for diversity

    def __init__(self, initial_canonical_smiles: List[str]):
        # makes sure each smiles string is in its canonical representation
        self._fingerprint_cache: Dict[Optional[str], Tuple[Optional[str], Optional[Any], Optional[Any]]] = {}
        self.initial_canonical_smiles = initial_canonical_smiles
        self.initial_canonical_fps = compute_fingerprints(
            self.initial_canonical_smiles,
            radius=self.fp_radius,
            n_bits=self.fp_nbits,
            use_chirality=self.fp_chirality
        )
        for smi, fp_tuple in zip(self.initial_canonical_smiles, self.initial_canonical_fps):
            self._fingerprint_cache[smi] = fp_tuple

    def __post_init__(self):
        # Precompute canonical training set & fingerprints
        self.training_smiles = list(self.training_smiles)
        self.training_fps = compute_fingerprints(
            self.training_smiles,
            radius=self.fp_radius,
            n_bits=self.fp_nbits,
            use_chirality=self.fp_chirality
        )
        self.training_canonical_set = {
            can for (can, mol, fp) in self.training_fps if can is not None
        }

    def _maybe_subsample(self, lst: List[Any]) -> List[Any]:
        if self.diversity_subsample and len(lst) > self.diversity_subsample:
            return random.sample(lst, self.diversity_subsample)
        return lst

    def _get_fingerprints_cached(
        self,
        smiles_list: List[str],
        *,
        radius: Optional[int] = None,
        n_bits: Optional[int] = None,
        use_chirality: Optional[bool] = None,
    ) -> List[Tuple[Optional[str], Optional[Any], Optional[Any]]]:
        if radius is None:
            radius = self.fp_radius
        if n_bits is None:
            n_bits = self.fp_nbits
        if use_chirality is None:
            use_chirality = self.fp_chirality

        to_compute: List[str] = []
        seen: Set[str] = set()
        for s in smiles_list:
            if s not in self._fingerprint_cache and s not in seen and s is not None:
                to_compute.append(s)
                seen.add(s)

        if to_compute:
            computed = compute_fingerprints(
                to_compute,
                radius=radius,
                n_bits=n_bits,
                use_chirality=use_chirality,
            )
            for s, fp_tuple in zip(to_compute, computed):
                self._fingerprint_cache[s] = fp_tuple

        results: List[Tuple[Optional[str], Optional[Any], Optional[Any]]] = []
        for s in smiles_list:
            if s is None:
                results.append((None, None, None))
                continue
            cached = self._fingerprint_cache.get(s)
            if cached is None:
                cached = (None, None, None)
                self._fingerprint_cache[s] = cached
            results.append(cached)
        return results

    def evaluate_all(
        self,
        generated_smiles: List[str],
        num_rules: Optional[int] = None,
        reference_for_chamfer: Optional[Iterable[str]] = None,
        metrics_to_compute: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Computes metrics.
        
        Args:
            generated_smiles: List of SMILES strings generated.
            num_rules: Total number of unique rules used/discovered (optional).
            reference_for_chamfer: If provided, computes Chamfer(G, reference).
                                   If None, uses training set as reference.
            metrics_to_compute: List of metric names to compute. If None, computes all default metrics.
                                Supported keys: "validity", "uniqueness", "novelty", "diversity", 
                                "chamfer_distance", "rule_count", "rule_score", "molecule_quality".
        """
        results = {}
        
        # Determine which metrics to run
        if metrics_to_compute is None:
            # Default set if not specified
            metrics_to_compute = default_metric_suite()
            
        # Pre-check dependencies
        need_fingerprints = any(m in metrics_to_compute for m in ["validity", "uniqueness", "novelty", "diversity", "chamfer_distance"])
        
        smiles_list = list(generated_smiles)
        gen_fps = []
        if need_fingerprints:
            gen_fps = self._get_fingerprints_cached(smiles_list)

        # 1. Validity / Uniqueness / Novelty
        if any(m in metrics_to_compute for m in ["validity", "uniqueness", "novelty"]):
            vun = metric_validity_uniqueness_novelty(
                gen_fps,
                initial_canonical_smiles=self.initial_canonical_smiles
            )
            for k in ["validity", "uniqueness", "novelty"]:
                if k in metrics_to_compute:
                    results[k] = vun[k]

        # 2. Diversity
        if "diversity" in metrics_to_compute:
            valid_only = [ item for item in gen_fps if item[0] is not None and WILD_STAR not in item[0] and X_VAR not in item[0] ]
            gen_fps_for_div = [ item[2] for item in valid_only ]
            if self.diversity_subsample:
                if len( valid_only ) > self.diversity_subsample:
                    import random
                    valid_only_subsample = random.sample(valid_only, self.diversity_subsample)
                    gen_fps_for_div = [ item[2] for item in valid_only_subsample ]
            results["diversity"] = metric_diversity( gen_fps_for_div )

        # 3. Chamfer Distance
        if "chamfer_distance" in metrics_to_compute:
            valid_only = [ item for item in gen_fps if item[0] is not None and WILD_STAR not in item[0] and X_VAR not in item[0] ]
            if reference_for_chamfer is None:
                ref_fps = self.initial_canonical_fps
            else:
                reference_smiles = list(reference_for_chamfer)
                ref_fps = self._get_fingerprints_cached(
                    reference_smiles,
                    radius=self.fp_radius,
                    n_bits=self.fp_nbits,
                    use_chirality=self.fp_chirality,
                )
            gen_fps_for_chamfer = [ item for item in valid_only if item[0] not in self.initial_canonical_smiles ]
            results["chamfer_distance"] = metric_chamfer_distance( gen_fps_for_chamfer, ref_fps )

        # 5. Molecule Quality (Composite Score)
        if "molecule_quality" in metrics_to_compute:
            # Calculate average quality score for valid molecules
            valid_smiles = [s for s in smiles_list if s and WILD_STAR not in s and X_VAR not in s]
            if not valid_smiles:
                results["molecule_quality"] = -100.0
            else:
                scores = [ evaluate_molecule_quality(s) for s in valid_smiles ]
                results["molecule_quality"] = sum(scores) / len(scores)

        # 4. Rule Metrics
        if num_rules is not None:
            if "rule_count" in metrics_to_compute:
                results["rule_count"] = metric_rule_count(num_rules)
            if "rule_score" in metrics_to_compute:
                results["rule_score"] = metric_rule_score(num_rules)

        # Always include counts for logging/debugging if fingerprints were computed
        if need_fingerprints:
            valid_only = [ item for item in gen_fps if item[0] is not None and WILD_STAR not in item[0] and X_VAR not in item[0] ]
            results["num_generated"] = len(smiles_list)
            results["num_valid"] = len(valid_only)

        return results


# ------------------------------------------------------------------------------------
# Convenience: Return names of default metrics (keys of evaluate_all output)
# ------------------------------------------------------------------------------------

def default_metric_suite() -> List[str]:
    return [
        "validity",
        "uniqueness",
        "novelty",
        "diversity",
        "chamfer_distance",
        "num_generated",
        "num_valid",
    ]