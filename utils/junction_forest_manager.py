# --------------------------------------------------------------------
# Copyright Nicolas Nemeth, Christoph Flamm 2025
# licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

from __future__ import annotations
import random
import math

from typing import Dict, List, Tuple, Optional, Iterable, Set
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.warning")

import torch
import torch.optim as optim
import torch.nn.functional as F

from rl.config import PotentialRLConfig
from rl.metrics_interface import MetricSpec
from rl.potential_model import PotentialFunction
from rl.metrics import MetricCalculator
from utils.chemutils import reconstruct_dataset_from_rules, _as_modrule

import mod
from .junction_tree import JunctionTree, ModRule

# CONSTs
WILD_STAR: str = "*"
X_VAR: str = "X"


class JunctionForestManager:
    """
    Manages a collection of JunctionTrees (one per input SMILES) and
    performs iterative grammar induction steps:

    Iteration:
      1. Pick random JT with remaining edges.
      2. Perform one split (discover one ModRule) on that JT.
      3. Apply that ModRule to all *other* JunctionTrees.
      4. Store rule + per-JT application counts.

    Parameters
    ----------
    smiles_list : iterable of str
        Input SMILES.
    rng : random.Random, optional
        Custom RNG for deterministic behavior.
    kekulize : bool
        Forwarded to JunctionTree.from_smiles.
    compute_2D_smiles : bool
        Forwarded to JunctionTree.from_smiles.
    """

    def __init__(
        self,
        smiles_list: Iterable[str],
        rng: Optional[random.Random] = None,
        kekulize: bool = True,
        compute_2D_smiles: bool = True,
        max_iter_space_exploration: int = 75,
        use_potential_model: bool = False,
        use_frequency_bias: bool = True,
        border_depth_limit: Optional[int] = 2,
        max_atom_cutoff: Tuple[int, int] = (6, 6),
        max_rule_application_execute_strat: int=10,
        for_eligible_edges_include_wildstars: bool=False,
    ):
        self.best_rules: List[ModRule] = []
        self.best_initial_graphs: List[mod.Graph] = []
        self.max_iter_space_exploration: int = max_iter_space_exploration
        self.max_rule_application_execute_strat: int = max_rule_application_execute_strat
        self.for_eligible_edges_include_wildstars: bool = for_eligible_edges_include_wildstars
        self.use_frequency_bias: bool = use_frequency_bias
        self.use_potential_model: bool = use_potential_model
        self.kekulize: bool = kekulize
        self.compute_2D: bool = compute_2D_smiles
        self.border_depth_limit: int = border_depth_limit
        self.max_atom_cutoff: Tuple[int, int] = max_atom_cutoff
        
        self._rng: random.Random = rng or random.Random()
        self.junction_trees: Dict[str, JunctionTree] = {}
        self.discovered_rules: List[ModRule] = []
        # key: training round, value: metric dict, tuple 1st item: normalized metric, tuple 2nd item: raw metric
        self.metrics_per_training_round: Dict[int, Tuple[ Dict[str, float], Dict[str, float] ] ] = dict()
        self._jt_key_cache: Dict[int, str] = {}

        # dict of substructure-smiles : num_total_occurrance
        self.substructure_occurances: Dict[str, int] = dict()
        self.substructure_prevalences: Dict[str, int] = dict()

        # Inverted index for efficiency: substructure_smiles -> Set[jt_key]
        self.substructure_index: Dict[str, Set[str]] = defaultdict(set)
        # Forward index: jt_key -> Set[substructure_smiles]
        self.jt_substructures: Dict[str, Set[str]] = defaultdict(set)

        for smi in smiles_list:
            jt = JunctionTree.from_smiles(smi, 
                                          kekulize=self.kekulize, 
                                          compute_2D_smiles=self.compute_2D, 
                                          use_potential_model=self.use_potential_model, 
                                          border_depth_limit=self.border_depth_limit, 
                                          max_atom_cutoff=self.max_atom_cutoff, 
                                          for_eligible_edges_include_wildstars=self.for_eligible_edges_include_wildstars)
            self.junction_trees[smi] = jt
            self._reindex_jt(smi, jt)
            
        self._refresh_jt_cache()

        # RL state / placeholders
        self.potential_model: Optional[PotentialFunction] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._baseline: Optional[float] = None
        self._feature_mean: Optional[torch.Tensor] = None
        self._feature_std: Optional[torch.Tensor] = None
        self._feature_stats_dirty: bool = True

        # For resetting forest each episode:
        # original canonical smiles
        self._original_smiles: List[str] = [ Chem.MolToSmiles(self.junction_trees[smi].mol) for smi in self.junction_trees ]
        self._metric_calculator: MetricCalculator = MetricCalculator(self._original_smiles)

    def _select_random_jt(self) -> Optional[JunctionTree]:
        candidates = [
            jt for jt in self.junction_trees.values()
            if jt.edges is not None and len(jt.edges) > 0
        ]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def _refresh_jt_cache(self) -> None:
        self._jt_key_cache = {id(jt): smi for smi, jt in self.junction_trees.items()}

    def _update_jt_cache_entry(
        self,
        key: str,
        jt: JunctionTree,
        previous: Optional[JunctionTree] = None,
    ) -> None:
        if previous is not None and id(previous) != id(jt):
            self._jt_key_cache.pop(id(previous), None)
        self._jt_key_cache[id(jt)] = key

    def _get_jt_key(self, jt: JunctionTree) -> Optional[str]:
        key = self._jt_key_cache.get(id(jt))
        if key is not None:
            return key
        for smi, candidate in self.junction_trees.items():
            if candidate is jt:
                self._jt_key_cache[id(jt)] = smi
                return smi
        return None

    def set_occurrence_and_prevalence_substructure_feats(self) -> None:
        """Sets the last two columns in feature matrix given num_occurrence of each substructure"""
        eligible_junction_trees: List[JunctionTree] = [
            jt
            for jt in self.junction_trees.values()
            if jt.eligible_edges_dict is not None and len(jt.eligible_edges_dict) > 0
        ]

        if not eligible_junction_trees:
            self.substructure_occurances = {}
            return

        denominators = float(len(eligible_junction_trees))
        global_occurrence_tracker: Counter = Counter()
        prevalence_tracker: Counter = Counter()

        # Optimization: Single pass to aggregate counts
        for _jt in eligible_junction_trees:
            occurrences = _jt.substructure_occurances
            if not occurrences:
                continue
            global_occurrence_tracker.update(occurrences)
            prevalence_tracker.update(occurrences.keys())

        if self.use_potential_model:
            # Precompute feature values for all unique substructures
            smi_to_feats = {}
            for smi, total_count in global_occurrence_tracker.items():
                occurrence_value = 0.0 if total_count <= 0 else -math.expm1(-float(total_count))
                prevalence_value = (prevalence_tracker[smi] / denominators) if denominators else 0.0
                smi_to_feats[smi] = (occurrence_value, prevalence_value)

            # Vectorized update for each JT
            for _jt in eligible_junction_trees:
                substructures = _jt.substructure_list
                if not substructures:
                    continue

                gnn_feats = _jt.edge_gnn_feats
                if gnn_feats is None or gnn_feats.size == 0:
                    continue
                
                n = len(substructures)
                if n > gnn_feats.shape[0]:
                    n = gnn_feats.shape[0]
                
                # Log-scale fragment size
                sizes = np.array(_jt.substructure_sizes[:n], dtype=np.float32)
                np.log1p(sizes, out=sizes)
                
                # Lookup precomputed occurrence/prevalence
                # Using list comprehension is faster than loop with dict lookup
                feats_list = [smi_to_feats.get(s, (0.0, 0.0)) for s in substructures[:n]]
                
                # Unzip into arrays
                occs = np.array([x[0] for x in feats_list], dtype=np.float32)
                prevs = np.array([x[1] for x in feats_list], dtype=np.float32)

                # Assign columns
                gnn_feats[:n, -3] = sizes
                gnn_feats[:n, -2] = occs
                gnn_feats[:n, -1] = prevs

        self.substructure_occurances = dict(global_occurrence_tracker)
        self.substructure_prevalences = dict(prevalence_tracker)

    def _gather_edge_feature_tensor(self) -> Optional[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for jt in self.junction_trees.values():
            feats = getattr(jt, "edge_gnn_feats", None)
            if feats is None or not len(feats):
                continue
            tensors.append(torch.as_tensor(feats, dtype=torch.float32))
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _ensure_feature_stats(self, epsilon: float) -> None:
        if not self._feature_stats_dirty and self._feature_mean is not None and self._feature_std is not None:
            return
        stacked = self._gather_edge_feature_tensor()
        if stacked is None:
            self._feature_mean = None
            self._feature_std = None
            self._feature_stats_dirty = False
            return
        self._feature_mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        self._feature_std = torch.clamp(std, min=epsilon)
        self._feature_stats_dirty = False

    def _normalize_edge_features(self, feats: torch.Tensor, epsilon: float) -> torch.Tensor:
        if self._feature_mean is None or self._feature_std is None:
            return feats
        mean = self._feature_mean.to(feats.device)
        std = self._feature_std.to(feats.device)
        return (feats - mean) / (std + epsilon)

    ## RL: initialization helpers
    def _initialize_potential_model(self, feature_dim: int, cfg: PotentialRLConfig):
        self.potential_model = PotentialFunction(
            feat_dim=feature_dim,
        ).to(cfg.device, dtype=torch.float32)
        self._optimizer = optim.Adam(
            self.potential_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        self._baseline = None

    ## RL: episode-level operations
    def reset_to_original_forest(self):
        """
        USER MUST IMPLEMENT: restore junction_trees to original unsplit state.

        Current simple implementation: rebuild from stored original SMILES.
        If you need to preserve discovered rules from previous training,
        handle separately. This always resets discovered_rules list for a fresh episode.
        """
        self.junction_trees.clear()
        self.discovered_rules.clear()
        
        # Clear indexes
        self.substructure_index.clear()
        self.jt_substructures.clear()
        
        for smi in self._original_smiles:
            jt = JunctionTree.from_smiles(smi, 
                                          kekulize=self.kekulize, 
                                          compute_2D_smiles=self.compute_2D,
                                          use_potential_model=self.use_potential_model,
                                          border_depth_limit=self.border_depth_limit,
                                          max_atom_cutoff=self.max_atom_cutoff,
                                          for_eligible_edges_include_wildstars=self.for_eligible_edges_include_wildstars,
                                          )
            self.junction_trees[smi] = jt
            self._reindex_jt(smi, jt)
            
        self._refresh_jt_cache()
        self._feature_stats_dirty = True

    def _sample_jt_with_edges(self) -> Optional[JunctionTree]:
        """
        Return a random JunctionTree that still has remaining edges, or None if none exist.
        Uses the manager's internal RNG (self._rng).
        """
        candidates = [
            jt for jt in self.junction_trees.values()
            if jt.eligible_edges_dict is not None and len(jt.eligible_edges_dict) > 0
        ]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def _any_edges_left(self) -> bool:
        return any(
            jt.edges is not None and len(jt.edges) > 0
            for jt in self.junction_trees.values()
        )

    def _get_edge_probabilities(
        self,
        jt: JunctionTree,
        rl_config: PotentialRLConfig,
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes edge probabilities (policy) for the given JunctionTree.

        Returns:
            probs: Tensor of shape (num_edges,) summing to 1.
            value: Scalar tensor for state value (if model used).
            bias_factors: Tensor of bias factors used (if any).
        """
        # Determine number of edges
        # We rely on eligible_edges_dict for count if features aren't used
        num_options = len(jt.edge_gnn_ekey_split_info_tuples)
        value = None
        bias_factors = None

        if num_options == 0:
             return torch.tensor([], device=device), None, None

        # 1. Base Probabilities
        if self.use_potential_model and self.potential_model is not None:
            feats_np = jt.edge_gnn_feats
            # Safety check: if feats mismatch edges or missing
            if feats_np is None or len(feats_np) == 0:
                 return torch.ones(num_options, device=device) / num_options, None, None

            feats = torch.as_tensor(feats_np, device=device, dtype=torch.float32)
            feats = self._normalize_edge_features(feats, rl_config.whitening_epsilon)

            potentials, phi, value = self.potential_model.forward_with_phi(feats)

            sum_phi = phi.sum()
            if sum_phi.item() <= 0:
                base_probs = torch.ones_like(phi) / phi.numel()
            else:
                base_probs = phi / sum_phi

        else:
            # Uniform weights
            base_probs = torch.ones(num_options, device=device, dtype=torch.float32) / num_options

        # 2. Frequency Bias
        if self.use_frequency_bias:
            freq_alpha = getattr(rl_config, "frequency_bias_alpha", 0.0)
            # global_counts: substructure_occurances
            global_counts = getattr(self, "substructure_prevalences", None)

            # Only apply if we have counts and a positive alpha
            if freq_alpha > 0.0 and global_counts and jt.substructure_list:
                # Ensure bias_factors matches base_probs size
                bias_factors = torch.ones_like(base_probs)
                limit = min(len(jt.substructure_list), bias_factors.numel())
                alpha_val = float(freq_alpha)

                for idx in range(limit):
                    sub_smi = jt.substructure_list[idx]
                    freq = global_counts.get(sub_smi, 0)
                    # Bias: more frequent -> higher probability
                    bias_factors[idx] = (freq + 1.0) ** alpha_val

                biased_probs = base_probs * bias_factors
                total_bias = biased_probs.sum()
                if total_bias.item() > 0:
                    base_probs = biased_probs / total_bias
                else:
                    # Fallback to uniform if bias kills everything
                    base_probs = torch.ones_like(base_probs) / base_probs.numel()

        return base_probs, value, bias_factors

    def train_reinforcement(
        self,
        feature_dim: int,
        rl_config: PotentialRLConfig,
        metric_specs: List[MetricSpec],
        molecule_set: List[str],
        reconstruct_dataset_alpha: float=0.05,
        max_heavy_atom_count: Optional[int]=40,
        make_all_mols_terminal: bool=True,
        verbose: bool=False,
    ) -> Optional[torch.nn.Module]:

        device = rl_config.device

        # Initialize model only if needed
        if self.use_potential_model:
            self._initialize_potential_model(feature_dim, rl_config)
        else:
            self.potential_model = None
            self._optimizer = None

        discount_factor = getattr(
            rl_config,
            "discount_factor",
            getattr(rl_config, "gamma", getattr(rl_config, "gammar", 0.99)),
        )

        metric_calculator = getattr(self, "_metric_calculator", None)
        if metric_calculator is None:
            metric_calculator = MetricCalculator(self._original_smiles)
            self._metric_calculator = metric_calculator

        baseline_momentum = rl_config.baseline_momentum
        entropy_coef = rl_config.entropy_coef

        # PPO Hyperparams
        clip_param = getattr(rl_config, "ppo_clip_param", 0.2)
        ppo_epochs = getattr(rl_config, "ppo_epochs", 4)
        value_loss_coef = getattr(rl_config, "value_loss_coef", 0.5)

        for ep in tqdm(range(1, rl_config.max_episodes + 1), desc="RL Training Rounds"):
            if ep > 1:
                self.reset_to_original_forest()

            # Storage for PPO
            ep_obs = []
            ep_actions = []
            ep_old_log_probs = []
            ep_values = []
            ep_bias_factors = []
            
            induced_rules: List[ModRule] = []
            steps = 0

            # Only compute substructure stats if needed (for bias or model features)
            if self.use_potential_model or self.use_frequency_bias:
                self.set_occurrence_and_prevalence_substructure_feats()

            if self.use_potential_model:
                self._ensure_feature_stats(rl_config.whitening_epsilon)

            while steps < rl_config.max_steps_per_episode:
                jt = self._sample_jt_with_edges()
                if jt is None:
                    break

                # Calculate probabilities
                biased_probs, value, bias_factors = self._get_edge_probabilities(
                    jt, rl_config, device
                )

                if biased_probs.numel() == 0:
                    break

                # Sample edge
                edge_weights = biased_probs.detach().cpu().tolist()
                sum_edge_weights = sum(edge_weights)
                if sum_edge_weights > 0:
                    edge_weights = [ ew / sum_edge_weights for ew in edge_weights ]
                else:
                    # Fallback
                    edge_weights = [ 1.0 / len(edge_weights) ] * len(edge_weights)

                jt_edge_gnn_feats: Optional[np.ndarray | torch.tensor] = jt.edge_gnn_feats
                rule_obj: ModRule = jt.initiate_random_grammar_rule_induction(
                    edge_weights=edge_weights,
                    random_mode=False,
                    verbose=False,
                    only_do_once=True,
                )
                modrule = _as_modrule(rule_obj)

                if modrule is None:
                    print("Stopping because mod rule is null.")
                    break

                chosen_edge_idx: int = rule_obj.chosen_edge_index if rule_obj is not None else None
                if chosen_edge_idx is None:
                    raise Exception("The chosen edge index must never be null!")

                # Store PPO data
                if self.use_potential_model:
                    # Store features
                    feats_np = jt_edge_gnn_feats
                    feats = torch.as_tensor(feats_np, device=device, dtype=torch.float32)
                    feats = self._normalize_edge_features(feats, rl_config.whitening_epsilon)
                    
                    ep_obs.append(feats)
                    ep_actions.append(chosen_edge_idx)
                    ep_values.append(value)
                    ep_bias_factors.append(bias_factors)
                    
                    log_prob = torch.log(biased_probs[chosen_edge_idx].clamp_min(1e-12))
                    ep_old_log_probs.append(log_prob.detach())

                if modrule is not None:
                    induced_rules.append(modrule)
                    # Reindex inducing JT because it changed
                    self._reindex_jt(self._get_jt_key(jt), jt)
                    self._apply_modrule_globally(modrule, inducing_jt=jt, verbose=False)

                steps += 1
                if not self._any_edges_left():
                    break

                # Update stats if needed
                if self.use_potential_model or self.use_frequency_bias:
                    self.set_occurrence_and_prevalence_substructure_feats()

            # Keep track of the best set of rules/graphs (lowest total count)
            unique_rules: List[ModRule] = ModRule.remove_redundant_modrules(induced_rules)
            unique_graphs: List[mod.Graph] = [
                    mod.Graph.fromSMILES(
                        smi,
                        allowAbstract=True,
                        add=False,
                        printStereoWarnings=False
                    )
                    for smi in set([ jt.modGraph.smiles.replace(WILD_STAR, X_VAR) for jt in self.junction_trees.values() ])
                ]
            if len(self.best_rules) == 0 or (len(unique_graphs) + len(unique_rules)) < (len(self.best_initial_graphs) + len(self.best_rules)):
                self.best_rules = unique_rules
                self.best_initial_graphs = unique_graphs

            if verbose:
                print(f"# unique graphs: {len(unique_graphs)}\n# unique rules: {len(unique_rules)}")

            if self.use_potential_model:
                dg_iterations: int = 0
                
                final_dataset_smiles: List[str] = []
                if not all( spec.name in ("rule_score", "rule_count") for spec in metric_specs ):
                    final_dataset_smiles, dg_iterations = reconstruct_dataset_from_rules(
                        smiles=[ g.smiles for g in unique_graphs],
                        rules=unique_rules, 
                        create_summary=False, 
                        apply_all_targets=False, 
                        alpha=reconstruct_dataset_alpha,
                        max_iter_space_exploration=self.max_iter_space_exploration,
                        max_rule_application_execute_strat=self.max_rule_application_execute_strat,
                        max_heavy_atom_count=max_heavy_atom_count,
                        make_all_mols_terminal=make_all_mols_terminal,
                        original_smiles=molecule_set,
                        rule_reapplication_decay=0.15,
                        enable_beam_search=True,
                        pruning_threshold=0.3,
                        # verbose=False
                    )

                    if verbose:
                        print(f"# terminal smiles: {len(final_dataset_smiles)}")

                    if len(final_dataset_smiles) == 0:
                        print("final_dataset_smiles is empty, skipping RL reward calculation")
                        print(f"ep={ep}, steps={steps}, num_rules={len(induced_rules)}")
                        print("\n\n".join([rule.modRule.getGMLString() for rule in induced_rules]))
                        continue

                    # final_dataset_smiles: List[str] = [smi for smi in final_dataset_smiles if "*" not in smi]
                
                subset_final_dataset_smiles: List[str] = list(np.array(final_dataset_smiles)[np.random.permutation(len(final_dataset_smiles))[: min(200, int(0.2 * len(final_dataset_smiles)))]])
                
                raw_metrics = metric_calculator.evaluate_all(
                    subset_final_dataset_smiles, len(unique_graphs) + len(unique_rules), metrics_to_compute=[ spec.name for spec in metric_specs ]
                )

                self.metrics_per_training_round[ep] = (raw_metrics, raw_metrics)

                metric_contributions: Dict[str, float] = {}
                for spec in metric_specs:
                    metric_value = raw_metrics.get(spec.name)
                    if metric_value is None:
                        continue
                    direction = 1.0 if spec.higher_is_better else -1.0
                    metric_contributions[spec.name] = spec.weight * direction * metric_value

                contributions: Dict[str, float] = dict(metric_contributions)
                reward = sum(contributions.values())

                if self._baseline is None:
                    self._baseline = reward
                else:
                    self._baseline = baseline_momentum * self._baseline + (1 - baseline_momentum) * reward
                advantage = reward - self._baseline

                loss_value = 0.0

                # PPO Update
                if self.use_potential_model and ep_obs:
                    # GAE Calculation
                    gae = 0
                    advantages = []
                    # Detach values to stop gradient backprop through target
                    values_detached = [v.detach().item() for v in ep_values]
                    values_detached.append(0.0) # Value of terminal state is 0

                    gamma = discount_factor
                    lam = getattr(rl_config, "gae_lambda", 0.95)

                    # Rewards: 0 for all intermediate steps, final 'reward' for last step
                    step_rewards = [0.0] * len(ep_obs)
                    step_rewards[-1] = reward

                    for t in reversed(range(len(ep_obs))):
                        delta = step_rewards[t] + gamma * values_detached[t+1] - values_detached[t]
                        gae = delta + gamma * lam * gae
                        advantages.insert(0, gae)
                    
                    advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
                    # Returns = Advantage + Value
                    returns = advantages + torch.tensor(values_detached[:-1], device=device, dtype=torch.float32)

                    # Normalize advantages
                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # PPO Epochs
                    for _ in range(ppo_epochs):
                        self._optimizer.zero_grad()
                        policy_loss_sum = 0
                        value_loss_sum = 0
                        entropy_sum = 0
                        
                        # Iterate over steps (batch_size=1 episode)
                        for t in range(len(ep_obs)):
                            feats = ep_obs[t]
                            action = ep_actions[t]
                            old_log_prob = ep_old_log_probs[t]
                            bias = ep_bias_factors[t]
                            ret = returns[t]
                            adv = advantages[t]
                            
                            # Re-evaluate
                            potentials, phi, val = self.potential_model.forward_with_phi(feats)
                            
                            # Re-calculate probs
                            sum_phi = phi.sum()
                            if sum_phi.item() <= 0:
                                base_probs = torch.ones_like(phi) / phi.numel()
                            else:
                                base_probs = phi / sum_phi
                                
                            if bias is not None:
                                biased_probs = base_probs * bias
                                total_bias = biased_probs.sum()
                                if total_bias.item() > 0:
                                    probs = biased_probs / total_bias
                                else:
                                    probs = base_probs
                            else:
                                probs = base_probs
                                
                            new_log_prob = torch.log(probs[action].clamp_min(1e-12))
                            dist_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
                            
                            ratio = torch.exp(new_log_prob - old_log_prob)
                            surr1 = ratio * adv
                            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
                            
                            policy_loss_sum += -torch.min(surr1, surr2)
                            value_loss_sum += F.smooth_l1_loss(val.squeeze(), ret)
                            entropy_sum += dist_entropy
                            
                        # Average over steps
                        n_steps = len(ep_obs)
                        loss = (policy_loss_sum + value_loss_coef * value_loss_sum - entropy_coef * entropy_sum) / n_steps
                        
                        loss.backward()
                        if rl_config.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.potential_model.parameters(), rl_config.grad_clip)
                        self._optimizer.step()
                        
                        loss_value = loss.item()

                rl_config.entropy_coef = entropy_coef

                if ep == 1 or ep % rl_config.log_interval == 0:
                    self._log_rl_episode(
                        ep=ep,
                        steps=steps,
                        num_rules=len(induced_rules),
                        reward=reward,
                        advantage=advantage,
                        baseline=self._baseline,
                        raw_metrics=raw_metrics,
                        contributions=contributions,
                        loss=loss_value,
                        entropy_coef=rl_config.entropy_coef,
                        dg_iterations=dg_iterations,
                    )

        return self.potential_model

    ## Global rule application reused in RL
    def _apply_modrule_globally(self, modrule: ModRule, inducing_jt: JunctionTree, verbose: bool=False) -> Dict[str, int]:
        """
        Apply a newly induced ModRule to all OTHER junction trees.

        Returns
        -------
        Dict[str, int]
            Mapping of SMILES keys to the number of successful applications for this rule.
        """
        rule_to_apply = modrule.modRule
        application_counts: Dict[str, int] = {}
        substruct_smi: str = modrule.substructure_smiles
        
        # Optimization: Use inverted index to find candidate JTs
        target_keys = self.substructure_index.get(substruct_smi, set())
        # Copy keys because _reindex_jt might modify the set during iteration
        candidates = list(target_keys)
        
        for smi in candidates:
            jt = self.junction_trees[smi]
            if jt is inducing_jt:
                continue

            # Double check (index might be stale if we didn't sync perfectly, or just safety)
            # if substruct_smi not in jt.substructure_list: continue 

            count = jt.apply_rule_and_recompute(rule_to_apply, verbosity=0, recompute=True, 
                                                max_rule_application_execute_strat=self.max_rule_application_execute_strat)
            
            if count > 0:
                jt.rule_application_and_counts.append((rule_to_apply, count))
                application_counts[smi] = count
                self._update_jt_cache_entry(smi, jt)
                
                # JT changed, update index
                self._reindex_jt(smi, jt)

            if verbose:
                print(f"    Applied RL rule to {smi}: {count} time(s)")

        return application_counts

    ## Logging
    def _log_rl_episode(
        self,
        ep: int,
        steps: int,
        num_rules: int,
        reward: float,
        advantage: float,
        baseline: float,
        raw_metrics: Dict[str, float],
        contributions: Dict[str, float],
        loss: float,
        entropy_coef: float,
        dg_iterations: int,
    ):
        print("=" * 90)
        print(f"[RL] Episode {ep}")
        print(f"  Steps: {steps} | Induced Rules: {num_rules}")
        print(f"  Reward: {reward:.4f} | Advantage: {advantage:.4f} | Baseline: {baseline:.4f}")
        print(f"  DG iterations: {dg_iterations}")
        print(f"  Loss: {loss:.4f} | Entropy Coef: {entropy_coef:.5f}")
        print("  Raw Metrics:")
        for k, v in raw_metrics.items():
            print(f"    {k}: {v:.4f}")
        print("  Contributions (weighted):")
        for k, v in contributions.items():
            print(f"    {k}: {v:.4f}")
        print("=" * 90)

    def _reindex_jt(self, key: str, jt: JunctionTree) -> None:
        """Updates the inverted index for a single JunctionTree."""
        old_substructs = self.jt_substructures.get(key, set())
        new_substructs = set(jt.substructure_list)
        
        # Remove key from old substructures that are not in new
        for s in old_substructs - new_substructs:
            if s in self.substructure_index:
                self.substructure_index[s].discard(key)
                if not self.substructure_index[s]:
                    del self.substructure_index[s]
        
        # Add key to new substructures
        for s in new_substructs - old_substructs:
            self.substructure_index[s].add(key)
            
        self.jt_substructures[key] = new_substructs