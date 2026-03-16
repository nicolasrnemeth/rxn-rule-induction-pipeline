# --------------------------------------------------------------------
# Copyright Nicolas Nemeth, Christoph Flamm 2025
# licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

import json
import os
import random

import torch
import mod
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.info")

from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .chemutils import (
    get_internal_to_external_id_map,
    get_nxgraph_from_mod_graph,
    get_junction_tree,
    create_modsubgraph,
    draw_junction_graph,
    display_mol,
    modify_smiles,
    get_num_wildstars,
    create_modsubgraph_original,
    find_vertex_by_id,
    get_molecule_embeddings
)
from pretraingnn.model_provider import get_model
from pretraingnn.model import GNN_graphpred
from .utils import replace_numbers_in_string
from .mod_rule import ModRule
from .tree_node import TreeNode

## CONST

X_VAR = "X"
WILD_STAR = "*"

##

class SplitInfo:
    # continue with left junction tree and intersection is taken from left
    JT_LEFT_INTERSECTION_LEFT = 1
    # continue with left jt and intersection is taken from right
    JT_LEFT_INTERSECTION_RIGHT = 2
    # continue with right jt and intersection is taken from left
    JT_RIGHT_INTERSECTION_LEFT = 3
    # continue with right jt and intersection is taken from right
    JT_RIGHT_INTERSECTION_RIGHT = 4

class JunctionTree(object):
    """
    Maintains junction tree representation for a given molecule, provided as smiles string.
    """
    # Singletons
    GNN_MODEL: GNN_graphpred = get_model()
    # with open("utils/data/buyables.json", "r") as f:
    #     buyables = json.load(f)
    # BUYABLES_SET: Set[str] = { buyable["smiles"].strip() for buyable in buyables if buyable["smiles"].strip() != "" }
    
    # Cache for RDKit Mol objects of buyables, grouped by heavy atom count
    # BUYABLES_MOLS_BY_SIZE: Dict[int, List[Chem.Mol]] = defaultdict(list)
    # for s in BUYABLES_SET:
    #     m = Chem.MolFromSmiles(s, sanitize=False)
    #     if m:
    #         try:
    #             m.UpdatePropertyCache(strict=False)
    #             BUYABLES_MOLS_BY_SIZE[m.GetNumHeavyAtoms()].append(m)
    #         except:
    #             pass

    def __init__(
        self,
        modGraph: mod.Graph,
        vertices: List[TreeNode],
        cliques: Dict[int, List[int]],
        edges: Dict[int, Tuple[int, int]],
        internal_to_external_id: Optional[Dict[int, int]] = None,
        use_potential_model: bool = False,
        border_depth_limit: Optional[int] = 2,  # maximum hops from any leaf
        max_atom_cutoff: Tuple[int, int] = (6, 6),    # cap external atom count per side
        for_eligible_edges_include_wildstars=True,
        kekulize: bool=True,
        compute_2D_smiles: bool=True,
    ) -> 'JunctionTree':
        """
        Args:
            modGraph: mod graph representation of smiles string
            vertices: the list of `TreeNode`s
            cliques: the list of cliques
            edges: list of edges
        """
        # junction trees that have been split off from the original junction tree
        self.junction_trees_split_off: List[JunctionTree] = []
        self.use_potential_model: bool = use_potential_model
        self.for_eligible_edges_include_wildstars: bool = for_eligible_edges_include_wildstars
        self.rule_application_and_counts: List[Tuple[ModRule, int]] = []

        self.modGraph: mod.Graph = modGraph
        self.tree_vertices: List[TreeNode] = vertices
        self.cliques, self.edges = cliques, edges

        self.original_atom_map_numbers: Dict[int, int] = {}
        self.modRules: List[ModRule] = []
        self.uniqueModRules: List[ModRule] = []
        self.curr_split_id: int = 0
        
        self.border_depth_limit: int = border_depth_limit
        self.max_atom_cutoff: int = max_atom_cutoff

        # ID maps internal <-> external
        if internal_to_external_id is None:
            self.internal_to_external_id: Dict[int, int] = get_internal_to_external_id_map(self.modGraph)
        else:
            # copy
            self.internal_to_external_id = dict(internal_to_external_id)
        self.external_to_internal_id: Dict[int, int] = {ext: intl for intl, ext in self.internal_to_external_id.items()}

        mapped_smiles_with_ids = replace_numbers_in_string(modGraph.smilesWithIds, self.internal_to_external_id)
        self.mol: Chem.Mol = Chem.MolFromSmiles(mapped_smiles_with_ids.replace(X_VAR, WILD_STAR), sanitize=True)
        self.nxgraph: nx.Graph = get_nxgraph_from_mod_graph(modGraph)

        self.recomputation_split_ids: List[int] = []
        self.selected_edges: Set[Tuple[int, int]] = set()
        self.selected_edge_before_recompute: Optional[Tuple[int, int]] = None
        
        # number of occurances of each substructure in molecule
        self.substructure_list: List[str] = []
        self.substructure_occurances: Dict[str, int] = {}
        self.substructure_sizes: List[float] = []
        
        # Machine Learning Features
        # GNN
        self.edge_gnn_feats: Optional[np.ndarray | torch.tensor] = None
        self.edge_gnn_ekey_split_info_tuples: List[Tuple[int, int]] = [] 
        
        self.left_right_intersection_atom_id_sets: Dict[int, Tuple[Set[int], Set[int], Set[int]]] = None
        self.eligible_edges_dict: Dict[int, Tuple[int, int, List[int]]] = None

        self._cached_adjacent_cliques: Optional[Dict[int, Tuple[int, ...]]] = None
        self._cached_clique_sets: Optional[Dict[int, frozenset[int]]] = None
        self._cached_edge_split_atom_sets: Optional[Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None
        self._mod_vertex_cache: Dict[int, Any] = {}
        
        self.kekulize: bool = kekulize
        self.compute_2D_smiles: bool = compute_2D_smiles

        # Register reference to JT on each TreeNode that belongs to it
        for tn in self.tree_vertices:
            tn.set_junction_tree(self)

    @classmethod
    def from_smiles(cls, smiles: str, kekulize: bool=True, compute_2D_smiles: bool=True, 
                    recompute_modGraph_dueto_extids: bool=True, use_potential_model: bool=False,
                    border_depth_limit: Optional[int]=2, max_atom_cutoff: Tuple[int,int]=(6,6), 
                    for_eligible_edges_include_wildstars: bool=False) -> 'JunctionTree':
        """
        Args:
            - smiles (str): the input molecule as smiles string
            - kekulize (bool, Default: True): whether to kekulize smiles string/molecule or not
        Return:
            The junction tree (JunctionTree) created given the smiles string.
        """
        smiles = modify_smiles(smiles, kekulize, compute_2D_smiles)
        modGraph: mod.Graph = mod.Graph.fromSMILES(smiles, allowAbstract=True, add=False, printStereoWarnings=False)
        if recompute_modGraph_dueto_extids:
            modGraph: mod.Graph = mod.Graph.fromSMILES(modGraph.smilesWithIds, allowAbstract=True, add=False, printStereoWarnings=False)
        internal_to_external_id: Dict[int, int] = get_internal_to_external_id_map(modGraph)

        cliques, edges = get_junction_tree(modGraph, internal_to_external_id)

        # Build TreeNode list
        vertices: List[TreeNode] = []
        root = 0
        for cid, c in cliques.items():
            mg_subgraph = create_modsubgraph_original(modGraph, {cid: c}, internal_to_external_id)
            vertex = TreeNode(mg_subgraph, cid, c)
            vertices.append(vertex)
            if min(c) == 0:
                root = cid

        # build for quick lookup of mapping from clique id to TreeNode
        cid2pos: Dict[int, int] = {v.clique_id: i for i, v in enumerate(vertices)}

        # Connect neighbors
        for (_k, (x, y)) in edges.items():
            vx = vertices[cid2pos[x]]
            vy = vertices[cid2pos[y]]
            vx.add_neighbor(vy)
            vy.add_neighbor(vx)

        # Ensure root at position 0 for determinism
        if root > 0:
            root_pos = cid2pos[root]
            vertices[0], vertices[root_pos] = vertices[root_pos], vertices[0]

        jt = cls(
            modGraph=modGraph,
            vertices=vertices,
            cliques=cliques,
            edges=edges,
            internal_to_external_id=internal_to_external_id,
            use_potential_model=use_potential_model,
            border_depth_limit=border_depth_limit,
            max_atom_cutoff=max_atom_cutoff,
            for_eligible_edges_include_wildstars=for_eligible_edges_include_wildstars,
            kekulize=kekulize,
            compute_2D_smiles=compute_2D_smiles,
        )
        jt.left_right_intersection_atom_id_sets = jt.collect_edge_split_atom_sets(True)
        jt.eligible_edges_dict = jt.get_eligible_edges()
        
        # Set machine learning features on juction tree
        jt.compute_edge_gnn_feats()
        
        jt.update_vertex_is_leaf()
        jt.reset_atom_map_numbers()
        
        jt.set_is_wildstar_flags()
        for tn in jt.tree_vertices:
            tn.set_junction_tree(jt)
        return jt

    @property
    def size(self) -> int:
        return len(self.tree_vertices)

    @staticmethod
    def find_treenode_pos_by_clique_id(target_clique_id: int, vertices: List[TreeNode]) -> Optional[int]:
        """Finds position of a tree node based on clique id."""
        for pos, node in enumerate(vertices):
            if node.clique_id == target_clique_id:
                return pos
        return None

    def get_tree_vertex_by_cid(self, cid: int) -> TreeNode:
        """
        Retrieves the tree vertex from self.tree_vertices by the clique id.
        """
        # Faster and clearer than next(filter(...))
        for n in self.tree_vertices:
            if n.clique_id == cid:
                return n
        raise StopIteration(f"TreeNode with clique_id={cid} not found")

    def update_vertex_is_leaf(self) -> None:
        # local attr lookup
        for vertex in self.tree_vertices:
            vertex.is_leaf = (len(vertex.neighbors) == 1)

    def _ensure_structure_caches(self) -> None:
        if self._cached_adjacent_cliques is None:
            adjacency_lists: Dict[int, List[int]] = defaultdict(list)
            for _, (u, v) in self.edges.items():
                adjacency_lists[u].append(v)
                adjacency_lists[v].append(u)
            adjacency = {cid: tuple(neis) for cid, neis in adjacency_lists.items()}
            for cid in self.cliques:
                adjacency.setdefault(cid, tuple())
            self._cached_adjacent_cliques = adjacency
        if self._cached_clique_sets is None:
            self._cached_clique_sets = {cid: frozenset(atom_ids) for cid, atom_ids in self.cliques.items()}

    def _get_all_edge_split_atom_id_sets(self) -> Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        if self._cached_edge_split_atom_sets is None:
            self._cached_edge_split_atom_sets = self._compute_edge_split_atom_sets()
        return self._cached_edge_split_atom_sets

    def _compute_edge_split_atom_sets(self) -> Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
        self._ensure_structure_caches()
        adjacency = self._cached_adjacent_cliques or {}
        clique_sets = self._cached_clique_sets or {}

        # Optimization: O(N) Tree Traversal instead of O(N^2) BFS
        results: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
        
        if not self.cliques:
            return results

        # Map edge (u, v) -> eid for quick lookup
        edge_lookup = {frozenset((u, v)): eid for eid, (u, v) in self.edges.items()}
        
        # 1. Calculate all atoms once
        all_atoms = set().union(*clique_sets.values())
        
        # 2. DFS to establish parent-child relationships and visit order
        root_id = next(iter(self.cliques))
        dfs_stack = [root_id]
        visited = {root_id}
        parent_map = {root_id: None}
        dfs_order = []
        
        while dfs_stack:
            u = dfs_stack.pop()
            dfs_order.append(u)
            for v in adjacency.get(u, []):
                if v not in visited:
                    visited.add(v)
                    parent_map[v] = u
                    dfs_stack.append(v)
        
        # 3. Aggregate atoms from leaves up (Reverse DFS order)
        subtree_atoms: Dict[int, Set[int]] = {}
        
        for u in reversed(dfs_order):
            # Start with atoms in this clique
            u_atoms = set(clique_sets[u])
            
            # Add atoms from all children
            for v in adjacency.get(u, []):
                if parent_map.get(v) == u:
                    u_atoms.update(subtree_atoms[v])
            
            subtree_atoms[u] = u_atoms
            
            # If not root, compute split for edge to parent
            p = parent_map[u]
            if p is not None:
                eid = edge_lookup.get(frozenset((u, p)))
                if eid is not None:
                    # Interface
                    interface = clique_sets[u].intersection(clique_sets[p])
                    
                    # Child side (u)
                    child_side = u_atoms
                    
                    # Parent side (p) = All - Child + Interface
                    parent_side = (all_atoms - child_side) | interface
                    
                    # Match to edge definition (u_def, v_def)
                    u_def, v_def = self.edges[eid]
                    
                    # No sorting needed here as it is converted to set later
                    t_child = tuple(child_side)
                    t_parent = tuple(parent_side)
                    t_inter = tuple(interface)
                    
                    if u_def == u:
                        # Edge is (u, p) -> (child, parent)
                        results[eid] = (t_child, t_parent, t_inter)
                    else:
                        # Edge is (p, u) -> (parent, child)
                        results[eid] = (t_parent, t_child, t_inter)
                        
        return results

    @staticmethod
    def _materialize_split_sets(parts: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]) -> Tuple[Set[int], Set[int], Set[int]]:
        left_raw, right_raw, interface_raw = parts
        return set(left_raw), set(right_raw), set(interface_raw)

    def _get_vertex(self, external_id: int):
        cached = self._mod_vertex_cache.get(external_id)
        if cached is None:
            cached = self.modGraph.getVertexFromExternalId(external_id)
            self._mod_vertex_cache[external_id] = cached
        return cached

    def compute_edge_gnn_feats(self) -> None:
        """Computes GNN edge feature matrices consisting of left and right substructure embeddings for all eligible junction-tree edges."""
        # Get current (1st item) graph and (2nd item) node embeddings of molecule
        model: Optional[GNN_graphpred] = None
        node_embeddings: Optional[torch.Tensor] = None
        graph_embedding: Optional[torch.Tensor] = None
        pool_subset: Optional[Callable] = None
        
        if self.use_potential_model:
            model = JunctionTree.GNN_MODEL
            graph_embedding, node_embeddings = get_molecule_embeddings(model, self.mol, device='cpu')
            pool_subset = model.pool_subset
        
        # Reset cached substructure metadata before recomputing emsbeddings
        self.substructure_list = []
        self.substructure_occurances = {}
        self.substructure_sizes = []

        if not any(a.GetAtomMapNum() > 0 for a in self.mol.GetAtoms()):
            if hasattr(self, "set_atom_map_numbers"):
                self.set_atom_map_numbers()
        ext2rd: Dict[int, int] = {a.GetAtomMapNum(): a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomMapNum() > 0}
        if not ext2rd:
            raise RuntimeError(
                "RDKit Mol has no atom-map numbers. Initialize mol from SMILES with IDs or restore map numbers."
            )
            
        rd_index_cache: Dict[frozenset[int], Tuple[int, ...]] = {}

        def map_ext2rd(ids_to_map: Set[int]) -> Tuple[int, ...]:
            if not ids_to_map:
                return tuple()
            key = frozenset(ids_to_map)
            cached = rd_index_cache.get(key)
            if cached is not None:
                return cached
            mapped = tuple(sorted(ext2rd[i] for i in ids_to_map if i in ext2rd))
            rd_index_cache[key] = mapped
            return mapped
        
        # 1) Gather per-edge left/right/interface external-id sets
        left_right_intersection_atom_id_sets: Dict[int, Tuple[Set[int], Set[int], Set[int]]] = self.collect_edge_split_atom_sets()
        if self.use_potential_model and len(left_right_intersection_atom_id_sets) == 0: #ninenote: add raise since this should not happen
            feat_dim = node_embeddings.shape[-1] + graph_embedding.shape[-1] + 3
            self.edge_gnn_feats = np.zeros((0, feat_dim), dtype=np.float32)
            return
        
        if self.eligible_edges_dict is None:
            self.eligible_edges_dict = self.get_eligible_edges()

        eligible_edges_dict = self.eligible_edges_dict or {}
        if any(eid not in eligible_edges_dict for eid in left_right_intersection_atom_id_sets):
            eligible_edges_dict = self.get_eligible_edges()
            self.eligible_edges_dict = eligible_edges_dict

        if self.use_potential_model and not eligible_edges_dict:
            feat_dim = node_embeddings.shape[-1] + graph_embedding.shape[-1] + 3
            self.edge_gnn_feats = np.zeros((0, feat_dim), dtype=np.float32)
            self.edge_gnn_ekey_split_info_tuples = []
            return

        self._ensure_structure_caches()
        cached_adjacency = self._cached_adjacent_cliques or {}

        def compute_clique_depths() -> Dict[int, int]:
            depths: Dict[int, int] = {}
            queue: deque[int] = deque()

            for cid, neighbors in cached_adjacency.items():
                if len(neighbors) <= 1:
                    depths[cid] = 0
                    queue.append(cid)

            if not queue:
                for cid in self.cliques:
                    depths[cid] = 0
                    queue.append(cid)

            while queue:
                node = queue.popleft()
                node_depth = depths[node]
                for nei in cached_adjacency.get(node, ()):  # tuples stored for speed
                    if nei in depths:
                        continue
                    depths[nei] = node_depth + 1
                    queue.append(nei)
            return depths

        if self.border_depth_limit is not None:
            clique_depths = compute_clique_depths()

        subgraph_cache: Dict[frozenset[int], Optional[mod.Graph]] = {}

        def register_atoms(atom_ids: Set[int]) -> bool:
            if not atom_ids:
                return False
            key = frozenset(atom_ids)
            subgraph = subgraph_cache.get(key)
            if subgraph is None:
                subgraph = create_modsubgraph_original(
                    self.modGraph,
                    {0: sorted(atom_ids)},
                    self.internal_to_external_id,
                )

                # Fallback: if MØD fails (e.g. disconnected graph), use RDKit to get SMILES
                if subgraph is None:
                    try:
                        rd_ids = map_ext2rd(atom_ids)
                        if rd_ids:
                            # Generate SMILES for the fragment
                            frag_smiles = Chem.MolFragmentToSmiles(self.mol, rd_ids, kekuleSmiles=True, canonical=True)
                            # Create a dummy object that mimics mod.Graph for _register_substructure
                            class DummyModGraph:
                                def __init__(self, s): self.smiles = s
                            subgraph = DummyModGraph(frag_smiles)
                    except Exception:
                        pass

                subgraph_cache[key] = subgraph
                
            if subgraph is not None:# and subgraph.smiles in JunctionTree.BUYABLES_SET:
                self._register_substructure(subgraph)
                return True
                
            #     if X_VAR in subgraph.smiles or WILD_STAR in subgraph.smiles:
            #         try:
            #             query_smiles = subgraph.smiles.replace(X_VAR, WILD_STAR)
            #             query_mol = Chem.MolFromSmarts(query_smiles)
            #             if query_mol:
            #                 params = Chem.AdjustQueryParameters()
            #                 params.adjustDegree = False
            #                 params.makeDummiesQueries = True
            #                 query_mol = Chem.AdjustQueryProperties(query_mol, params)
            #                 query_num_atoms = query_mol.GetNumHeavyAtoms() + query_smiles.count(WILD_STAR)
                            
            #                 candidates = JunctionTree.BUYABLES_MOLS_BY_SIZE.get(query_num_atoms, [])
            #                 for buyable_mol in candidates:
            #                     if buyable_mol.HasSubstructMatch(query_mol, useChirality=False):
            #                         self._register_substructure(subgraph)
            #                         return True
            #         except Exception:
            #             pass
            return False

        edge_ids_sorted = sorted( left_right_intersection_atom_id_sets.keys() )
        ekey_split_info_tuples: List[Tuple[int, int]] = []
        rows_edge_gnn_feats: List[torch.Tensor] = []

        # 3) Build matrix rows per edge, each edge results in one or two feature vectors one incl. / one excl. intersection
        for eid in edge_ids_sorted:
            L_atoms_incl, R_atoms_incl, I_atoms = left_right_intersection_atom_id_sets[eid]
            
            edge_entry = eligible_edges_dict.get(eid)
            if edge_entry is None:
                continue

            u_cid, v_cid = self.edges[eid]
            depth_condition_left: bool = self.border_depth_limit is None or clique_depths.get(u_cid, self.border_depth_limit + 1) <= self.border_depth_limit
            depth_condition_right: bool = self.border_depth_limit is None or clique_depths.get(v_cid, self.border_depth_limit + 1) <= self.border_depth_limit
            size_condition_left: bool = self.max_atom_cutoff[0] <= len(R_atoms_incl) and len(R_atoms_incl) <= self.max_atom_cutoff[1]
            size_condition_right: bool = self.max_atom_cutoff[0] <= len(L_atoms_incl) and len(L_atoms_incl) <= self.max_atom_cutoff[1]

            _, _, possible_split_infos = edge_entry
            assert len(possible_split_infos) > 0
            
            left_rd_ids_1: Optional[Tuple[int, ...]] = None
            left_rd_ids_2: Optional[Tuple[int, ...]] = None
            if size_condition_left and depth_condition_left:
                if SplitInfo.JT_LEFT_INTERSECTION_LEFT in possible_split_infos:
                    if register_atoms(R_atoms_incl):
                        left_rd_ids_1 = map_ext2rd(R_atoms_incl)
                        ekey_split_info_tuples.append((eid, SplitInfo.JT_LEFT_INTERSECTION_LEFT))
                if SplitInfo.JT_LEFT_INTERSECTION_RIGHT in possible_split_infos:
                    R_atoms_excl = R_atoms_incl - I_atoms
                    if register_atoms(R_atoms_excl):
                        left_rd_ids_2 = map_ext2rd(R_atoms_excl)
                        ekey_split_info_tuples.append((eid, SplitInfo.JT_LEFT_INTERSECTION_RIGHT))
                
            right_rd_ids_1: Optional[Tuple[int, ...]] = None
            right_rd_ids_2: Optional[Tuple[int, ...]] = None
            if size_condition_right and depth_condition_right:
                if SplitInfo.JT_RIGHT_INTERSECTION_RIGHT in possible_split_infos:
                    if register_atoms(L_atoms_incl):
                        right_rd_ids_1 = map_ext2rd(L_atoms_incl)
                        ekey_split_info_tuples.append((eid, SplitInfo.JT_RIGHT_INTERSECTION_RIGHT))
                if SplitInfo.JT_RIGHT_INTERSECTION_LEFT in possible_split_infos:
                    L_atoms_excl = L_atoms_incl - I_atoms
                    if register_atoms(L_atoms_excl):
                        right_rd_ids_2 = map_ext2rd(L_atoms_excl)
                        ekey_split_info_tuples.append((eid, SplitInfo.JT_RIGHT_INTERSECTION_LEFT))
            
            if self.use_potential_model:
                # Helper to get embedding for a set of atoms
                def get_embedding(atom_ids: Set[int]) -> torch.Tensor:
                    if not atom_ids:
                        return torch.zeros((1, node_embeddings.shape[-1]), device=node_embeddings.device)
                    rd_ids = map_ext2rd(atom_ids)
                    if not rd_ids:
                        return torch.zeros((1, node_embeddings.shape[-1]), device=node_embeddings.device)
                    return pool_subset(node_embeddings, rd_ids).view(1, -1)

                # Precompute embeddings for this edge's components
                L_excl = L_atoms_incl - I_atoms
                R_excl = R_atoms_incl - I_atoms
                
                emb_L_incl = get_embedding(L_atoms_incl)
                emb_L_excl = get_embedding(L_excl)
                emb_R_incl = get_embedding(R_atoms_incl)
                emb_R_excl = get_embedding(R_excl)
                emb_I = get_embedding(I_atoms)
                
                n_total = float(self.mol.GetNumAtoms()) if self.mol.GetNumAtoms() > 0 else 1.0
                
                def make_feat_row(emb_frag, emb_comp, n_frag, n_comp):
                    size_feats = torch.tensor([
                        n_frag / n_total,
                        n_comp / n_total,
                        len(I_atoms) / n_total
                    ], device=graph_embedding.device).view(1, -1)
                    
                    # Feature Vector: [Fragment, Complement, Intersection, WholeGraph, SizeFeats]
                    return torch.cat([
                        emb_frag, emb_comp, emb_I, 
                        graph_embedding.view(1, -1), 
                        size_feats
                    ], dim=1)

                if left_rd_ids_1 is not None and size_condition_left and depth_condition_left:
                    # Case 1: Frag=R_incl, Comp=L_excl
                    row = make_feat_row(emb_R_incl, emb_L_excl, len(R_atoms_incl), len(L_excl))
                    rows_edge_gnn_feats.append(row)
                
                if left_rd_ids_2 is not None and size_condition_left and depth_condition_left:
                    # Case 2: Frag=R_excl, Comp=L_incl
                    row = make_feat_row(emb_R_excl, emb_L_incl, len(R_excl), len(L_atoms_incl))
                    rows_edge_gnn_feats.append(row)

                if right_rd_ids_1 is not None and size_condition_right and depth_condition_right:
                    # Case 3: Frag=L_incl, Comp=R_excl
                    row = make_feat_row(emb_L_incl, emb_R_excl, len(L_atoms_incl), len(R_excl))
                    rows_edge_gnn_feats.append(row)

                if right_rd_ids_2 is not None and size_condition_right and depth_condition_right:
                    # Case 4: Frag=L_excl, Comp=R_incl
                    row = make_feat_row(emb_L_excl, emb_R_incl, len(L_excl), len(R_atoms_incl))
                    rows_edge_gnn_feats.append(row)
        
        self.edge_gnn_ekey_split_info_tuples = ekey_split_info_tuples
        
        if self.use_potential_model:
            if rows_edge_gnn_feats:
                stacked = torch.cat(rows_edge_gnn_feats, dim=0)
                base_feats = stacked.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                # 3 * node_dim + graph_dim + 3 size features
                feat_dim = 3 * node_embeddings.shape[-1] + graph_embedding.shape[-1] + 3
                base_feats = np.zeros((0, feat_dim), dtype=np.float32)

            extra_cols = np.zeros((base_feats.shape[0], 3), dtype=np.float32)
            self.edge_gnn_feats = np.concatenate((base_feats, extra_cols), axis=1)

            if self.edge_gnn_feats.ndim == 1:
                self.edge_gnn_feats = self.edge_gnn_feats[None, :]

    def _register_substructure(self, substructure_mod_graph: mod.Graph) -> str:
        """Track fragment statistics so auxiliary features stay aligned with embeddings."""
        substructure_smiles: str = substructure_mod_graph.smiles
        self.substructure_list.append(substructure_smiles)
        self.substructure_occurances[substructure_smiles] = (
            self.substructure_occurances.get(substructure_smiles, 0) + 1
        )
        self.substructure_sizes.append(self._estimate_fragment_size(substructure_mod_graph))
        return substructure_smiles

    @staticmethod
    def _estimate_fragment_size(substructure_mod_graph: mod.Graph) -> float:
        """Return heavy-atom count (fallback: non-placeholder vertex count)."""
        smiles = substructure_mod_graph.smiles.replace(X_VAR, WILD_STAR)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return float(mol.GetNumHeavyAtoms())
        except Exception:
            pass

        heavy_count = 0
        placeholders = { WILD_STAR, X_VAR }
        vertices = getattr(substructure_mod_graph, "vertices", [])
        for vertex in vertices:
            label = getattr(vertex, "stringLabel", "")
            if label in placeholders:
                continue
            atom_id = getattr(vertex, "atomId", None)
            if atom_id is not None and getattr(atom_id, "symbol", None) == "H":
                continue
            heavy_count += 1
        return float(heavy_count)

    def collect_edge_split_atom_sets(self, for_all_edges: bool=False) -> Dict[int, Tuple[Set[int], Set[int], Set[int]]]:
        all_splits_raw = self._get_all_edge_split_atom_id_sets()
        if for_all_edges:
            splits = {eid: self._materialize_split_sets(parts) for eid, parts in all_splits_raw.items()}
            self.left_right_intersection_atom_id_sets = splits
            return splits

        if self.eligible_edges_dict is None:
            self.eligible_edges_dict = self.get_eligible_edges()

        eligible_keys = self.eligible_edges_dict.keys() if self.eligible_edges_dict else all_splits_raw.keys()
        return {
            eid: self._materialize_split_sets(all_splits_raw[eid])
            for eid in eligible_keys
            if eid in all_splits_raw
        }

    def get_eligible_edges(self) -> Dict[int, Tuple[int, int, List[int]]]:
        if self.left_right_intersection_atom_id_sets is None:
            self.left_right_intersection_atom_id_sets = self.collect_edge_split_atom_sets(for_all_edges=True)

        left_right_sets = self.left_right_intersection_atom_id_sets or {}
        cid2node = {v.clique_id: v for v in self.tree_vertices}

        eligible_edges: Dict[int, Tuple[int, int, List[int]]] = {}
        
        # Optimization: Pre-fetch all labels to avoid repeated lookups
        label_cache: Dict[int, str] = {}
        if hasattr(self.modGraph, 'vertices'):
             for v in self.modGraph.vertices:
                 label_cache[v.id] = v.stringLabel

        def atom_label(atom_id: int) -> str:
            # Fallback if not in cache (should not happen if modGraph is consistent)
            return label_cache.get(atom_id, "")

        placeholder_labels = { WILD_STAR, X_VAR }

        for e_key, (left_set, right_set, interface_set) in left_right_sets.items():
            left_non_interface = left_set - interface_set
            right_non_interface = right_set - interface_set

            if self.for_eligible_edges_include_wildstars:
                left_only_wildstars = False
                right_only_wildstars = False
                interface_only_wildstars = False
            else:
                left_only_wildstars = all(atom_label(aidx) in placeholder_labels for aidx in left_non_interface)
                right_only_wildstars = all(atom_label(aidx) in placeholder_labels for aidx in right_non_interface)
                interface_only_wildstars = all(atom_label(aidx) in placeholder_labels for aidx in interface_set)

            v1_id, v2_id = self.edges[e_key]
            if v1_id > v2_id:
                v1_id, v2_id = v2_id, v1_id

            v1 = cid2node[v1_id]
            v2 = cid2node[v2_id]

            if (left_only_wildstars and len(v2.clique) == 1) or (right_only_wildstars and len(v1.clique) == 1):
                continue

            if left_only_wildstars:
                if right_only_wildstars:
                    if interface_only_wildstars:
                        continue
                    split_infos = [
                        SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                        SplitInfo.JT_RIGHT_INTERSECTION_RIGHT,
                    ]
                else:
                    if interface_only_wildstars:
                        split_infos = [
                            SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                            SplitInfo.JT_LEFT_INTERSECTION_RIGHT,
                        ]
                    else:
                        split_infos = [
                            SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                            SplitInfo.JT_LEFT_INTERSECTION_RIGHT,
                            SplitInfo.JT_RIGHT_INTERSECTION_RIGHT,
                        ]
                eligible_edges[e_key] = (v1_id, v2_id, split_infos)
            elif right_only_wildstars:
                if interface_only_wildstars:
                    split_infos = [
                        SplitInfo.JT_RIGHT_INTERSECTION_LEFT,
                        SplitInfo.JT_RIGHT_INTERSECTION_RIGHT,
                    ]
                else:
                    split_infos = [
                        SplitInfo.JT_RIGHT_INTERSECTION_LEFT,
                        SplitInfo.JT_RIGHT_INTERSECTION_RIGHT,
                        SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                    ]
                eligible_edges[e_key] = (v1_id, v2_id, split_infos)
            else:
                split_infos = [
                    SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                    SplitInfo.JT_LEFT_INTERSECTION_RIGHT,
                    SplitInfo.JT_RIGHT_INTERSECTION_LEFT,
                    SplitInfo.JT_RIGHT_INTERSECTION_RIGHT,
                ]
                eligible_edges[e_key] = (v1_id, v2_id, split_infos)

        self.eligible_edges_dict = eligible_edges
        return eligible_edges

    def display_graph(self, fig_size: Tuple[int, int] = (4, 4)) -> None:
        """Visualizes the junction graph/(tree)."""
        draw_junction_graph(self.cliques, self.edges, fig_size)

    def random_edge(self, weights: Optional[List[float]]=None, only_do_once: bool=False) -> Optional[Dict[int, Tuple[int, int] | Dict[int, Tuple[int, int, int]]]]:
        """
        Sample one eligible edge for splitting.
        """
        if only_do_once:
            if not self.edge_gnn_ekey_split_info_tuples or not self.eligible_edges_dict:
                return None

            valid_indices = [
                idx for idx, (ekey, _)
                in enumerate(self.edge_gnn_ekey_split_info_tuples)
            ]

            if not valid_indices:
                return None

            if weights is None:
                chosen_pos = int(np.random.choice(valid_indices, size=1, replace=True)[0])
            else:
                weights_arr = np.asarray(weights, dtype=float)
                if weights_arr.shape[0] != len(self.edge_gnn_ekey_split_info_tuples):
                    raise ValueError(
                        "Length of weights does not match edge_gnn_ekey_split_info_tuples."
                    )
                sub_w = weights_arr[valid_indices]
                sub_w[~np.isfinite(sub_w)] = 0.0
                if (sub_w < 0).any():
                    sub_w = np.clip(sub_w, 0, None)
                w_sum = sub_w.sum()
                if w_sum <= 0:
                    sub_probs = np.ones_like(sub_w) / sub_w.shape[0]
                else:
                    sub_probs = sub_w / w_sum
                chosen_pos = int(np.random.choice(valid_indices, size=1, p=sub_probs, replace=True)[0])

            ekey, spinfo = self.edge_gnn_ekey_split_info_tuples[chosen_pos]
            e1, e2, _ = self.eligible_edges_dict[ekey]
            return { ekey: (e1, e2, spinfo) }
         
        eligible_keys: List[int] = list( self.eligible_edges_dict.keys() )
        if not eligible_keys:
            return None

        # Case 1: No weights provided
        if weights is None:
            e_key = random.choice(eligible_keys)
            spinfo = random.choice(self.eligible_edges_dict[e_key][2])
            edge = self.eligible_edges_dict[e_key]
            return { e_key: (edge[0], edge[1], spinfo) }

        # Sample index
        e_key = random.choices( eligible_keys, weights=weights, k=1 )[0]
        spinfo = random.choice( self.eligible_edges_dict[e_key][2] )
        edge = self.eligible_edges_dict[ e_key ]
        return { e_key: (edge[0], edge[1], spinfo) }

    def recompute(self) -> None:
        """
        Recomputes the junction tree for its modgraph, when its tree nodes or cliques and edges became invalid,
        e.g. when a ring is broken into a linear structure. Modifications happen in-place so it does not return anything.
        """
        mapped_smiles_with_ids = replace_numbers_in_string(self.modGraph.smilesWithIds, self.internal_to_external_id)
        new_jt: 'JunctionTree' = JunctionTree.from_smiles(mapped_smiles_with_ids, 
                                                          kekulize=self.kekulize,
                                                          compute_2D_smiles=self.compute_2D_smiles,
                                                          recompute_modGraph_dueto_extids=False,
                                                          use_potential_model=self.use_potential_model,
                                                          border_depth_limit=self.border_depth_limit,
                                                          max_atom_cutoff=self.max_atom_cutoff,
                                                          for_eligible_edges_include_wildstars=self.for_eligible_edges_include_wildstars)
        self.copy_from(new_jt)
        self.set_is_wildstar_flags()
        self.recomputation_split_ids.append(self.curr_split_id - 1)
        self.selected_edges = set()
        if self.selected_edge_before_recompute is not None:
            self.selected_edges.add(self.selected_edge_before_recompute)
            self.selected_edge_before_recompute = None

    def set_is_wildstar_flags(self) -> None:
        """
        A tree vertex (molecule) is wildstar iff every non-hydrogen atom
        has label exactly X_VAR. (Hydrogens are ignored.)
        """
        for vertex in self.tree_vertices:
            # Collect non-H labels in this molecule
            non_h_labels = []
            for atom in vertex.modGraph.vertices:
                lbl = atom.stringLabel
                if lbl == 'H':
                    continue
                non_h_labels.append(lbl)
            # there should always be at least one non-H label in each 'TreeNode' of a JunctionTree
            assert len(non_h_labels) > 0
            vertex.is_wildstar = all(lbl == X_VAR for lbl in non_h_labels)

    def copy_from(self, junction_tree: 'JunctionTree') -> None:
        """
        Copies properties from passed in junction tree to current instance's properties.
        Except certain properties for which we want to maintain their value, e.g. junction_tree_split_off
        i.e. all junction trees that have previously been split off or the mod rules obtained so far.
        
        Args:
            junction_tree (JunctionTree): new junction tree to copy from.
        """
        self.modGraph = junction_tree.modGraph
        self.tree_vertices = junction_tree.tree_vertices
        self.cliques, self.edges = junction_tree.cliques, junction_tree.edges
        self.original_atom_map_numbers = junction_tree.original_atom_map_numbers
        self.internal_to_external_id = junction_tree.internal_to_external_id
        self.external_to_internal_id = junction_tree.external_to_internal_id
        self.mol = junction_tree.mol
        self.nxgraph = junction_tree.nxgraph
        self.selected_edges = set()
        self.left_right_intersection_atom_id_sets = junction_tree.left_right_intersection_atom_id_sets
        self.eligible_edges_dict = junction_tree.eligible_edges_dict
        self.substructure_list: List[str] = junction_tree.substructure_list
        self.substructure_occurances: Dict[str, int] = junction_tree.substructure_occurances
        self.substructure_sizes: List[float] = junction_tree.substructure_sizes
        # GNN features
        self.edge_gnn_feats = junction_tree.edge_gnn_feats
        self.edge_gnn_ekey_split_info_tuples = junction_tree.edge_gnn_ekey_split_info_tuples

    def initiate_random_grammar_rule_induction(self, verbose: bool=False, only_do_once:bool=True, random_mode: bool=True,
                                               edge_weights:Optional[List[float]]=None) -> List['ModRule'] | ModRule:
        """
        Initiates (yet random, later ML informed) splitting process
        until there is no more rule to generate, which is the case if one of these is true:

        - self.edges is empty
        - the is are only singleton tree vertices left, which cannot be split any further

        Returns:
            Copy of set of rules (ModRule) generated during the splitting process
        """
        self.original_edges: Dict[int, Tuple[int, int]] = { k:v for k,v in self.edges.items() }
        split_num: int = 1
        edge_sel: Optional[Dict[int, Tuple[int, int]]] = None

        while self.eligible_edges_dict:
            edge_sel = self.random_edge(edge_weights, only_do_once)
            if edge_sel is None:
                break

            # Perform the split deterministically for the selected edge
            successful = self.initiate_deterministic_grammar_rule_induction([ edge_sel ], verbose, split_num, random_mode=random_mode)
            split_num += 1
            
            if only_do_once and successful:
                return self.modRules[-1]
            
            if not successful:
                break

        if only_do_once:
            return None

        chosen_edge_idx_ = None
        if edge_sel is not None:
            chosen_edge_idx_ = 0

        # Add initial rule: initial vertex X -> remaining substructure (for completeness)
        initial_rule: 'ModRule' = self.create_initial_rule(self.modGraph, chosen_edge_idx=chosen_edge_idx_)
        self.modRules.append(initial_rule)
        self.uniqueModRules = ModRule.remove_redundant_modrules(self.modRules)
        return self.modRules

    def initiate_deterministic_grammar_rule_induction(
        self,
        edges_to_split: List[Dict[int, Tuple[int, int]]],
        verbose: bool=False,
        split_num: Optional[int]=None,
        random_mode: bool=True,
        with_initial_rule: bool=False,
        remove_redundant_rules: bool=False,
    ) -> bool:
        """
        Same as random induction, but follows a deterministic, caller-specified edge order.
        Returns True if really a split occurred, False otherwise.
        """
        successful = False

        if split_num is None:
            split_num = 1

        for edge in edges_to_split:
            if verbose:
                print(f"Split {split_num}: {edge}")
            right_jt, do_recompute = self.split_edge(edge, random_mode=random_mode, verbose=verbose)
            if right_jt is not None:
                successful = True
                self.junction_trees_split_off.append(right_jt)
            if do_recompute:
                self.recompute()
                if verbose:
                    print("RECOMPUTATION")
            split_num += 1

        if with_initial_rule:
            initial_rule: 'ModRule' = self.create_initial_rule(self.modGraph)
            self.modRules.append(initial_rule)

        if remove_redundant_rules:
            self.uniqueModRules = ModRule.remove_redundant_modrules(self.modRules)
            
        return successful

    def split_edge(
        self,
        singleEdgeDict: Dict[int, Tuple[int, int]],
        random_mode: bool=True,
        verbose: bool=False,
    ) -> Tuple[Optional["JunctionTree"], bool]:
        """
        Splits the junction tree by removing a specified edge.
        
        This method removes the connection between the two vertices specified in the
        `edge` tuple. The original tree (`self`) is modified in-place. The portion
        of the tree connected to the second vertex in the `edge` tuple is detached
        and returned as a new `JunctionTree` object.

        The order of vertices in the `edge` tuple dictates which part is kept
        and which part is returned, only if edge_switch_logic is set to `False`, otherwise left tree will always be bigger part:
          - If edge = (u, v), the original tree keeps the component containing vertex `u`,
            and the component containing vertex `v` is returned.
          - If edge = (v, u), the original tree keeps the component containing vertex `v`,
            and the component containing vertex `u` is returned.

        If no edge exists between the specified vertices, the original tree is
        unmodified, and the method returns None.
        
        It also creates class "ModRule(mod.Rule)" that represents the reaction of splitting of
        the right junction tree part from the left junction tree (self instance)
        and adds the rule to the modRules list.

        Args:
            - singleEdgeDict (Dict[int, Tuple[int, int]]): A dict with a single tuple (u, v) containing the indices of the two vertices
                  defining the edge where the split should occur.
            - verbose (bool, Default: True): whether to log steps to console or not
            
        Returns:
            A tuple where first item is
               a new junction tree object representing the subtree detached from
            the second vertex specified in `edge`, or None if the edge does not exist. Junction tree will be null
            in case we do not perform split but instead trigger recomputation and perform new split.
            The type of the returned object is the same as the class instance (`self`).
               and second item is a bool indicating whether left_junction_tree
               so junction tree we keep splitting should be recomputed because it became
               invalid, e.g. because a ring was broken into a linear structure or need revalidation.
        """
        previousModGraph: mod.Graph = self.modGraph
        edges_dict = self.edges
        cliques_dict = self.cliques
        int2ext = self.internal_to_external_id

        # # rings before split
        # num_rings_original: int = len(nx.cycle_basis(self.nxgraph))
        key, edge_metadata = next(iter(singleEdgeDict.items()))
        leid, reid, sele_split_info = edge_metadata
        original_split_info: int = sele_split_info
        # IMPORTANT: to ensure consistent ordering and therefore removal of correct substructure based on rl
        leid, reid = min(leid, reid), max(leid, reid)
        
        assert sele_split_info in [ SplitInfo.JT_LEFT_INTERSECTION_LEFT,
                                    SplitInfo.JT_LEFT_INTERSECTION_RIGHT,
                                    SplitInfo.JT_RIGHT_INTERSECTION_LEFT,
                                    SplitInfo.JT_RIGHT_INTERSECTION_RIGHT ], "Invalid SplitInfo value."
        
        if sele_split_info in [ SplitInfo.JT_RIGHT_INTERSECTION_LEFT, SplitInfo.JT_RIGHT_INTERSECTION_RIGHT ]:
            # swap because if we always continue with left side, so if right is selected we swap
            # so we really continue with the right side :-)
            leid, reid = reid, leid
            if sele_split_info == SplitInfo.JT_RIGHT_INTERSECTION_LEFT:
                sele_split_info = SplitInfo.JT_RIGHT_INTERSECTION_RIGHT
            elif sele_split_info == SplitInfo.JT_RIGHT_INTERSECTION_RIGHT:
                sele_split_info = SplitInfo.JT_RIGHT_INTERSECTION_LEFT
        
        if key not in edges_dict:
            if not edges_dict:
                raise Exception("The set of edges is empty, so junction tree cannot be split any further.")
            raise Exception(f"The edge {(leid, reid)} does not exist in the set of edges.")

        # Left and right vertices along the chosen edge
        left_vertex: TreeNode = self.get_tree_vertex_by_cid(leid)
        right_vertex: TreeNode = self.get_tree_vertex_by_cid(reid)

        # Detach neighbors across the split edge; record presplit neighbors
        lcid = left_vertex.clique_id
        rcid = right_vertex.clique_id
        left_vertex.neighbors = [v for v in left_vertex.neighbors if v.clique_id != rcid]
        left_vertex.presplit_neighbors.append(right_vertex)
        right_vertex.neighbors = [v for v in right_vertex.neighbors if v.clique_id != lcid]
        right_vertex.presplit_neighbors.append(left_vertex)

        # Connected sets (used to compute intersection below)
        left_vertices = left_vertex.get_connected()
        right_vertices = right_vertex.get_connected()
        
        # Optimization: Intersection of components is intersection of the edge cliques.
        intersection: Set[int] = set(left_vertex.clique).intersection(right_vertex.clique)

        left_vertex_is_cycle = left_vertex.has_cycle()
        right_vertex_is_cycle = right_vertex.has_cycle()

        # Helper for leaf collapse
        def collapse_if_leaf(curr_vertex: TreeNode, clique_minus_inter: Set[int]) -> Tuple[TreeNode, Optional[TreeNode]]:
            if len(curr_vertex.neighbors) == 1:
                neighbor_vertex = curr_vertex.neighbors[0]
                if clique_minus_inter.issubset(set(neighbor_vertex.clique)):
                    neighbor_vertex.neighbors.remove(curr_vertex)
                    if curr_vertex.clique_id in cliques_dict:
                        del cliques_dict[curr_vertex.clique_id]
                    wild = TreeNode.create_wildstar_treenode(
                        curr_vertex.clique_id, curr_vertex.clique, curr_vertex.modGraph, intersection, curr_vertex.internal_to_external_id,
                    )
                    return neighbor_vertex, wild
            return curr_vertex, None

        is_left: Optional[bool] = sele_split_info in [ SplitInfo.JT_LEFT_INTERSECTION_LEFT, SplitInfo.JT_RIGHT_INTERSECTION_LEFT ]
        wild_star_treenode: Optional[TreeNode] = None

        singleton_clique_id: Optional[int] = None
        if is_left and len(left_vertex.clique) == 1:
            singleton_clique_id = left_vertex.clique_id
        elif not is_left and len(right_vertex.clique) == 1:
            singleton_clique_id = right_vertex.clique_id
            
        if singleton_clique_id is not None:
            target_node: TreeNode = self.get_tree_vertex_by_cid(singleton_clique_id)

            anchor_ids: Set[int] = set(intersection)
            wild_node = TreeNode.create_wildstar_treenode(
                singleton_clique_id,
                list(target_node.clique),
                target_node.modGraph,
                anchor_ids,
                target_node.internal_to_external_id,
            )

            wild_node.neighbors = []
            wild_node.presplit_neighbors = list(target_node.presplit_neighbors)

            for neighbor in target_node.neighbors:
                neighbor.neighbors = [wild_node if n is target_node else n for n in neighbor.neighbors]
                wild_node.neighbors.append(neighbor)

            for idx, node in enumerate(self.tree_vertices):
                if node is target_node:
                    self.tree_vertices[idx] = wild_node
                    break

            if target_node in left_vertices:
                left_vertices.remove(target_node)
                left_vertices.add(wild_node)
            if target_node in right_vertices:
                right_vertices.remove(target_node)
                right_vertices.add(wild_node)

            if target_node is left_vertex:
                left_vertex = wild_node
            if target_node is right_vertex:
                right_vertex = wild_node

            target_node = wild_node
            wild_star_treenode = wild_node
        

        # REPLACES BELOW COMMENTED CODE
        if sele_split_info in [ SplitInfo.JT_LEFT_INTERSECTION_LEFT, SplitInfo.JT_RIGHT_INTERSECTION_LEFT ]:
            left_external_ids: Set[int] = {y for x in left_vertices for y in x.clique}
            left_minus_inter = left_external_ids.difference(intersection)
            left_vertex, wild_star_treenode = collapse_if_leaf(left_vertex, left_minus_inter)
        elif sele_split_info in [ SplitInfo.JT_LEFT_INTERSECTION_RIGHT, SplitInfo.JT_RIGHT_INTERSECTION_RIGHT ]:
            right_external_ids: Set[int] = {y for x in right_vertices for y in x.clique}
            right_minus_inter = right_external_ids.difference(intersection)
            right_vertex, wild_star_treenode = collapse_if_leaf(right_vertex, right_minus_inter)

        # Sanitize nodes and edges after selection
        reattach_on_left, nodes_remove_fragment = self.sanitize_tree_nodes_remove_fragments_update_cliques(
            is_left, list(intersection), left_vertices, right_vertices, left_vertex, right_vertex, singleton_clique_id
        )

        if wild_star_treenode is not None and is_left is not None:
            self.cliques[wild_star_treenode.clique_id] = list(wild_star_treenode.clique)
            if not is_left:
                left_vertex.neighbors.append(wild_star_treenode) 
                wild_star_treenode.neighbors.append(left_vertex)
                left_vertex = wild_star_treenode
        elif not is_left:
            anchor_ids: Set[int] = intersection if is_left else (set(right_vertex.clique) - intersection)
            wild_star_treenode = TreeNode.create_wildstar_treenode(
                right_vertex.clique_id, right_vertex.clique, right_vertex.modGraph, anchor_ids, right_vertex.internal_to_external_id,
            )
            left_vertex.neighbors.append(wild_star_treenode)
            wild_star_treenode.neighbors.append(left_vertex)

        # Recompute connected components on updated vertices
        left_vertices = left_vertex.get_connected()
        right_vertices = right_vertex.get_connected()
        left_clique_ids = [n.clique_id for n in left_vertices]
        right_clique_ids = [n.clique_id for n in right_vertices]
        left_ids_set = set(left_clique_ids)
        right_ids_set = set(right_clique_ids)

        left_cliques = {cid: cliques_dict[cid] for cid in left_ids_set if cid in cliques_dict}
        right_cliques = {cid: cliques_dict[cid] for cid in right_ids_set if cid in cliques_dict}

        left_edges = {}
        right_edges = {}
        for eid, (u, v) in edges_dict.items():
            if u in left_ids_set and v in left_ids_set:
                left_edges[eid] = (u, v)
            elif u in right_ids_set and v in right_ids_set:
                right_edges[eid] = (u, v)

        singleton_ext_ids: Set[int] = set()
        singleton_in_left = False
        singleton_in_right = False
        if singleton_clique_id is not None:
            if wild_star_treenode is not None and wild_star_treenode.clique_id == singleton_clique_id:
                singleton_ext_ids = set(wild_star_treenode.clique)
            elif singleton_clique_id in cliques_dict:
                singleton_ext_ids = set(cliques_dict[singleton_clique_id])

            if singleton_ext_ids:
                singleton_in_left = any(node.clique_id == singleton_clique_id for node in left_vertices)
                singleton_in_right = any(node.clique_id == singleton_clique_id for node in right_vertices)

        # Build vertex set for anchors on left side
        if is_left:
            inter_set_tmp = set(intersection)
            vertex_ids_flattened = [c for cluster in left_cliques.values() if set(cluster).isdisjoint(inter_set_tmp) for c in cluster]
        else:
            vertex_ids_flattened = [c for cluster in left_cliques.values() for c in cluster]
        vertex_id_set = set(vertex_ids_flattened)

        base_anchor_ids: Set[int] = set(intersection)
        if not is_left:
            neighbor_ids: Set[int] = set()
            for id_ in intersection:
                v = previousModGraph.getVertexFromExternalId(id_)
                if v.isNull():
                    v = find_vertex_by_id(previousModGraph, id_)
                for e in v.incidentEdges:
                    t = e.target
                    if (t.stringLabel != WILD_STAR and t.stringLabel != X_VAR) and t.atomId.symbol != "H":
                        neighbor_ids.add(int2ext[t.id])
            if wild_star_treenode:
                base_anchor_ids = neighbor_ids - (vertex_id_set - set(wild_star_treenode.clique))
            else:
                base_anchor_ids = neighbor_ids - vertex_id_set

        anchor_vertex_ids: Set[int] = set(base_anchor_ids)
        if singleton_ext_ids and singleton_in_left:
            anchor_vertex_ids.update(singleton_ext_ids)

        intersection_minus_anchors: Set[int] = set(intersection) - anchor_vertex_ids
        if intersection:
            for node in nodes_remove_fragment:
                node.clique = [e for e in node.clique if e not in intersection_minus_anchors]
                if node.clique_id in cliques_dict:
                    cliques_dict[node.clique_id] = [ce for ce in cliques_dict[node.clique_id] if ce not in intersection_minus_anchors]
                if node.clique_id in left_cliques:
                    node.remove_fragment(intersection_minus_anchors, node.clique, anchor_vertex_ids)
                if node.clique_id in right_cliques:
                    right_cliques[node.clique_id] = list(node.clique)
                    node.remove_fragment(intersection_minus_anchors, node.clique, anchor_vertex_ids)

        # Components must be exactly two and no intersection between left/right after sanitization
        nx_left = nx.Graph()
        nx_left.add_edges_from(left_edges.values())
        nx_left.add_nodes_from([k for k, v in left_cliques.items() if v])
        num_components_left = nx.number_connected_components(nx_left)

        nx_right = nx.Graph()
        nx_right.add_edges_from(right_edges.values())
        nx_right.add_nodes_from([k for k, v in right_cliques.items() if v])
        num_components_right = nx.number_connected_components(nx_right)

        num_components = num_components_left + num_components_right

        left_atoms = [ce for lst in left_cliques.values() for ce in lst]
        right_atoms = [ce for lst in right_cliques.values() for ce in lst]
        left_right_intersection: Set[int] = set(left_atoms).intersection(right_atoms)
        left_right_intersection -= anchor_vertex_ids
        if wild_star_treenode is not None:
            left_right_intersection -= set(wild_star_treenode.clique)
        has_intersection_ = bool(left_right_intersection)

        if num_components != 2 or has_intersection_:
            if verbose:
                print(
                    f"After sanitization junction tree consisted of {num_components} components, only 2 components are allowed, "
                    f"and left and right cliques had {'a' if has_intersection_ else 'no'} intersection!"
                )
            return None, True

        break_bonds_after_ring_split = (left_vertex_is_cycle and right_vertex_is_cycle and bool(is_left))

        anchors_for_left_graph: Set[int] = set(anchor_vertex_ids) if anchor_vertex_ids else set()
        anchors_for_right_graph: Set[int] = set()

        if singleton_ext_ids:
            if singleton_in_left:
                anchors_for_left_graph.update(singleton_ext_ids)
            if singleton_in_right:
                anchors_for_right_graph.update(singleton_ext_ids)

        right_mod_graph = create_modsubgraph(
            previousModGraph,
            right_cliques,
            int2ext,
            None if is_left else intersection_minus_anchors,
            anchors_for_right_graph,
            False,
            None,
        )
        left_mod_graph = create_modsubgraph(
            previousModGraph,
            left_cliques,
            int2ext,
            intersection_minus_anchors if is_left else None,
            anchors_for_left_graph,
            break_bonds_after_ring_split,
            intersection,
        )

        # Verify non-H atoms accounting (allow wildstars to be removed)
        non_h_left: Set[int] = set(get_internal_to_external_id_map(left_mod_graph, False).values())
        non_h_right_iter = get_internal_to_external_id_map(right_mod_graph, False).values()
        non_h_prev: Set[int] = set(get_internal_to_external_id_map(previousModGraph, False).values())
        lr_union: Set[int] = non_h_left.union(non_h_right_iter)
        assert len(lr_union) == len(non_h_prev) - get_num_wildstars(previousModGraph, lr_union, int2ext)

        prev_int2ext = int2ext

        # MOD RULE creation from detached right substructure (plus its relevant neighbors)
        right_cliques_vertex_ids: Set[int] = {e for c in right_cliques.values() for e in c}
        right_neighbor_vertex_ids: Set[int] = set()
        for rcvi in right_cliques_vertex_ids:
            v_ = previousModGraph.getVertexFromExternalId(rcvi)
            if v_.isNull():
                v_ = find_vertex_by_id(previousModGraph, rcvi)
            for e in v_.incidentEdges:
                tgt = e.target
                if (tgt.stringLabel == WILD_STAR or tgt.stringLabel == X_VAR) or tgt.atomId.symbol != "H":
                    mapped_ext = prev_int2ext.get(tgt.id)
                    if mapped_ext is not None:
                        right_neighbor_vertex_ids.add(mapped_ext)
        subgraph_set: Set[int] = right_cliques_vertex_ids.union(right_neighbor_vertex_ids)

        chosen_edge_index: int = key if random_mode else self.edge_gnn_ekey_split_info_tuples.index( (key, original_split_info) )
        substructure_smiles: str = None if random_mode else self.substructure_list[chosen_edge_index]
        created_rules: List[ModRule] = JunctionTree.extract_rule_given_subgraph(
            right_mod_graph, previousModGraph, self.curr_split_id, subgraph_set, prev_int2ext, chosen_edge_index,
            substructure_smiles=substructure_smiles
        )
        self.modRules.extend(created_rules)


        rule_to_apply: mod.Rule = created_rules[0].modRule
        new_left_mod_graph, rule_application_count = self.apply_rule_until_done(previousModGraph, rule_to_apply, 0)
        self.rule_application_and_counts.append((rule_to_apply, rule_application_count))
        left_mod_graph = new_left_mod_graph

        # In-place update of left JT
        self.modGraph = left_mod_graph
        self.internal_to_external_id = get_internal_to_external_id_map(self.modGraph)
        self.external_to_internal_id = {ext: intl for intl, ext in self.internal_to_external_id.items()}
        mapped_smiles_with_ids = replace_numbers_in_string(self.modGraph.smilesWithIds, self.internal_to_external_id)
        self.mol = Chem.MolFromSmiles(mapped_smiles_with_ids.replace(X_VAR, WILD_STAR), sanitize=False)
        self.nxgraph = get_nxgraph_from_mod_graph(self.modGraph)
        self.tree_vertices = left_vertices
        self.cliques, self.edges = left_cliques, left_edges
        self.reset_atom_map_numbers()
        self.update_vertex_is_leaf()

        # Right JT construction
        right_jt = JunctionTree(
            modGraph=right_mod_graph, vertices=right_vertices, cliques=right_cliques, edges=right_edges
        )
        right_jt.update_vertex_is_leaf()
        right_jt.reset_atom_map_numbers()

        # Check need for recomputation
        # num_rings_left = len(nx.cycle_basis(self.nxgraph))
        # num_rings_right = len(nx.cycle_basis(get_nxgraph_from_mod_graph(right_mod_graph)))
        # single_clique_present_less_than_three_neighbors = any(len(n.clique) == 1 and len(n.neighbors) < 3 for n in left_vertices)
        do_recompute = (
            True 
            # or (num_rings_left + num_rings_right) != num_rings_original)
            # or reattach_on_left
            # or single_clique_present_less_than_three_neighbors
        )

        if is_left or len(self.cliques[leid]) == 1 or len(self.cliques[reid]) == 1:
            sele_edge: Tuple[int, int] = (leid, reid) if leid <= reid else (reid, leid)
            if do_recompute:
                self.selected_edge_before_recompute = sele_edge
            else:
                self.selected_edges.add(sele_edge)

        return right_jt, do_recompute

    def apply_rule_and_recompute(
        self,
        rule: mod.Rule,
        verbosity: int = 0,
        recompute: bool = True,
        max_rule_application_execute_strat: int=10,
    ) -> int:
        """
        Apply a mod.Rule repeatedly (max bounded internally) to this JunctionTree's
        underlying modGraph. If the rule matches at least once the internal modGraph
        is replaced by the final product and (optionally) the junction tree structure
        is recomputed (keeping previously collected ModRules lists intact).

        Parameters
        ----------
        rule : mod.Rule
            The rewrite rule to apply repeatedly.
        verbosity : int
            Verbosity forwarded to apply_rule_until_done.
        recompute : bool
            If True rebuild the junction tree (structure + features) after the change.

        Returns
        -------
        int
            Number of successful applications of the rule (0 => no change).
        """
        new_graph, count = JunctionTree.apply_rule_until_done(self.modGraph, rule, verbosity, 
                                                              max_rule_application_execute_strat=max_rule_application_execute_strat)
        if count <= 0 or new_graph is None:
            return 0

        # Replace current molecule graph
        # self.modGraph = self.replace_xs_by_wildstar(new_graph)
        self.modGraph = new_graph
        if recompute:
            self.recompute() 

        return count

    @staticmethod
    def apply_rule_until_done(mol: mod.Graph, rule: mod.Rule, verbosity: int=0, max_rule_application_execute_strat: int=10) -> Tuple[Optional[mod.Graph], int]:
        """
        Applies the rule to the molecule as many times as possible using the repeat pattern,
        returning:
          - the final molecule,
          - the number of rule applications (depth).
        """
        def _keep_first_only(_g: mod.Graph, _gs: mod.DGStrat.GraphState, first: bool) -> bool:
            return first

        ls = mod.LabelSettings(mod.LabelType.Term, mod.LabelRelation.Specialisation)
        dg = mod.DG(graphDatabase=[mol], labelSettings=ls)

        strat = mod.addSubset([mol]) >> mod.repeat[max_rule_application_execute_strat](rule >> mod.filterSubset(_keep_first_only))

        with dg.build() as b:
            res = b.execute(strat, verbosity=verbosity)

        final = res.subset[0] if res.subset else None

        # Count rule applications by walking the DG
        v_start = dg.findVertex(mol)
        v_final = dg.findVertex(final)
        count = 0
        v = v_final
        # Iterate backwards while we can find an in-edge with the same rule
        while (v != v_start) and (not v.isNull()):
            advanced = False
            for e in v.inEdges:
                if any(r == rule for r in e.rules):
                    srcs = list(e.sources)
                    if srcs:
                        v = srcs[0]
                        count += 1
                        advanced = True
                        break
            if not advanced:
                break
        return final, count

    def visualize_molecule_and_junction_trees(self, size=(500, 500), fig_size=(7, 7)) -> None:
        """Convenience method, not further important."""
        self.set_atom_map_numbers()
        self.visualize(size=size, addAtomIndices=False)
        self.display_graph(fig_size=fig_size)
        self.reset_atom_map_numbers()

    def sanitize_tree_nodes_remove_fragments_update_cliques(
        self,
        is_left: bool,
        intersection: List[int],
        left_vertices: List['TreeNode'],
        right_vertices: List['TreeNode'],
        left_vertex: 'TreeNode',
        right_vertex: 'TreeNode',
        allowed_singleton_clique_id: Optional[int]=None,
    ) -> Tuple[bool, List['TreeNode']]:
        """
        In-place sanitization of tree node connections.
        Ensures that if tree nodes that are not directly connected to intersection atom in junction tree,
        that they are properly reconnected to the correct tree nodes after the split.

        Args:
            - is_left (bool): Indicates whether intersection atoms will be included
                in left or right part of junction tree and accordingly we know for which 
                part of the junction tree we need to perform sanitization work.
            - intersection (List[int]): list of clique elements that are part of intersection,
                i.e. the border that is to be broken/split in the graph.
            - left_vertices (List[TreeNode]): tree nodes of left part after split
            - right_vertices (List[TreeNode]): tree nodes of right part after split
            - left_vertex (TreeNode): anchor node of left part of split
            - right_vertex (TreeNode): anchor node of right part of split
            
        Returns:
            - True if reattachment on left tree occurred (necessitates recomputation), False otherwise
            - list of tree nodes where remove_fragment should be called (after anchor node ids were found)
        """
        edges_dict = self.edges
        cliques_dict = self.cliques

        inter_set = set(intersection)
        vertices, other_vertices = (left_vertices, right_vertices) if is_left else (right_vertices, left_vertices)
        vertex_id, _ = (left_vertex.clique_id, right_vertex.clique_id) if is_left else (right_vertex.clique_id, left_vertex.clique_id)

        # Work on a sorted view of vertices (by decreasing clique size)
        vertices = sorted(list(vertices), key=lambda x: len(x.clique), reverse=True)

        # Initialize node flags
        for v in vertices:
            v.is_reattached = False
            v.do_remove = False
            v.has_neighbor_without_intersection = False

        # Precompute clique sets for faster intersections
        all_nodes = vertices + list(other_vertices)
        clique_sets = {n.clique_id: set(n.clique) for n in all_nodes}

        # Precompute edges incident to cliques and keyed by undirected pairs
        clique_to_eids: Dict[int, List[int]] = defaultdict(list)
        pair_to_eids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for eid, (u, v) in edges_dict.items():
            clique_to_eids[u].append(eid)
            clique_to_eids[v].append(eid)
            key = (u, v) if u <= v else (v, u)
            pair_to_eids[key].append(eid)

        nodes_remove_fragments: List['TreeNode'] = []
        reattachment_on_left_tree_occurred = False
        delete_eids: List[int] = []
        add_edges: List[Tuple[int, int]] = []

        for node in vertices:
            node_cid = node.clique_id
            node_set = clique_sets[node_cid]

            if node_cid == allowed_singleton_clique_id:
                continue

            # Only process nodes that intersect with the split border
            if node_set.isdisjoint(inter_set):
                continue

            # Compute new clique after removing intersection atoms
            new_clique = [e for e in node.clique if e not in inter_set]
            if not new_clique:
                # Delete clique and mark all incident edges for deletion
                if node_cid in cliques_dict:
                    del cliques_dict[node_cid]
                delete_eids.extend(clique_to_eids.get(node_cid, []))
                continue

            new_set = set(new_clique)

            # Identify neighbors that no longer share intersection with this node
            neighbors_wo_intersection = [nei for nei in node.neighbors if nei.clique_id != allowed_singleton_clique_id and new_set.isdisjoint(clique_sets[nei.clique_id])]
            if neighbors_wo_intersection:
                # Detach neighbors that lost intersection
                for nei in neighbors_wo_intersection:
                    nei_cid = nei.clique_id
                    nei.neighbors = [_n for _n in nei.neighbors if _n.clique_id != node_cid]
                    key = (node_cid, nei_cid) if node_cid <= nei_cid else (nei_cid, node_cid)
                    delete_eids.extend(pair_to_eids.get(key, []))
                    nei.has_neighbor_without_intersection = True

                neigh_ids = {n_.clique_id for n_ in neighbors_wo_intersection}
                node.neighbors = [_n for _n in node.neighbors if _n.clique_id not in neigh_ids]

                # If already reattached, just enqueue for fragment removal if needed
                if node.is_reattached:
                    if node.do_remove:
                        nodes_remove_fragments.append(node)
                    continue

                # Build ignore list to avoid self and current neighbors
                ignore_list = {node_cid, *(e.clique_id for e in node.neighbors)}

                # Preferred candidates from current side
                cand = [
                    n for n in vertices
                    if (n.clique_id not in ignore_list) and (not new_set.isdisjoint(clique_sets[n.clique_id]))
                ]
                selected_candidate = min(cand, key=lambda n: len(n.clique)) if cand else None
                do_remove = True
                from_vertices = True

                # Fallback to other side if nothing found (and node is not the anchor)
                if selected_candidate is None and node_cid != vertex_id:
                    do_remove = False
                    from_vertices = False
                    cand = [
                        n for n in other_vertices
                        if (n.clique_id not in ignore_list) and (not node_set.isdisjoint(clique_sets[n.clique_id]))
                    ]
                    selected_candidate = min(cand, key=lambda n: len(n.clique)) if cand else None
                else:
                    nodes_remove_fragments.append(node)

                # Attach if a candidate exists
                if selected_candidate is not None:
                    selected_candidate.neighbors.append(node)
                    node.neighbors.append(selected_candidate)
                    add_edges.append((node_cid, selected_candidate.clique_id))
                    selected_candidate.is_reattached = True
                    selected_candidate.do_remove = do_remove
                    if (from_vertices == is_left):
                        reattachment_on_left_tree_occurred = True

            elif node.has_neighbor_without_intersection:
                # Node itself still fine, but some neighbor had lost intersection; reattach to other side if possible
                ignore_list = {node_cid, *(e.clique_id for e in node.neighbors)}
                cand = [
                    n for n in other_vertices
                    if (n.clique_id not in ignore_list) and (not node_set.isdisjoint(clique_sets[n.clique_id]))
                ]
                selected_candidate = min(cand, key=lambda n: len(n.clique)) if cand else None
                if selected_candidate is not None:
                    selected_candidate.neighbors.append(node)
                    node.neighbors.append(selected_candidate)
                    add_edges.append((node_cid, selected_candidate.clique_id))
            else:
                # Node keeps fragment to remove later
                nodes_remove_fragments.append(node)

        # Perform deletions of edges (deduplicated)
        for eid in set(delete_eids):
            if eid in edges_dict:
                del edges_dict[eid]

        # Add new edges with fresh incremental ids (deduplicated)
        if add_edges:
            next_eid = max(edges_dict.keys(), default=0)
            for u, v in set(add_edges):
                next_eid += 1
                edges_dict[next_eid] = (u, v)

        return reattachment_on_left_tree_occurred, nodes_remove_fragments

    def has_cycle(self) -> bool:
        """Returns `True` if underlying graph has cycle, `False` otherwise."""
        # Avoid exception-driven control flow
        return bool(nx.cycle_basis(self.nxgraph))

    @staticmethod
    def extract_rule_given_subgraph(
        subgraph: mod.Graph,
        modGraph: mod.Graph,
        current_split_id: int,
        subgraph_set: Set[int],
        internal_to_external_id_map: Dict[int, int],
        chosen_edge_idx: int,
        all_monomorphisms: bool = False,
        substructure_smiles: Optional[str] = None
    ) -> List[ModRule]:
        """
        Constructs a ModRule on the current molecule of the junction tree.
        
        Args:
            subgraph (mod.Graph): subgraph / substructure contained in junction tree molecule
                to construct rule from
            modGraph (mod.Graph): super graph the subgraph is a subset of
            current_split_id (int): id of the current split, equals number of split this will be
            subgraph_set (Set[int]): the subgraph set includes all ids of subgraph and of neighbors (necessary because there is a bug in enumerateMonomorphisms)
            all_monomorphisms (bool): whether to create rules for all occurrences, or only the specific subgraph for which the external ids match
        Returns:
            The ModRule.
        """
        ls = mod.LabelSettings(mod.LabelType.Term, mod.LabelRelation.Specialisation)

        def callback(res: List[ModRule], m) -> bool:
            """
            Callback for every monomorphism. Creates the rule and adds it to the rule list.
            """
            continueMonomorphisms: bool = True
            vs = set()
            es = []

            domain: mod.Graph = m.domain

            # Collect mapped vertices in codomain
            for vDom in domain.vertices:
                vCodom = m[vDom]
                vs.add(vCodom)

            matched_ext_ids: Set[int] = set()
            for v in vs:
                # Skip hydrogens to mirror includeHs=False behaviour
                if v is not None and v.stringLabel == 'H':
                    continue
                ext_id = internal_to_external_id_map.get(v.id)
                if ext_id is not None:
                    matched_ext_ids.add(ext_id)

            neigh_ids = [
                internal_to_external_id_map[e.target.id]
                for v in vs
                for e in v.incidentEdges
                if ((e.target.stringLabel == WILD_STAR or e.target.stringLabel == X_VAR) or (e.target.atomId.symbol != 'H'))
                and (e.target.id in internal_to_external_id_map)
            ]
            # domain_set: Set[int] = matched_ext_ids.union(neigh_ids)

            # Only accept the concrete substructure (unless all monomorphisms requested)
            # if (not all_monomorphisms) and (domain_set != subgraph_set):
            #     return continueMonomorphisms

            # Edges internal to vs (in codomain)
            for eDom in domain.edges:
                vSrcCodom = m[eDom.source]
                vTarCodom = m[eDom.target]
                # find actual edge in codomain
                eCodom = next(e for e in vSrcCodom.incidentEdges if e.target == vTarCodom)
                es.append(eCodom)

            # Perimeter (neighbors outside vs)
            nh = set()
            phs = set()
            phes = []
            for v in vs:
                for e in v.incidentEdges:
                    if e.target not in vs:
                        nh.add(e.target)
                        es.append(e)  # include boundary edge on LHS
                        phs.add(v)
                        phes.append(e)

            # Build GML fragments via joins
            vStr = "\t" + "\n\t".join(f'node [ id {v.id} label "{v.stringLabel}" ]' for v in vs)
            nhStr = ("\n\t" + "\n\t".join(f'node [ id {v.id} label "*" ]' for v in nh)) if nh else ""
            eStr = "\t" + "\n\t".join(
                f'edge [ source {e.source.id} target {e.target.id} label "{e.stringLabel}" ]' for e in es
            )

            # Right side: placeholders 'X' for removed interiors, keep neighbors and their edges
            phStr = "\t" + "\n\t".join(f'node [ id {v.id + v.graph.numVertices} label "X" ]' for v in phs) if phs else ""
            pheStr = "\t" + "\n\t".join(
                f'edge [ source {e.source.id + e.graph.numVertices} target {e.target.id} label "{e.stringLabel}" ]'
                for e in phes
            ) if phes else ""

            parts = [
                "rule [",
                'labelType "term"',
                "left [",
                vStr,
                nhStr,
                "",
                eStr,
                "]",
                "right [",
                phStr,
                nhStr,
                "",
                pheStr,
                "]",
                "]",
            ]
            rStr = "\n".join(p for p in parts if p != "")

            r: mod.Rule = mod.Rule.fromGMLString(rStr)
            modR: ModRule = ModRule(current_split_id, r, chosen_edge_index=chosen_edge_idx, substructure_smiles=substructure_smiles)
            res.append(modR)
            # Returning False stops enumeration unless caller explicitly requests all matches
            return False

        rules: List[ModRule] = []
        subgraph.enumerateMonomorphisms(modGraph, callback=lambda m, res=rules: callback(res, m), labelSettings=ls)

        # enforce expectations
        if all_monomorphisms:
            assert len(rules) > 0
        else:
            assert len(rules) == 1

        return rules

    def print_modrules(self, map_to_external_ids: bool = False) -> None:
        rule: ModRule
        for rule in sorted(self.modRules, key=lambda x: x.split_id):
            gml_string: str = rule.modRule.getGMLString()
            if map_to_external_ids:
                gml_string = replace_numbers_in_string(
                    gml_string, number_map=self.internal_to_external_id, replace_pattern="\\d+"
                )
            print(gml_string)
            print("######################################################")

    def create_initial_rule(self, graph: mod.Graph, chosen_edge_idx: Optional[int]) -> ModRule:
        """Generates a concise GML string for creating the input graph from empty."""
        # Faster string construction
        esc = str.replace
        vertices_gml = [
            f'node [ id {v.id} label "{esc(v.stringLabel, "\"", "\\\"")}" ]' for v in getattr(graph, 'vertices', [])
        ]
        edges_gml = [
            f'edge [ source {e.source.id} target {e.target.id} label "{esc(e.stringLabel, "\"", "\\\"")}" ]'
            for e in getattr(graph, 'edges', [])
        ]

        vertices_str = "\n\t\t".join(vertices_gml)
        edges_str = "\n\t\t".join(edges_gml)
        separator = "\n" if vertices_str and edges_str else ""

        parts = [
            "rule [",
            '    labelType "term"',
            "    left [",
            "    ]",
            "    right [",
            f"        {vertices_str}{separator}{edges_str}",
            "    ]",
            "]",
        ]
        gml_rule_string = "\n".join(parts)
        modRule: mod.Rule = mod.Rule.fromGMLString(gml_rule_string.strip())
        return ModRule(self.curr_split_id, modRule, chosen_edge_index=chosen_edge_idx)

    def write_rules_to_files(self, folder_path: str, unique: bool = True) -> None:
        """Writes all rules to files under given folder path."""
        os.makedirs(folder_path, exist_ok=True)
        rulesToWrite: List[ModRule] = self.uniqueModRules if unique else self.modRules
        for mr in rulesToWrite:
            r: mod.Rule = mr.modRule
            full_path: str = os.path.join(folder_path, f"rule_{mr.split_id}.gml")
            with open(full_path, 'w') as ofile:
                ofile.write(r.getGMLString().strip())

    # === RDKit dependent section ===

    def visualize(
        self,
        addAtomIndices: bool = True,
        fixedFontSize: float = 16,
        annotationFontScale: float = 0.5,
        size: Tuple[int, int] = (300, 300),
        removeHs: bool = True,
    ) -> None:
        """Visualizes the original molecule"""
        display_mol(self.mol, addAtomIndices, fixedFontSize, annotationFontScale, size=size, removeHs=removeHs)

    def reset_atom_map_numbers(self) -> None:
        """Resets the atom map number for each atom of the molecule."""
        self.original_atom_map_numbers = {}
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            self.original_atom_map_numbers[idx] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(0)

    def set_atom_map_numbers(self) -> None:
        """Sets the atom map number to their original values again."""
        if self.original_atom_map_numbers is None:
            return
        for idx, atom_map_num in self.original_atom_map_numbers.items():
            self.mol.GetAtomWithIdx(idx).SetAtomMapNum(atom_map_num)

    def reset_vertices_atommap_nums(self) -> None:
        for vertex in self.tree_vertices:
            vertex.reset_atom_map_numbers()

    def set_vertices_atommap_nums(self) -> None:
        for vertex in self.tree_vertices:
            vertex.set_atom_map_numbers()