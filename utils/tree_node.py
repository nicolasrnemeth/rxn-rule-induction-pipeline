# --------------------------------------------------------------------
# Copyright Nicolas Nemeth 2025 licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

import re
import uuid
import mod
import numpy as np
import collections
import rdkit.Chem as Chem
from rdkit import RDLogger
import networkx as nx

RDLogger.DisableLog("rdApp.warning")

from typing import List, Optional, Set, Tuple, Dict

from .utils import (
    replace_numbers_in_list,
    replace_numbers_in_string,
)
from .chemutils import (
    get_internal_to_external_id_map,
    get_nxgraph_from_mod_graph,
    display_mol,
)

## === Consts ===
WILD_STAR = '*'
X_VAR = "X"
NUMERIC_TOKEN_RE = re.compile(r"\d+")
H_LABEL_TOKEN = 'label "H"'
##



class TreeNode(object):
    """
    Represents a node in a molecule decomposition tree (junction tree).

    Each node corresponds to a cluster (clique) of atoms in the original molecule.
    It stores the SMILES representation of the cluster, the RDKit molecule object,
    the indices of atoms belonging to this clique, and references to neighboring nodes.
    """

    def __init__(
        self,
        mg: mod.Graph,
        clique_id: int,
        clique: Optional[List[int]],
        presplit_neighbors: List['TreeNode'] = [],
        is_wildstar: bool = False,
    ) -> None:
        """
        Initializes a TreeNode.

        Args:
            mg (mod.Graph): The mod graph representing the molecular fragment (clique).
            clique_id (int): id representing the clique
            clique (List[int]): A list of atom indices (integers) belonging to this clique
                                in the context of the original molecule.
            presplit_neighbors (List[TreeNode]): list of all neighbors that have been split off of this TreeNode instance
            is_wildstar (bool): whether this node's graph uses '*' wildcards for anchors
        """
        # Core graph/state
        self.modGraph: mod.Graph = mg
        self.clique_id: int = clique_id
        self.clique: List[int] = list(clique) if clique is not None else []

        # Graph neighborhood / tree structure
        self.neighbors: List['TreeNode'] = []
        # sanitize tree node action does not take account for presplit_neighbors update
        self.presplit_neighbors: List['TreeNode'] = list(presplit_neighbors)

        # Bookkeeping / metadata
        self.tn_uuid: uuid.UUID = uuid.uuid4()  # unique id of tree node
        self.features: np.array = self.compute_features()
        self.original_atom_map_numbers: Dict[int, int] = {}  # map from RDKit atom idx to original atom map number

        # ID mappings
        self.internal_to_external_id: Dict[int, int] = get_internal_to_external_id_map(self.modGraph)
        # invert mapping: external -> internal
        self.external_to_internal_id: Dict[int, int] = {ext: intl for intl, ext in self.internal_to_external_id.items()}

        # Tree annotations
        self.junction_tree: Optional[object] = None
        self.is_leaf: bool = len(self.neighbors) == 1
        self.is_reattached: Optional[bool] = None
        self.do_remove: Optional[bool] = None
        self.has_neighbor_without_intersection: Optional[bool] = None
        self.is_wildstar: bool = is_wildstar

        # Build RDKit and NetworkX representations
        mapped_smiles_with_ids = replace_numbers_in_string(mg.smilesWithIds, self.internal_to_external_id)
        self.mol: Chem.Mol = Chem.MolFromSmiles(mapped_smiles_with_ids.replace(X_VAR, WILD_STAR), sanitize=False)
        self.nxgraph: nx.Graph = get_nxgraph_from_mod_graph(self.modGraph)

    def set_junction_tree(self, junction_tree):
        """Sets the parent junction tree on the tree node."""
        self.junction_tree = junction_tree

    def compute_features(self) -> np.array:
        """Compute feature vector for the node (placeholder)."""
        return np.array([])

    def has_cycle(self) -> bool:
        """Returns True if underlying graph has cycle, False otherwise."""
        # Avoid exception-driven control flow; use cycle_basis
        return bool(nx.cycle_basis(self.nxgraph))

    @classmethod
    def create_wildstar_treenode(
        cls,
        clique_id: int,
        clique: List[int],
        mod_graph: mod.Graph,
        anchor_ids: Set[int],
        internal_to_external_map: Dict[int, int],
    ) -> 'TreeNode':
        """Create a Wildstar tree node with anchors labeled '*' in a fresh mod.Graph."""
        int2ext = internal_to_external_map
        anchors_contains = anchor_ids.__contains__

        # Build GML nodes (external IDs)
        vertices_gml: List[str] = []
        vappend = vertices_gml.append
        for v in mod_graph.vertices:
            ext_id = int2ext[v.id]
            # If the external id is an anchor, label with '*', else keep original string label
            label = WILD_STAR if anchors_contains(ext_id) else v.stringLabel
            vappend(f'  node [ id {ext_id} label "{label}" ]')

        # Build GML edges (external IDs)
        edges_gml: List[str] = []
        eappend = edges_gml.append
        for e in mod_graph.edges:
            source_ext_id = int2ext[e.source.id]
            target_ext_id = int2ext[e.target.id]
            eappend(f'  edge [ source {source_ext_id} target {target_ext_id} label "{e.stringLabel}" ]')

        # Assemble GML
        gml_lines: List[str] = ["graph ["]
        gml_lines.extend(vertices_gml)
        gml_lines.extend(edges_gml)
        gml_lines.append("]")
        new_gml_string = "\n".join(gml_lines)

        new_mod_graph: mod.Graph = mod.Graph.fromGMLString(new_gml_string, add=False, printStereoWarnings=False)
        tree_node: TreeNode = cls(new_mod_graph, clique_id, clique, is_wildstar=True)
        return tree_node

    def remove_fragment(self, vertex_indices: List[int], clique: List[int], anchor_vertex_ids: Set[int]):
        """
        Removes the subgraph induced by the list of vertex_indices (external IDs)
        from the current mod graph, since this subgraph is split off of the current graph.

        Args:
            vertex_indices (List[int]): list of vertex indices to remove (external IDs)
            clique (List[int]): updated clique
            anchor_vertex_ids (Set[int]): ids of anchor vertices (external IDs)
        """
        mg = self.modGraph
        int2ext = self.internal_to_external_id

        # Get input GML lines
        gml_lines = mg.getGMLString().splitlines()

        # Sets for quick membership
        vertex_id_set = set(vertex_indices)
        anchors = set(anchor_vertex_ids)

        # Collect H atoms (by INTERNAL id) to remove: attached to a vertex we're removing (even if vertex is an X_VAR or WILD_STAR) and not an anchor
        h_internal_ids_to_remove: Set[int] = set()
        for e in mg.edges:
            s = e.source
            t = e.target

            s_is_wild = (s.stringLabel == WILD_STAR or s.stringLabel == X_VAR)
            t_is_wild = (t.stringLabel == WILD_STAR or t.stringLabel == X_VAR)

            s_is_H = (not s_is_wild) and (s.atomId.symbol == "H")
            t_is_H = (not t_is_wild) and (t.atomId.symbol == "H")

            if s_is_H and t_is_H:
                raise Exception(
                    "Edge-case: bond between two H-atoms occurred, case not handled yet! see TreeNode.remove_fragment"
                )

            if s_is_H:
                mapped_t_ext = int2ext[t.id]
                if (mapped_t_ext in vertex_id_set) and (mapped_t_ext not in anchors): #and (not t_is_wild): # required because H-atoms can also be on Xs
                    h_internal_ids_to_remove.add(s.id)
            elif t_is_H:
                mapped_s_ext = int2ext[s.id]
                if (mapped_s_ext in vertex_id_set) and (mapped_s_ext not in anchors): #and (not s_is_wild): # required because H-atoms can also be on Xs
                    h_internal_ids_to_remove.add(t.id)

        def any_num_in_set(line: str, ids: Set[int]) -> bool:
            # Fast path: skip regex if line has no digits at all
            # This avoids unnecessary regex invocations on most lines
            for ch in line:
                if ch.isdigit():
                    # At least one digit present: evaluate full test
                    return any((int(n) in ids) for n in NUMERIC_TOKEN_RE.findall(line))
            return False

        # 1) Remove H node lines that correspond to hydrogens to be removed (numbers are INTERNAL at this point)
        #    Narrow the check using the exact label marker to avoid scanning most lines.
        gml_lines = [
            line
            for line in gml_lines
            if (H_LABEL_TOKEN not in line) or (not any_num_in_set(line, h_internal_ids_to_remove))
        ]

        # 2) Replace internal IDs with external IDs everywhere (preserves original logic)
        gml_lines = replace_numbers_in_list(gml_lines, int2ext)

        # 3) Remove all lines that reference any of the vertex external IDs we want to drop
        vertex_id_set_ext = vertex_id_set  # already external
        final_lines = [line for line in gml_lines if not any_num_in_set(line, vertex_id_set_ext)]
        new_gml = "\n".join(final_lines)

        # Update object state (preserve order and logic)
        self.clique = list(clique)
        try:
            self.modGraph = mod.Graph.fromGMLString(new_gml, add=False, printStereoWarnings=False)
        except Exception as e:
            print(e)
        self.internal_to_external_id = get_internal_to_external_id_map(self.modGraph)
        # external -> internal (same inversion as original)
        self.external_to_internal_id = {ext: intl for intl, ext in self.internal_to_external_id.items()}

        mapped_smiles_with_ids = replace_numbers_in_string(self.modGraph.smilesWithIds, self.internal_to_external_id)
        self.mol = Chem.MolFromSmiles(mapped_smiles_with_ids.replace(X_VAR, WILD_STAR), sanitize=False)
        self.nxgraph = get_nxgraph_from_mod_graph(self.modGraph)
        self.reset_atom_map_numbers()

    def get_num_all_vertices(self) -> int:
        """
        Returns:
            number of vertices of whole substructure the vertex is part of
        """
        nodes: Set[TreeNode] = self.get_connected()
        # Avoid building intermediate concatenated list
        return sum(len(n.clique) for n in nodes)

    def get_connected(self) -> Set['TreeNode']:
        """
        Performs a graph traversal (BFS) starting from self
        and returns a set of all unique tree nodes encountered.

        Returns:
            A set containing all unique nodes found.
        """
        visited_nodes: Set[TreeNode] = set()
        queue: collections.deque[TreeNode] = collections.deque()

        visited_nodes.add(self)
        queue.append(self)

        while queue:
            current_node: TreeNode = queue.popleft()
            for neighbor in current_node.neighbors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
        return visited_nodes

    def add_neighbor(self, nei_node: 'TreeNode') -> None:
        """
        Adds a neighboring TreeNode to this node's list of neighbors.

        Args:
            nei_node (TreeNode): The TreeNode instance to add as a neighbor.
        """
        self.neighbors.append(nei_node)

    # === RDKit dependent section ===

    def visualize(
        self,
        addAtomIndices: bool = False,
        fixedFontSize: float = 16,
        annotationFontScale: float = 0.5,
        size: Tuple[int, int] = (300, 300),
        removeHs: bool = True,
    ):
        """Visualizes the chemical substructure the tree node holds."""
        display_mol(self.mol, addAtomIndices, fixedFontSize, annotationFontScale, size, removeHs=removeHs)

    def reset_atom_map_numbers(self) -> None:
        """Resets the atom map number for each atom of the molecule."""
        self.original_atom_map_numbers = {}
        atoms = self.mol.GetAtoms()
        for atom in atoms:
            idx = atom.GetIdx()
            self.original_atom_map_numbers[idx] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(0)

    def set_atom_map_numbers(self):
        """Sets the atom map number to their original values again."""
        if self.original_atom_map_numbers is None:
            return
        for idx, atom_map_num in self.original_atom_map_numbers.items():
            self.mol.GetAtomWithIdx(idx).SetAtomMapNum(atom_map_num)

