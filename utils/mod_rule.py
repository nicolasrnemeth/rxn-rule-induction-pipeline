# --------------------------------------------------------------------
# Original code Copyright (c) 2025 Nicolas Nemeth licensed under Attribution-NonCommercial-ShareAlike 4.0 International
# --------------------------------------------------------------------

import os
import re

import mod
from typing import List, Optional, Tuple
from rdkit import Chem
from IPython.display import display, SVG

# Constants
WILD_STAR: str = "*"
X_VAR: str = "X"


class ModRule(object):
    """
    ModRule with additional functionalities.
    """
    LABEL_SETTINGS: mod.LabelSettings = mod.LabelSettings(mod.LabelType.Term, mod.LabelRelation.Isomorphism)
    
    def __init__(self, split_id: int, mod_rule: mod.Rule, chosen_edge_index: Optional[int]=None, substructure_smiles: Optional[str]=None) -> 'ModRule':
        self.split_id: int = split_id # indicates which split, the rule was created by
        self.modRule: mod.Rule = mod_rule
        self.chosen_edge_index: Optional[int] = chosen_edge_index # indicates what selected edge led to this rule
        self.substructure_smiles: Optional[str] = substructure_smiles # the SMILES string of the substructure this rule represents
        
    def areFullyIsomorphic(self, other: 'ModRule') -> bool:
        """
        Checks whether this rule and the given rule are fully isomorphic.
        This means that their LHS, context graphs, and RHS are all pairwise
        isomorphic with respect to node and edge structure and labels.

        Args:
            - other (mod.Rule): the rule to compare with.
        Returns:
            bool: True if the left, context, and right graphs of both rules are isomorphic; False otherwise.
        """
        numIsomorphisms: int = self.modRule.isomorphism(other.modRule, maxNumMatches=1, labelSettings=ModRule.LABEL_SETTINGS)
        return bool(numIsomorphisms)
    
    @staticmethod
    def remove_redundant_modrules(modrules: List['ModRule']) -> List['ModRule']:
        """
        Removes redundant `ModRule`s from the input list.
        A ModRule is considered redundant if it is fully isomorphic to an earlier `ModRule` in the list.

        Args:
            modrules (List[ModRule]): List of ModRule objects to filter.
        Returns:
            List[ModRule]: A new list containing only unique (non-redundant) `ModRule`s
        """
        unique_rules: list[ModRule] = []
        for r in modrules:
            if not any(r.areFullyIsomorphic(existing) for existing in unique_rules):
                unique_rules.append(r)
        return unique_rules
    
    def write_to_file(self, file_path: str):
        """Writes the rule as GML string to file under given path."""
        with open(file_path, 'w') as ofile:
            ruleGmlString: str = self.modRule.getGMLString()
            ofile.write(ruleGmlString.strip())
    
    def visualize(self, size: Tuple[int, int]=(300, 300), removeHs: bool=True):
        """Visualizes component of rule that is (de)/attached including anchor nodes."""
        smiles_str: str = ModRule.extract_subcomponent_from_rule(self.modRule, False)
        ModRule.display_smiles(smiles_str, addAtomIndices=False, size=size, removeHs=removeHs)
       
    @staticmethod
    def load_rules_from_folder(folder_path: str) -> List['ModRule']:
        """Loads all rules from folder and parses them into ModRule object"""
        rules: List[ModRule] = []
        if not os.path.exists(folder_path):
            return rules
        for fname in os.listdir(folder_path):
            with open(f"{folder_path}/{fname}", 'r') as ifile:
                ruleGmlString: str = ifile.read().strip()
            rule: mod.Rule = mod.Rule.fromGMLString(ruleGmlString)
            split_id: int = int(re.findall(r'\d+', fname)[0])
            mrule: ModRule = ModRule(split_id, rule)
            rules.append(mrule)
        return rules
    
    @staticmethod
    def extract_subcomponent_from_rule(rule: mod.Rule, asGML: bool=False) -> Optional[str]:
        """
        Extracts the subcomponent that is part of the rule as smiles string,
        otherwise returns `None` if neither left or right represent an 1 component.
        
        Args:
            rule (mod.Rule): the rule obj
            asGML (bool, default: False): whether to return GML string or smiles string
        Returns:
            Smiles string if `asGML` is `True` otherwise Smiles string
        """
        nodes, edges = ModRule.combine_graph_with_context(rule, True)
        nodeStrings, edgeStrings = ModRule.stringify_nodes_edges(nodes, edges)
        gmlString = f"graph [\n\t{ "\n\t".join(nodeStrings) }\n\t{ "\n\t".join(edgeStrings) }\n]"
        
        try:
            mod_graph: mod.Graph = mod.Graph.fromGMLString(gmlString, add=False, printStereoWarnings=False)
            if mod_graph is None:
                raise Exception("Graph cannot be null.")
            # if 'X' in mod_graph.smiles:
            #     raise Exception("Graph cannot be visualized if it contains 'X' as label.")
            if asGML:
                return gmlString
            return mod_graph.smiles
        except Exception as e:
            print(f"Error occurred: {e}")
            
        nodes, edges = ModRule.combine_graph_with_context(rule, False)
        nodeStrings, edgeStrings = ModRule.stringify_nodes_edges(nodes, edges)
        gmlString = f"graph [\n\t{ "\n\t".join(nodeStrings) }\n\t{ "\n\t".join(edgeStrings) }\n]"
        
        try:
            mod_graph: mod.Graph = mod.Graph.fromGMLString(gmlString, add=False, printStereoWarnings=False)
            if mod_graph is None:
                raise Exception("Graph cannot be null.")
            if 'X' in mod_graph.smiles:
                raise Exception("Graph cannot be visualized if it contains 'X' as label.")
            if asGML:
                return gmlString
            return mod_graph.smiles
        except Exception as e:
            print(f"Error occurred: {e}")
        ###
        return None
    
    @staticmethod
    def combine_graph_with_context(rule: mod.Rule, left: bool) -> Tuple[List[mod.Graph.Vertex], List[mod.Graph.Edge]]:
        """
        Combines left or right and context.
        
        Args:
            rule (mod.Rule): 
            left (bool): if `True` then left combined with context, otherwise right with context  
        Returns:
            Tuple where first item is list of vertices of left/right combined with context
            and second item is list of edges of left/right component.
        """
        leftrightgraph = rule.left if left else rule.right
        nodes: List[mod.Graph.Vertex] = list(filter(lambda v:"stringLabel" in dir(v), sorted(list(leftrightgraph.vertices) + list(rule.context.vertices), key=lambda x:x.id)))
        edges: List[mod.Graph.Edge] = list(filter(lambda e:"stringLabel" in dir(e), sorted(list(leftrightgraph.edges), key=lambda x:x.source.id)))
        return nodes, edges
    
    @staticmethod
    def stringify_nodes_edges(nodes: List[mod.Graph.Vertex], edges: List[mod.Graph.Edge]) -> Tuple[List[str], List[str]]:
        """
        Turns nodes and edges into their GML string representation to be used to construct a `mod.Graph` object from it.
        
        Args:
            nodes (List[mod.Graph.Vertex]): list of vertices
            edges (List[mod.Graph.Edge]): list of edges
        Returns:
            Tuple of lists of GML string representations of nodes and edges
        """
        nodeStrings: List[str] = [ f"node [ id { n.id } label \"{ n.stringLabel if n.stringLabel != X_VAR else WILD_STAR }\" ]" for n in nodes ]
        edgeStrings: List[str] = [ f"edge [ source { e.source.id } target { e.target.id } label \"{ e.stringLabel }\" ]" for e in edges ]
        return nodeStrings, edgeStrings
    
    
    ## RdKit dependent
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