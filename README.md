# rxn-rule-induction-pipeline

This repository contains the pipeline and source code developed for the Master's thesis: **Automated Chemical Reaction Rule Induction via Junction-Tree Guided Graph Transformations and Reinforcement Learning**. The goal of this project is to automate the extraction of chemically sensible reaction rules from molecule datasets using graph grammar modelling and reinforcement learning. 

### Description

Molecular structures can be formally represented as labeled graphs. In this formalism, chemical reactions correspond to local graph transformations that must adhere to strict physicochemical constraints, such as valence compliance and atom conservation. While graph grammars represent a powerful generative language for chemical reaction spaces, automated extraction of valid reaction rules remains complex.

This pipeline addresses this challenge by utilizing the graph grammar modelling language (GML) provided by the MØD software package. By combining tree-structured, hierarchical molecule representations—specifically, an adapted Junction-Tree decomposition—with Reinforcement Learning (RL), this framework generates valid reaction rules. These rules can reproduce initial molecular training datasets, generalize to generate novel valid molecules, and optimize against specific evaluation metrics. The decomposition method breaks molecular graphs into chemically sensible substructures, such as rings or functional groups, and arranges them in cycle-free tree structures for improved chemical validity and interpretability.