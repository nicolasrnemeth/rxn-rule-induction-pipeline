# rxn-rule-induction-pipeline

## Automated Chemical Reaction Rule Induction via Junction-Tree Guided Graph Transformations and Reinforcement Learning

[span_0](start_span)This repository contains the pipeline and source code developed for the Master's thesis: **Automated Chemical Reaction Rule Induction via Junction-Tree Guided Graph Transformations and Reinforcement Learning**[span_0](end_span). [span_1](start_span)The goal of this project is to automate the extraction of chemically sensible reaction rules from molecule datasets using graph grammar modelling and reinforcement learning[span_1](end_span). 

### Description

[span_2](start_span)Molecular structures can be formally represented as labeled graphs[span_2](end_span). [span_3](start_span)In this formalism, chemical reactions correspond to local graph transformations that must adhere to strict physicochemical constraints, such as valence compliance and atom conservation[span_3](end_span). [span_4](start_span)While graph grammars represent a powerful generative language for chemical reaction spaces, automated extraction of valid reaction rules remains complex[span_4](end_span).

[span_5](start_span)This pipeline addresses this challenge by utilizing the graph grammar modelling language (GML) provided by the MØD software package[span_5](end_span). [span_6](start_span)By combining tree-structured, hierarchical molecule representations—specifically, an adapted Junction-Tree decomposition—with Reinforcement Learning (RL), this framework generates valid reaction rules[span_6](end_span). [span_7](start_span)These rules can reproduce initial molecular training datasets, generalize to generate novel valid molecules, and optimize against specific evaluation metrics[span_7](end_span). [span_8](start_span)The decomposition method breaks molecular graphs into chemically sensible substructures, such as rings or functional groups, and arranges them in cycle-free tree structures for improved chemical validity and interpretability[span_8](end_span).

### Key Features

* **[span_9](start_span)Junction-Tree Decomposition**: Breaks down molecular graphs into meaningful chemical substructures to avoid chemically invalid flat graph manipulations[span_9](end_span).
* **[span_10](start_span)Reinforcement Learning Optimization**: Employs an AI model to iteratively select and substitute substructures with non-terminal symbols, optimizing the generated rules against specific chemical metrics[span_10](end_span).
* **[span_11](start_span)MØD Integration**: Uses the graph grammar modelling language (GML) from the MØD software package for robust graph transformations[span_11](end_span).
* **[span_12](start_span)Generative Extensibility**: Generated rules can produce novel, chemically valid molecules that extend beyond the initial training datasets[span_12](end_span).