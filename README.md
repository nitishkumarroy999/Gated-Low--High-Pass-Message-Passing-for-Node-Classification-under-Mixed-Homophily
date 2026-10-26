# Gated-Low--High-Pass-Message-Passing-for-Node-Classification-under-Mixed-Homophily
Gated Low-/High-Pass Message Passing for Node Classification under Mixed Homophily
Abstract: Standard graph neural networks (GNNs) such as GCN average features over neighbours, a
low-pass operation that works when linked nodes share labels (homophily) but degrades when
they do not (heterophily). Real graphs are rarely uniformly one or the other: the local ho-
mophily of individual nodes varies within a single graph. We propose GLP-GNN, a message-
passing layer that learns, per node and per layer, a gate interpolating between a low-pass filter
ˆA and a high-pass filter I − ˆA, while keeping a separate ego channel. We show the layer can
preserve high-frequency signal that repeated low-pass smoothing destroys, and we describe a
full evaluation protocol on homophilous and heterophilous benchmarks against GCN, GAT and
GraphSAGE with ablations of each component. This draft reports the protocol and implemen-
tation
