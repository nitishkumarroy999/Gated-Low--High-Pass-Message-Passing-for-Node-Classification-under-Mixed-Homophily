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

Nitish Kumar Roy (Author 1st only)  https://nitishkumarroy999.github.io/Nitish-Kumar-Roy-CSE/


Related Work
Heterophily-oriented GNNs separate ego and neighbours zhu2020h2gcn, learn signed attention-like

coefficients or frequency responses bo2021fagcn, or reorganise the graph pei2020geomgcn. Over-
smoothing is analysed by oono2020. Our layer differs by gating per node between two fixed spectral

filters rather than learning a global frequency response. A fuller comparison and positioning against
the latest work should be written after a literature search before submission.



References

[Kipf and Welling(2017)] T. Kipf, M. Welling. Semi-supervised classification with graph convolu-
tional networks. ICLR, 2017.

[Veliˇckovi ́c et al.(2018)] P. Veliˇckovi ́c et al. Graph attention networks. ICLR, 2018.
[Hamilton et al.(2017)] W. Hamilton, R. Ying, J. Leskovec. Inductive representation learning on
large graphs. NeurIPS, 2017.
[Zhu et al.(2020)] J. Zhu et al. Beyond homophily in graph neural networks: current limitations
and effective designs. NeurIPS, 2020.
[Pei et al.(2020)] H. Pei et al. Geom-GCN: Geometric graph convolutional networks. ICLR, 2020.
[Oono and Suzuki(2020)] K. Oono, T. Suzuki. Graph neural networks exponentially lose expressive
power for node classification. ICLR, 2020.
[Bo et al.(2021)] D. Bo et al. Beyond low-frequency information in graph convolutional networks.
AAAI, 2021.
[Yang et al.(2016)] Z. Yang, W. Cohen, R. Salakhutdinov. Revisiting semi-supervised learning with
graph embeddings. ICML, 2016.

[Rozemberczki et al.(2021)] B. Rozemberczki, C. Allen, R. Sarkar. Multi-scale attributed node em-
bedding. Journal of Complex Networks, 2021.

4

[Platonov et al.(2023)] O. Platonov et al. A critical look at the evaluation of GNNs under het-
erophily: are we really making progress? ICLR, 2023.

[Hu et al.(2020)] W. Hu et al. Open Graph Benchmark: Datasets for machine learning on graphs.
NeurIPS, 2020.
