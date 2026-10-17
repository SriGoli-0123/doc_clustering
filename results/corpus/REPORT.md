# Corpus experiment (PubMed + arXiv)

9986 abstracts, 20 subject labels (used only for evaluation).

## LDA K sweep

| K | C_v | NPMI | diversity | perplexity |
|---|---|---|---|---|
| 10 | 0.643 | 0.106 | 0.832 | 1745.593 |
| 20 | 0.665 | 0.113 | 0.742 | 1527.987 |
| 30 | 0.666 | 0.114 | 0.740 | 1430.135 |
| 50 | 0.640 | 0.105 | 0.713 | 1347.277 |

K* = 30; gensim variational LDA at K*: C_v=0.490, NPMI=-0.048

## Clustering vs subject labels

| method | NMI | ARI | purity | n_clusters | silhouette |
|---|---|---|---|---|---|
| K-Means SBERT (k=20) | 0.831 | 0.758 | 0.851 | 20 | 0.150 |
| K-Means SPECTER (k=20) | 0.786 | 0.687 | 0.819 | 20 | 0.163 |
| K-Means SPECTER (k=30) | 0.772 | 0.669 | 0.860 | 30 | 0.171 |
| HDBSCAN SBERT | 0.709 | 0.231 | 0.489 | 10 | 0.200 |
| LDA Gibbs argmax θ (K=30) | 0.664 | 0.571 | 0.713 | 30 | 0.091 |
| LDA Gibbs argmax θ (K=20, closest to #labels) | 0.643 | 0.507 | 0.618 | 20 | 0.095 |
| LDA variational argmax θ (K=30) | 0.424 | 0.206 | 0.372 | 30 | -0.079 |
| HDBSCAN SPECTER | 0.362 | 0.098 | 0.100 | 2 | 0.296 |
| BERTopic-style (HDBSCAN + c-TF-IDF) | 0.361 | 0.098 | 0.100 | 2 | 0.296 |

## Summaries (ROUGE / BERTScore vs held-out abstracts)

LDA clusters:

| system | rouge1 | rouge2 | rougeL | bertscore_f1 | clusters |
|---|---|---|---|---|---|
| TextRank + centroid + MMR | 0.219 | 0.026 | 0.117 | 0.750 | 30 |
| first-sentence baseline | 0.148 | 0.018 | 0.095 | 0.724 | 30 |
| random-sentence baseline | 0.197 | 0.022 | 0.111 | 0.739 | 30 |

Embedding clusters:

| system | rouge1 | rouge2 | rougeL | bertscore_f1 | clusters |
|---|---|---|---|---|---|
| TextRank + centroid + MMR | 0.226 | 0.032 | 0.120 | 0.755 | 30 |
| first-sentence baseline | 0.143 | 0.021 | 0.092 | 0.725 | 30 |
| random-sentence baseline | 0.187 | 0.021 | 0.105 | 0.737 | 30 |

## t-SNE faithfulness

| space | perplexity | dims | trustworthiness | continuity | knn_preservation | n | k |
|---|---|---|---|---|---|---|---|
| SPECTER | 15 | 2 | 0.982 | 0.970 | 0.421 | 3000 | 10 |
| SPECTER | 30 | 2 | 0.983 | 0.972 | 0.431 | 3000 | 10 |
| SPECTER | 50 | 2 | 0.984 | 0.973 | 0.429 | 3000 | 10 |
| LDA θ | 15 | 2 | 0.993 | 0.980 | 0.503 | 3000 | 10 |
| LDA θ | 30 | 2 | 0.993 | 0.980 | 0.514 | 3000 | 10 |
| LDA θ | 50 | 2 | 0.994 | 0.981 | 0.512 | 3000 | 10 |
| SPECTER | 50 | 3 | 0.989 | 0.976 | 0.500 | 3000 | 10 |

## Discovery on the stratified subset

| docs | candidate_pairs | all_pairs | conflict_pairs | analogy_pairs | explore_pairs | predicted | abstention_rate | families | blocked_by_guards | mean_ms_per_pair |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 24037 | 1999000 | 9121 | 11941 | 3030 | {'incomparable': 15674, 'agreement': 2251, 'analogy': 4898, 'abstain': 747, 'heterogeneity': 466, 'contradiction': 1} | 0.031 | 490 | {'analogy': 5572, 'agreement': 179, 'heterogeneity': 134, 'contradiction': 10} | 3.994 |

## Relation summaries (top generalised relations per topic)

- **topic 0** (design, algorithm, optimization, search, adaptive): matching raw trajectories --enables--> rugged objective that sparse surrogates cannot learn [1]; annealed --decreases--> -confidence-bound [1]; removing sequential lcb --increases--> six-scene mean [1] — pair counts {'incomparable': 284, 'analogy': 32, 'abstain': 13, 'agreement': 26}
- **topic 1** (induce, gene, response, pathway, protein): metformin --enables--> apoptosis [2]; metformin --decreases--> aging-induced genes [2]; study investigates the --associated--> vitamin d supplementation [1] — pair counts {'incomparable': 34, 'analogy': 6, 'agreement': 3, 'abstain': 2}
- **topic 2** (cell, growth, cellular, type, tissue): exhibiting a stronger --associated--> on pca [1]; dha --associated--> stromal components [1]; cancer-associated fibroblast differentiation --enables--> microenvironment favorable to pca initiation [1] — pair counts {'analogy': 34, 'incomparable': 244, 'agreement': 42, 'abstain': 13}
- **topic 3** (brain, neural, temporal, neuron, memory): coffee consumption --increases--> auditory neuropathy [1]; auditory neuropathy --enables--> by diabetes [1]; coffee consumption --decreases--> hearing threshold shift of the auditory evoked potential [1] — pair counts {'incomparable': 209, 'analogy': 21, 'abstain': 9, 'agreement': 16, 'heterogeneity': 2}
- **topic 4** (dynamic, phase, noise, dynamical, stability): noise --enables--> transitions [2]; spatially structured network noise fields --enables--> partially overlapping subnetworks [1]; mismatch --decreases--> capacity [1] — pair counts {'incomparable': 474, 'analogy': 41, 'abstain': 14, 'agreement': 19}
- **topic 5** (population, level, age, estimate, individual): ssb consumption --enables--> difficulty [1]; how different aspects of social relationships --associated--> mortality [1]; how different aspects of social relationships --associated--> all-cause mortality [1] — pair counts {'incomparable': 2}
- **topic 6** (economic, market, price, policy, country): study seeks to understand --enables--> mortality [1]; leading --enables--> of death were vehicle accidents [1]; all-cause mortality --no_effect--> annual fluctuations [1] — pair counts {'incomparable': 264, 'analogy': 28, 'agreement': 33, 'abstain': 8, 'heterogeneity': 4}
- **topic 7** (metformin, cancer, drug, treatment, survival): metformin --decreases--> all-cause mortality [4]; metformin --associated--> of death [4]; metformin --associated--> cancer incidence [3] — pair counts {'incomparable': 304, 'analogy': 40, 'abstain': 39, 'agreement': 158, 'heterogeneity': 55}
- **topic 8** (information, source, knowledge, field, physical): menopausal hormone therapy --no_effect--> conversion to either [1]; omega-3 fatty acids --increases--> cardiovascular outcomes [1]; kalshimultivariate-event architecture --enables--> market objects [1] — pair counts {'analogy': 10, 'incomparable': 19, 'agreement': 2}
- **topic 9** (control, condition, constraint, parameter, trajectory): penalty --decreases--> concentrations [1]; penalty --associated--> relative prominence of centers [1]; distinct converged states --enables--> multi-start sweeps on three synthetic two-dimensional landscapes [1] — pair counts {'incomparable': 186, 'analogy': 17, 'agreement': 12, 'abstain': 1}
- **topic 10** (evolution, framework, dynamic, interaction, specie): generation z showed --decreases--> excitement seeking [1]; dispositions --enables--> contemporary cultural conditions [1]; developmental encoding --increases--> phenotypic diversity [1] — pair counts {'incomparable': 43, 'agreement': 4, 'analogy': 11}
- **topic 11** (role, recent, human, research, health): interest --increases--> past decade [1]; papers --increases--> past decade [1]; vitamin d supplementation --associated--> colorectal cancer [1] — pair counts {'agreement': 5, 'analogy': 2, 'incomparable': 7}