# Item 5 — candidate-recall improvement

Protocol: `docs/candidate_recall_protocol.md` (written before running). Weights and codebook use the training subset only; numbers below are on the disjoint evaluation subset (N = 500, 124,750 exact pairs).

Learned weights: {'sem': 0.5, 'concept': 0.0, 'mech': 1.0, 'ctx': 0.0} (training recall@5 at k = 10: 0.498).

| index | k | pairs | pair_fraction | recall@5 | seconds |
|---|---|---|---|---|---|
| original + topic blocking | 2 | 2021 | 0.016 | 0.228 | 0.243 |
| original | 2 | 2471 | 0.020 | 0.244 | 0.336 |
| combined, learned weights | 2 | 1410 | 0.011 | 0.190 | 0.192 |
| combined, uniform weights | 2 | 1439 | 0.012 | 0.137 | 0.192 |
| original + topic blocking | 5 | 3541 | 0.028 | 0.394 | 0.278 |
| original | 5 | 4647 | 0.037 | 0.420 | 0.260 |
| combined, learned weights | 5 | 2222 | 0.018 | 0.329 | 0.317 |
| combined, uniform weights | 5 | 2355 | 0.019 | 0.255 | 0.191 |
| original + topic blocking | 10 | 5931 | 0.048 | 0.560 | 0.285 |
| original | 10 | 8493 | 0.068 | 0.605 | 0.274 |
| combined, learned weights | 10 | 3571 | 0.029 | 0.486 | 0.193 |
| combined, uniform weights | 10 | 3836 | 0.031 | 0.395 | 0.196 |
| original + topic blocking | 20 | 9963 | 0.080 | 0.705 | 0.297 |
| original | 20 | 15121 | 0.121 | 0.763 | 0.258 |
| combined, learned weights | 20 | 6284 | 0.050 | 0.654 | 0.200 |
| combined, uniform weights | 20 | 6686 | 0.054 | 0.572 | 0.201 |