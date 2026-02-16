# Grid-Resilience-GNN

A small portfolio project on power-grid resilience using
cascade simulation + graph learning.

## Approach

- Model the grid as a geometric graph with node-level `load`, `capacity`, and `status`.
- Run a rule-based cascade process: failed neighbors add stress; nodes fail when load exceeds capacity.
- Convert simulated scenarios into graph training samples.
- Train a 2-layer GCN for node-level failure prediction.

## Results

From `artifacts/visualizations/summary_latest.csv`:

- Mean failure rate is about **98%** across attack sizes (1 to 8 initial failures).
- The curve is almost flat (`0.9800 -> 0.9822`), with low variance (`std <= 0.0038`).

This suggests the system is already in a high-fragility regime: once a cascade starts, attack size matters less than structural vulnerability.

## Visuals

![Cascade Snapshot](artifacts/visualizations/snapshot_latest.png)
![Failure Distribution](artifacts/visualizations/distribution_latest.png)
![Intensity Curve](artifacts/visualizations/intensity_curve_latest.png)


