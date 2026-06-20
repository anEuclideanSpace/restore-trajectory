# RestoreTrajectory

RestoreTrajectory is a benchmark for evaluating how well Earth-observation models understand
ecological **restoration trajectories** rather than momentary land cover. It treats the recovery
pathway of a restoration unit — its phase, its trajectory, and weak recovery signals under
distribution shift — as the primary object of evaluation.

The benchmark spans three restoration regimes (post-fire recovery, afforestation, and mining-land
rehabilitation) and pairs optical and SAR recovery trajectories with structure references and
per-unit phase labels. Its central expectation is that models judged on optical greenness alone
systematically overestimate true ecological recovery: they conflate canopy cover with vegetation
structure, and they do not generalize across restoration regimes.
