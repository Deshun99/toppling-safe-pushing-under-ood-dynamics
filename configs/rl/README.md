# RL object configurations

These files describe cuboid geometry and hidden physical parameters. They do
not contain SAC hyperparameters.

| Config | Role | Contents |
|---|---|---|
| [`training_distributions/uniform_excluding_evaluation_regions.json`](training_distributions/uniform_excluding_evaluation_regions.json) | Default training distribution | Uniform sampling within the specified bounds, rejecting eight joint parameter regions reserved for evaluation. |
| [`training_distributions/uniform_without_exclusions.json`](training_distributions/uniform_without_exclusions.json) | Training distribution for comparison | The same sampling bounds, with no excluded regions. |
| [`evaluation_sets/fixed_cases.json`](evaluation_sets/fixed_cases.json) | Fixed evaluation cases | Explicit object specifications grouped into the historical eight cases and mass, friction, COM-height, and joint extrapolation cases. |
| [`evaluation_sets/sampled_reserved_regions.json`](evaluation_sets/sampled_reserved_regions.json) | Sampled evaluation set definition | Eight sampling regions matching the default training distribution's exclusions. |
| [`evaluation_sets/sampled_mass_extrapolation.json`](evaluation_sets/sampled_mass_extrapolation.json) | Sampled evaluation set definition | Light and heavy object sampling regions beyond the supplied training distributions' mass bounds. |

A training distribution is a rule for drawing objects. An `EvaluationCase` is
one concrete object, together with its group and seed index. An `EvaluationSet`
contains resolved cases and their source configuration. A sampled evaluation
set definition becomes concrete cases after choosing a count and seed.

Training uses `--distribution-config`. Held-out evaluation uses
`--evaluation-config` for explicit cases or `--sampler-config` for sampled sets.
Group labels describe the intended experiment; actual membership in training
support is checked against the evaluated checkpoint's metadata.
