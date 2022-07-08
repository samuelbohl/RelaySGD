# RelaySGD

Implementation of the decentralized learning algorithm RelaySGD[^1] inside of Bagua[^2].

[^1]: https://doi.org/10.48550/arXiv.2110.04175

[^2]: https://github.com/BaguaSys/bagua/tree/master

## Observation so far

### CIFAR10 - VGG11

#### Comparing the decentraliced algorithm in bagua with RelaySGD

![](plots/algo_comparison.png)

#### RelaySGD vs Allreduce

![](plots/algo_comparison_2.png)

#### Comparing different topologies of RelaySGD

![](plots/relay_topologies.png)
