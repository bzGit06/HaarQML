# HaarQML
This repository contains the official Python implementation of [*Holographic deep thermalization
for secure and efficient quantum random state generation*](https://arxiv.org/abs/2411.03587), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Peng Xu](https://francis-hsu.github.io/), [Xiaohui Chen](https://the-xiaohuichen.github.io/), and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2024quantum,
      title={Holographic deep thermalization
for secure and efficient quantum random state generation}, 
      author={Zhang, Bingzhi and Xu, Peng and Chen, Xiaohui and Zhuang, Quntao},
      year={2024},
      eprint={2411.03587},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuits is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package with [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backend. Use of GPU is not required, but highly recommended. 

Additionally, the packages [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) is used for speeding up certain evaluation, and [Qiskit](https://docs.quantum.ibm.com/guides) is needed for experiments on IBM Quantum device.

## File Structure
The file `QTM_haar.ipynb` contains all numerical simulations and codes for plotting figures.
