# TASA
This repository contains the source code for our paper [**Unsupervised Domain Adaptation for Cross-Patient Seizure Classification**](https://iopscience.iop.org/article/10.1088/1741-2552/ad0859/meta) (JNE, 2023).

## main files
- tasa_sds_gda merges the proposed tasa, sds, and gda methods;
- mldg and maml are two meta-learning baselines;
- mlp is a deep neural network baseline without using any tricks.
## data
Dataset can be obtained in directory "./data/fts_labels/", containing S1-S27.

For more details regarding the original EEG signals from the CHSZ dataset, please contact via email at vivi@hust.edu.cn.
## utils
Some necessary functions are in utils directory.

## Citation
If you find this repo helpful, please cite our work:
```
@article{Wang2023TASA,
  title={Unsupervised domain adaptation for cross-patient seizure classification},
  author={Wang, Ziwei and Zhang, Wen and Li, Siyang and Chen, Xinru and Wu, Dongrui},
  journal={Journal of Neural Engineering},
  volume={20},
  number={6},
  pages={066002},
  year={2023},
}
```
