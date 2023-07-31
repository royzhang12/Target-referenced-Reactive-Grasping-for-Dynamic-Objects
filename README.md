# Target-Referenced-Reactive-Grasping-for-Dynamic-Objects

This repo contains the implementation of the paper:  
"Target Referenced Reactive Grasping for Dynamic Objects"


## Requirements
- Python 3.7
- PyTorch 1.7
- Open3d 
- TensorBoard
- NumPy
- SciPy
- Pillow
- tqdm
- einops
- Pytorch3d

## Installation

Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## License
All data, labels, code and models belong to the graspnet team, MVIG, SJTU and are freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an email at fhaoshu at gmail_dot_com and cc lucewu at sjtu.edu.cn .
