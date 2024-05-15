# Maximizing Malicious Influence (MaxiMal)
Official implementation of [Maximizing Malicious Influence in Node Injection Attack](https://dl.acm.org/doi/abs/10.1145/3616855.3635790), WSDM2024


Built based on [CLGA](https://github.com/RinneSz/CLGA), [GCA](https://github.com/CRIPAC-DIG/GCA) and [DeepRobust](https://deeprobust.readthedocs.io/en/latest/#).


## Usage
1.To produce perturbed graphs with MaxiMal
```
python MaxiMal.py 
```
2.To train the GCN for node classification with the perturbed graph
```
python evaluation.py 
```
## Citation
If you use our code in your research, please cite the following article:
```
@inproceedings{zhang2024maximizing,
  title={Maximizing Malicious Influence in Node Injection Attack},
  author={Zhang, Xiao and Bao, Peng and Pan, Shirui},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  pages={958--966},
  year={2024}
}
```
