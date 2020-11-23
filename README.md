# CS453 Final Project

## Paper

[_Surface Simplification Using Quadric Error Metrics_](https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf)

### Abstract

> Many applications in computer graphics require complex, highly detailed models. However, the level of detail actually necessary may vary considerably. To control processing time, it is often desirable to use approximations in place of excessively detailed models. We have developed a surface simplification algorithm which can rapidly produce high quality approximations of polygonal models. The algorithm uses iterative contractions of vertex pairs to simplify models and maintains surface error approximations using quadric matrices. By contracting arbitrary vertex pairs (not just edges), our algorithm is able to join unconnected regions of models. This can facilitate much better approximations, both visually and with respect to geometric error. In order to allow topological joining, our system also supports non-manifold surface models

[Citation](./assets/citation-2417323.bib)

## Setup

### conda

``` bash
conda create --file requirements.txt --name viz python=3.9 -y
conda activate viz
```

### pip

``` bash
# NOTE try pip3 if that doesn't work on Mac OS Catalina or later.
pip install -r requirements.txt
```

## This repository

### assets/

* **citation-2417323.bib**: BibTEX citation for original paper downloaded from [ResearchGate](https://www.researchgate.net/publication/2417323_Surface_Simplification_Using_Quadric_Error_Metrics/citation/download).
* **quadrics.pdf**: Original CMU paper, for convenience.

### models/

* **cow.obj**: Original 3D cow model from the original paper.

### src/

* **garland_heckbert.py**: Main file for the alorithm implementation.

## Visualize Model

``` bash
python scripts/viz_cow.py
```

## Resources

* Tips for writing mesh code in Python can be found [here](https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/README.md).
