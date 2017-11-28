Gemello: Creating a Detailed Energy Breakdown from just the Monthly Electricity Bill
-------------------

This repository contains code for Gemello: Creating a Detailed Energy Breakdown from just the Monthly Electricity Bill. This paper was accepted at [SIGKDD 2016](http://www.kdd.org/kdd2016/).

Please use the following bib entry to cite the paper.
```
@inproceedings{Batra:2016:GCD:2939672.2939735,
 author = {Batra, Nipun and Singh, Amarjeet and Whitehouse, Kamin},
 title = {Gemello: Creating a Detailed Energy Breakdown from Just the Monthly Electricity Bill},
 booktitle = {Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 series = {KDD '16},
 year = {2016},
 location = {San Francisco, California, USA},
 pages = {431--440},
  url = {http://doi.acm.org/10.1145/2939672.2939735},
 doi = {10.1145/2939672.2939735}
}
```

More pertinent links to the paper

1. [Paper pdf](https://www.iiitd.edu.in/~nipunb/papers/gemello.pdf)
2. [Youtube video](https://www.youtube.com/watch?v=pzgqd9OhvDA)
3. [Poster](https://www.iiitd.edu.in/~nipunb/slides/kdd_poster_final.pdf)

This work was also presented at the [3rd NILM workshop](http://nilmworkshop.org/2016/). You can find the [slides](http://nilmworkshop.org/2016/slides/NipunBatra2.pdf) and the [talk recording](https://www.youtube.com/watch?v=LUauYdlbH74).




This Readme gives a description of the repository structure and how one can repeat the experiments. The final plots and analysis is all done in IPython notebooks.
Each folder in this repository has a Readme describing the contents of the folder.


First, links to the notebooks for the figures

| Figure/Table| Link |
| --- | --- |
| Figure 1 | [Approach](https://docs.google.com/drawings/d/1R68GnSezUbC-RiGcwy3E8cSYYHZAgf50YiYkTOqUWFg/edit?usp=sharing) |
| Figure 2 | [Dataset description](https://github.com/nipunbatra/Gemello/blob/master/code/dataset_description.ipynb) |
| Figure 3 |[Code for producing estimates](https://github.com/nipunbatra/Gemello/blob/master/code/main_result_parallel_new.py), [Notebook for ingesting the estimates and producing plots](https://github.com/nipunbatra/Gemello/blob/master/code/main-result.ipynb) |
| Figure 4 | [Comparison with state-of-art at higher frequency](https://github.com/nipunbatra/Gemello/blob/master/code/lbm-2min-15min-vs-gemello.ipynb)|
| Figure 5 and Table 3| [Sensitivity analysis on features](https://github.com/nipunbatra/Gemello/blob/master/code/sensitivity-features.ipynb)|
| Figure 6| [Sensitivity analysis on number of homes](https://github.com/nipunbatra/Gemello/blob/master/code/sensitivity-numhomes.ipynb) |
