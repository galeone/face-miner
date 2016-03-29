Face Miner
==========
Data mining applied to face detection

# How Face Miner works

*Do you want to see a presentation that's a summary of the work? Checkout the project website: <https://galeone.github.io/face-miner/>*

Face Miner is based on the paper of Wen-Kwang Tsao [[1]]. To understand Face Miner, you have to read it before. You can find a copy of the paper into the `docs/` folder.

The aim of Face Miner is to build a classifier capable of detecting faces in images.

To reach this aim, the paper's authors built three different classifiers, using a simple-to-complex and coarse-to-fine approach.

In Face Miner, a critical discussion of the paper has been done. Therefore Face Miner __is not__ a reproduction of the work described in the paper, but a different implementation resulting from this critical discussion.

The original paper was not reproducible due to the usage of private datasets and neither the resulting work was publicly avaible.

Face Miner is completely reproducible and the resulting work is avaiable in this repo.

The main differences between the original paper and Face Miner are:
- The datasets used to train and test the classifiers
- Thus the resulting Maximal Frequent Patterns mined from the MAFIA algorithm
- The features extracted from the train dataset, to train the Support Vecotr Machine.
- The way to reduce the dimensionality of theese features (the authors used a k-d-tree, and I'm still asking myself why and how. They're not searching anything but they're using the k-d-tree to project featuers along different axis (???)), in Face Miner the PCA has been used.
- The preprocessing thresholds used in the binarization step (and also this is due to the usage of different datasets)
- The single response criterion: in Face Miner we stop asap without collecting a set of regions and scoring every extracted region.
- The performance: are the performance described in the paper real?

# Build and Install
Load the project into Qt Creator. Build the subproject MAFIA before and than build Face Miner.

# Theory
The theory behind face miner, and thus the critical discussion can be found in the `docs/pdf` folder.
At the moment the complete theorical discussion is in __Italian only__.

# TODO
- Make face miner a library, separated from the example implementation that uses QT.
- Improve README.md (translate PDF documentation in English)

# Results
The `results/` folder contains some example of the results of Face Miner

# Performance and comparision with the Viola & Jones algorithm
The performance of Face Miner are __bad__ in term of speed, especially if compared to the speed achieved by the state of art algorithm used for the face detection task: the Viola & Jones algorithm.

The other main difference, is the region detected. In Face Miner the ROI is smaller wrt the ROI detected with Viola & Jones. This can be __good__ in applications of facial recognition, where you have to reduce noisy parts to focus on the face only.

Below you can see the different ROI detected with Face Miner and Viola & Jones.

![Comparison](https://media.nerdz.eu/8ltnyr7GKCPI.png)

The benchmark has been done on the Yale Face Database B [[2]]

# License
Face Miner is licensed under Mozilla Public License version 2.0

# References
1. A data mining approach to face Detection
[1]: http://www.sciencedirect.com/science/article/pii/S0031320309003434 "A data mining approach to face detection"
2. Yale Face Database B
[2]: http://vision.ucsd.edu/content/yale-face-database "P. Belhumeur, J. Hespanha, D. Kriegman, Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection, IEEE Transactions on Pattern Analysis and Machine Intelligence, July 1997, pp. 711-720."
