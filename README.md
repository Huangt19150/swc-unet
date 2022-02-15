# SWCUnet

A Skeleton Weighted multi-Channel U-net (SWCUnet) model prototyped for neurite segmentation facilitating automatic neuron morphology reconstruction, based on my work developing a full image processing and reconstruction pipeline for our novel SMART imaging system.

* Chen H, **Huang T**, Yang Y, ... & Guo Z V (2021). Sparse imaging and reconstruction tomography for high-speed high-resolution whole-brain imaging. *Cell Reports Methods*, 1(6), 100089.

<https://doi.org/10.1016/j.crmeth.2021.100089>
No demo data included in this repository. Data availability please contact the Correspondence author.

## Acknowledgement
Baseline model referencing the DeepMACT system:
* Pan C, Schoppe O, Parra-Damas A, ... & Ertürk A (2019). Deep learning reveals cancer metastasis and therapeutic antibody targeting in the entire body. *Cell*, 179(7), 1661-1676.

## Model Design
* Multi-Channel Input
Given the sparsity characteristic of our signal, 3D image volume was projected to the X, Y, Z direction, respectively, to enable an effective training utilizing a 2D U-net architecture, though multiple channel of initial compression was tested, with n = 32 resulting the best outcome. Huge computation resource and time was saved while still achieving good performance.
 
* Skeleton Weighted Loss Function
Single axonal signal usually appears in “dotted”, instead of smoothly continuous, curve due to its biological nature (<1 um thinness in diameter but with sudden swelling at widely distributed bouton sites). To enhance the continuity of segmentation outcome, which is crucial for the following auto-reconstruction, extra weight was added to “skeleton” pixels among all foreground pixels (5-times weight gave the best performance according to experiments).

* Complete Workflow
Complete segmentation-based auto-reconstruction workflow illustrated below. [MOST tracing](https://doi.org/10.1016/j.neuroimage.2013.10.036) is a commonly used auto-reconstruction algorithm.


