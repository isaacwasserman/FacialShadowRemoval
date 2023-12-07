# Unsupervised Facial Shadow Removal
Final Project for CIS 581 by Grace Benner, Chenchen Lin, and Isaac Wasserman

## Using this Repo
As two of the methods below (ShadowGP and Blind Shadow Removal) are forks of existing repositories, we have included our forks as submodules of this repository. To reproduce our results, refer to the instructions contained in each of the methods below:
- [ShadowGP Usage](#ShadowGP)

## ShadowGP [1]

### Description
ShadowGP, a pioneering unsupervised shadow removal method, distinguishes itself from supervised approaches by leveraging deep generative priors from a pretrained GAN model. The model decomposes a shadowed portrait into a shadow-free image, a full-shadow image, and a shadow mask, reconstructing the image using progressive optimization. Given an input shadow portrait $I$, the decomposition is represented as $I=I_\text{free} \otimes M + I_\text{full} \otimes (1-M)$, where $I_\text{free}$, $I_\text{full}$, and $M$ are the shadow-free, full-shadow, and shadow mask components respectively. Each of these components, as well as a color matrix $C$ which defines differences between $I_\text{free}$ and $I_\text{full}$, are optimized to best reconstruct $I$. Using a pretrained StyleGAN2 to reproduce the original face greatly reduces the facial parameter space to StyleGAN's embedding space. When a loss threshold is reached, we assume that $I_\text{free}$ sufficiently approximates a naturally occuring shadow-free image. ShadowGP outperforms some supervised methods in LPIPS and SSIM metrics, demonstrating robustness to scenarios like watermarks and tattoos. However, it exhibits limitations in smoothing effects on facial features and slight lighting/color discrepancies. Despite these, ShadowGP offers an unsupervised alternative with notable generalization capabilities.

### Usage


## Blind Shadow Removal [2]
### Description
### Usage

## Novel Classical Approach
### Description
We explore an alternative to deep learning techniques for facial image manipulation, opting for a handcrafted approach with intentional design choices. Our model starts by detecting facial landmarks, generating a face mask that allows us to focus on skin regions. To address harsh adjustment boundaries, we introduce a vertex-coloring technique, softening edges of the mask. Using a Gaussian mixture model in LAB space, we identify shadowed and well-lit face regions. After further refining the shadow mask through morphological operations, color adjustments in shadowed areas are made by treating skin color as a function of pixel coordinates, employing linear regression trained on well-lit regions. We enhance the inpainting process for shadowed boundaries using the PatchMatch algorithm [3], significantly improving image fidelity. However, challenges arise in high-frequency areas like eyes or mouth, leading us to apply targeted sharpening to mitigate blurring effects. Our non-deep learning model demonstrates effectiveness in achieving somewhat-realistic facial image manipulations, offering a promising alternative to more complex neural approaches.

### Usage

## References
[1] Yingqing He, Yazhou Xing, Tianjia Zhang, and Qifeng Chen. Unsupervised portrait
shadow removal via generative priors. In Proceedings of the 29th ACM International
Conference on Multimedia, pages 236–244, 2021.

[2] Yaojie Liu, Andrew Hou, Xinyu Huang, Liu Ren, and Xiaoming Liu. Blind removal of
facial foreign shadows. 2022.

[3] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. PatchMatch:
A randomized correspondence algorithm for structural image editing. ACM Transactions
on Graphics (Proc. SIGGRAPH), 28(3), Aug. 2009.