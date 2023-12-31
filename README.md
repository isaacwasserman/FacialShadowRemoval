# Unsupervised Facial Shadow Removal
Final Project for CIS 581 by Grace Benner, Chenchen Lin, and Isaac Wasserman

## Using this Repo
> 🚧 **Dependencies**
> 
> Note that we have not included a `requirements.txt` file in this repo. Our codebase is made up of two preexisting shadow removal methods and one novel method, each with their own conflicting dependencies. Additionally, in the case of the novel method, not all of the dependencies can be installed from `pip` or `conda`. 
>
> To reproduce our results with each method, refer to the various "Usage" sections listed below.
>
>- [ShadowGP Usage](#Usage)
>- [Blind Shadow Removal Usage](#Usage-1)
>- [Novel Classical Approach Usage](#Usage-2)

## ShadowGP [1]

### Description
ShadowGP, a pioneering unsupervised shadow removal method, distinguishes itself from supervised approaches by leveraging deep generative priors from a pretrained GAN model. The model decomposes a shadowed portrait into a shadow-free image, a full-shadow image, and a shadow mask, reconstructing the image using progressive optimization. Given an input shadow portrait $I$, the decomposition is represented as $I=I_\text{free} \otimes M + I_\text{full} \otimes (1-M)$, where $I_\text{free}$, $I_\text{full}$, and $M$ are the shadow-free, full-shadow, and shadow mask components respectively. Each of these components, as well as a color matrix $C$ which defines differences between $I_\text{free}$ and $I_\text{full}$, are optimized to best reconstruct $I$. Using a pretrained StyleGAN2 to reproduce the original face greatly reduces the facial parameter space to StyleGAN's embedding space. When a loss threshold is reached, we assume that $I_\text{free}$ sufficiently approximates a naturally occuring shadow-free image. ShadowGP outperforms some supervised methods in LPIPS and SSIM metrics, demonstrating robustness to scenarios like watermarks and tattoos. However, it exhibits limitations in smoothing effects on facial features and slight lighting/color discrepancies. Despite these, ShadowGP offers an unsupervised alternative with notable generalization capabilities.

### Usage
The ShadowGP component can be run from a self contained Jupyter Notebook at [ShadowGP/shadowgp_colab.ipynb](ShadowGP/shadowgp_colab.ipynb).

This notebook installs all of its required packages and weights. When run inside of Google Colab, it will automatically import the rest of the repository. 

## Blind Shadow Removal [2]
### Description
BlindShadowRemoval is a supervised model specialized for portrait shadow removal. Different from prior work, Blind Shadow Removal method only
look at gray-scale image for first stage decomposition: $I_{free,gs} = I_{gs} \odot (1 - B) + I_{gs} \odot B \oslash M_I'= I_{gs} \odot (1-B +B\oslash M_I')$ This method avoids turning this problem into a memorization mode. The model consists of two major steps: 1) grayscale shadow removal; 2) colorization. Grayscale shadow removal module predicts the deshadowed face in grayscale. This module consists of an encoder, a stack of residual non-local blocks, and a decoder. Colorization module breaks down into 3 steps: 1) erasing, 2) inpainting, and 3) color space transformation. Overall, BlindShadowRemoval is a novel approach to decompose RGB shadow removal into grayscale shadow removal and colorization, which provides state-of-the-art shadow removal and shadow segmentation results and photo-realistic deshadow quality.

### Usage
The Blind Shadow Removal component can be run from a self contained Jupyter Notebook at [BlindShadowRemoval/BlindShadowRemoval.ipynb](BlindShadowRemoval/BlindShadowRemoval.ipynb).

This notebook installs all of its required packages and weights. When run inside of Google Colab, it will automatically import the rest of the repository. 

## Novel Classical Approach
### Description
We explore an alternative to deep learning techniques for facial image manipulation, opting for a handcrafted approach with intentional design choices. Our model starts by detecting facial landmarks, generating a face mask that allows us to focus on skin regions. To address harsh adjustment boundaries, we introduce a vertex-coloring technique, softening edges of the mask. Using a Gaussian mixture model in LAB space, we identify shadowed and well-lit face regions. After further refining the shadow mask through morphological operations, color adjustments in shadowed areas are made by treating skin color as a function of pixel coordinates, employing linear regression trained on well-lit regions. We enhance the inpainting process for shadowed boundaries using the PatchMatch algorithm [3], significantly improving image fidelity. However, challenges arise in high-frequency areas like eyes or mouth, leading us to apply targeted sharpening to mitigate blurring effects. Our non-deep learning model demonstrates effectiveness in achieving somewhat-realistic facial image manipulations, offering a promising alternative to more complex neural approaches.

### Usage
Running our classical approach requires installation of PyPatchMatch, an implementation of the PatchMatch algorithm from InvokeAI. This can be difficult, and we have not been successful in completing this installation in Google Colab. Therefore, we recommend running this method locally.

Here are instructions for running this method:
1. Install `pypatchmatch`. Full instructions with details for each operating system can be found [here](https://invoke-ai.github.io/InvokeAI/installation/060_INSTALL_PATCHMATCH), but a summary is given below.

    a. Install OpenCV (both the development kit and the Python bindings)

    b. `pip install pypatchmatch`

    c. Confirm installation with:
    ```
    python
    >>> from patchmatch import patch_match
    ```
    which should print logs from building the binaries.

2. From the `/classical` directory, run `pip install -r requirements.txt`

3. Run the cells contained in [classical/classical.ipynb](classical/classical.ipynb).

### Exploratory Files

For the remaining exploratory files that are not a key portion of our research, such as baseline or shadow classifier. The requirements, inputs, and results are self-contained in the respective folders and .ipynb files.

## References
[1] Yingqing He, Yazhou Xing, Tianjia Zhang, and Qifeng Chen. Unsupervised portrait
shadow removal via generative priors. In Proceedings of the 29th ACM International
Conference on Multimedia, pages 236–244, 2021.

[2] Yaojie Liu, Andrew Hou, Xinyu Huang, Liu Ren, and Xiaoming Liu. Blind removal of
facial foreign shadows. 2022.

[3] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. PatchMatch:
A randomized correspondence algorithm for structural image editing. ACM Transactions
on Graphics (Proc. SIGGRAPH), 28(3), Aug. 2009.
