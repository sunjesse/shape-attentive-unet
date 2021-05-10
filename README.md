Code for our paper "SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation": https://arxiv.org/pdf/2001.07645v3.pdf. Set to appear in MICCAI 2020.

### Running the code
To run the code, you can follow the steps below:
<ol>
<li> Register on https://acdc.creatis.insa-lyon.fr/#challenges and download the ACDC - Segmentation dataset.</li>
<li> Assign the root directory of the dataset to the DATA_ROOT variable at the bottom of train.py. Alternatively, you can fill the flag -data-root to the root directory each time you run the code.</li>
<li> Run train.py using </li> **python3 train.py**.
</ol>
If you find our work helpful, please consider citing our work: 

```
@misc{sun2020saunet,
    title={SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation},
    author={Jesse Sun and Fatemeh Darbehani and Mark Zaidi and Bo Wang},
    year={2020},
    eprint={2001.07645},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
