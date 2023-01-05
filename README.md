# Segment-then-segment: Context-Preserving Crop-Based Segmentation for Large Biomedical Images

The code repository for the paper:

Benčević, Marin, Yuming Qiu, Irena Galić, and Aleksandra Pizurica. 2023. "Segment-then-Segment: Context-Preserving Crop-Based Segmentation for Large Biomedical Images" Sensors 23, no. 2: 633. https://doi.org/10.3390/s23020633 https://www.mdpi.com/1424-8220/23/2/633

BibTeX:

```
@Article{bencevic2023,
AUTHOR = {Benčević, Marin and Qiu, Yuming and Galić, Irena and Pizurica, Aleksandra},
TITLE = {Segment-then-Segment: Context-Preserving Crop-Based Segmentation for Large Biomedical Images},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {2},
ARTICLE-NUMBER = {633},
URL = {https://www.mdpi.com/1424-8220/23/2/633},
ISSN = {1424-8220},
DOI = {10.3390/s23020633}
}
```

## Requirements:

 - Python 3.8
 - PyTorch 1.10
 - PyTorch Lightning 1.4.9
 - Check `environment.yml` for more packages.

## Usage

### Training

Use the following script for training:

```python train.py -h```

To reproduce all of the experiments in the paper, use:

```python train_all.py```

This script is also useful to train your own experiments if needed.

### Testing

The testing script requires both a model trained on **cropped** and **uncropped** images, and will evaluate both of them, using the uncropped model as rough segmentation.

```python test_model_crops.py -h```

If, instead, you would like to only test one of the models use:

```python test.py -h```

If you use the `--cropped` option the model will use ground-truth ideal crops. This can be seen as a best-case measure, or as a proxy result for using manual bounding boxes.

To reproduce all of the experiments in the paper, use:

```python test_all.py```

## Datasets

### `hist` - Kaggle 2018 Data Science Bowl / BBBC038v1

We used image set [BBBC038v1](https://bbbc.broadinstitute.org/BBBC038), available from the Broad Bioimage Benchmark Collection ([Caicedo et al., Nature Methods, 2019](https://doi.org/10.1038/s41592-019-0612-7)).

Download the **stage1_train.zip** and uncrompress it inside the `data` folder as follows:

```
data/
  hist/
    dsb2018/
      stage1_train/
        0a7.../
        0ac.../
        ...
```

Then run `python split_dsb18_dataset.py` to prepare the dataset.

### `polyp` - Kvasir-SEG

Jha, D. et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. In: MultiMedia Modeling. MMM 2020. Lecture Notes in Computer Science, vol 11962. Springer, Cham. https://doi.org/10.1007/978-3-030-37734-2_37

Download link: https://datasets.simula.no/kvasir-seg/

Download **kvasir-seg.zip** and uncrompress it to make the following folder structurre:

```
data/
  polyp/
    Kvasir-SEG/
      images/
      masks/
```

Then run `python split_polyp_dataset.py` to prepare the dataset.

### `aa` - AVT

Radl, L., Jin, Y., Pepe, A., Li, J., Gsaxner, C., Zhao, F., & Egger, J. (2022). AVT: Multicenter aortic vessel tree CTA dataset collection with ground truth segmentation masks. In Data in Brief (Vol. 40, p. 107801). Elsevier BV. https://doi.org/10.1016/j.dib.2022.107801

The dataset needs to added to the project in the form of the following folder structure (note that we have flattened the original directory structure):

```
data/
  aorta/
    data/
      avt/
        D1.nrrd
        D1.seg.nrrd
        ...
```

Then run `python split_aa_dataset.py` to prepare the dataset.
