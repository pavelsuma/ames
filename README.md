# AMES: Asymmetric and Memory-Efficient Similarity
***
This repository contains the code for the paper ["AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval"](https://arxiv.org/abs/2408.03282), by the authors Pavel Suma, Giorgos Kordopatis-Zilos, Ahmet Iscen, and Girogos Tolias.
In Proceedings of the European Conference on Computer Vision (ECCV), 2024

## TLDR

Transformer-based model that offers a good balance between performance and memory.

***

## Setup
***
This code was implemented using Python 3.11.5 and the following dependencies:

```
torch==2.4.1
hydra-core==1.3.2
numpy==2.1.2
tqdm==4.66.5
h5py==3.12.1
```


## Trained models
***
We provide AMES trained on GLDv2 in four variants. Available models are trained with full-precision (fp) or binary (dist) local descriptors extracted from either DINOv2 or CVNet backbone. 

You can download all models from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks).

## Evaluation
***
In order to evaluate the performance of our models, you need to have the extracted local descriptors of the datasets.
We provide them for ROxford5k, and RParis6k. For other datasets, please see below how to extract them yourself.
The descriptors along with the extracted global similarities for the query nearest neighbors can be downloaded from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data).

You can also run the following command to download them into the `data` folder.
```
wget -r -nH --cut-dirs=5 --no-parent --reject="index.html*" -P data http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data/
```

A sample command to run the evaluation on these two datasets is as follows:

```
python3 -u src/evaluate.py --multirun \
        desc_name=dinov2 \
        data_root=data \
        resume=networks/dinov2_ames.pt \
        model.binarized=False \
        dataset@test_dataset=roxford \
        test_dataset.query_sequence_len=600 \
        test_dataset.sequence_len=50 \
        test_dataset.batch_size=300 \
        test_dataset.lamb=[0.55] \
        test_dataset.temp=[0.3] \
        test_dataset.num_rerank=[100]
```

Hyperparameters used for our best performing AMES experiments, tuned on GLDv2 public test split, are as follows:

| Parameter  | DINOv2 (fp) | DINOV2 (dist) | CVNet (fp) | CVNet (dist) |
|------------|-------------|---------------|------------|--------------|
| `lamb` (λ) | 0.55        | 0.35          | 0.85       | 0.65         |
| `temp` (γ) | 0.30        | 0.10          | 0.80       | 0.20         |



## Training
***

Coming soon...

## Extracting descriptors
***

The code contains scripts to extract global and local descriptors of GLDv2, ROxford5k, and RParis6k.
Supported backbones are CVNet and DINOv2, however the code can be easily extended to other CNN and ViT backbones.  

Revisited Oxford and Paris (ROP) dataset, along with 1M distractors can be downloaded from the [original site](http://cmp.felk.cvut.cz/revisitop/).
Likewise, GLDv2 train and test can be downloaded in the [official repository](https://github.com/cvdfoundation/google-landmark).

You will need additional dependencies for the extraction of local descriptors:
```
opencv-python-headless==4.10.0.84
```

By default, descriptors are stored in format such as `dinov2_gallery_local.hdf5` in a corresponding dataset folder under `save_path`.
Images are loaded from the `data_path` folder. For each dataset split, a `.txt` file is required to specify the image paths. 
We provide these files for each dataset in the `data` folder.

Extraction of descriptors can be done by running the following command:
```
export PYTHONPATH=$(pwd):$PYTHONPATH
python extract/extract_descriptors.py --dataset [gldv2|roxford5k|rparis6k] \
                              --backbone [cvnet|dinov2] \
                              --weights [path_to_weights] \
                              --save_path data \
                              --data_path [path_to_images] \
                              --split [_gallery|_query|] \
                              --file_name test_gallery.txt \
                              --desc_type "local" \
                              --detector [path_to_detector_weights]
```

Weights parameter is only needed for CVNet. Please follow the [original repository](https://github.com/sungonce/CVNet) to download them.
Weights for our two trained feature detectors (one for cvnet, and one for dinov2) are available [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks).
Take a look into the `extract/extract_descriptors.py` file for more argument parameter details.

## Citation
***

```
@InProceedings{Suma_2024_ECCV,
    author    = {Suma, Pavel and Kordopatis-Zilos, Giorgos and Iscen, Ahmet and Tolias, Giorgos},
    title     = {AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2024}
}
```

## Acknowledgements

This code is based on the repository of RRT:
[Instance-level Image Retrieval using Reranking Transformers](https://github.com/uvavision/RerankingTransformer).

CVNet extraction code is based on the repository of CVNet:
[Correlation Verification for Image Retrieval](https://github.com/sungonce/CVNet)
