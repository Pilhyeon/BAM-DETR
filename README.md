# BAM-DETR
### Official Pytorch Implementation of 'BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos'
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bam-detr-boundary-aligned-moment-detection/moment-retrieval-on-qvhighlights)](https://paperswithcode.com/sota/moment-retrieval-on-qvhighlights?p=bam-detr-boundary-aligned-moment-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bam-detr-boundary-aligned-moment-detection/moment-retrieval-on-charades-sta)](https://paperswithcode.com/sota/moment-retrieval-on-charades-sta?p=bam-detr-boundary-aligned-moment-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bam-detr-boundary-aligned-moment-detection/natural-language-moment-retrieval-on-tacos)](https://paperswithcode.com/sota/natural-language-moment-retrieval-on-tacos?p=bam-detr-boundary-aligned-moment-detection)


![architecture](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points/assets/16102333/774267a6-8a65-4a7b-bc60-c832d9e5c745)

> **BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos**<br>
> [Pilhyeon Lee](https://pilhyeon.github.io/)&dagger;, Hyeran Byun <br>
> (&dagger;: Corresponding author)
>
> Paper: https://arxiv.org/abs/2312.00083
>
> **Abstract:** *Temporal sentence grounding aims to localize moments relevant to a language description. Recently, DETR-like approaches have shown notable progress by decoding the center and length of a target moment from learnable queries. However, they suffer from the issue of center misalignment raised by the inherent ambiguity of moment centers, leading to inaccurate predictions. To remedy this problem, we introduce a novel boundary-oriented moment formulation. In our paradigm, the model no longer needs to find the precise center but instead suffices to predict any anchor point within the interval, from which the onset and offset are directly estimated. Based on this idea, we design a Boundary-Aligned Moment Detection Transformer (BAM-DETR), equipped with a dual-pathway decoding process. Specifically, it refines the anchor and boundaries within parallel pathways using global and boundary-focused attention, respectively. This separate design allows the model to focus on desirable regions, enabling precise refinement of moment predictions. Further, we propose a quality-based ranking method, ensuring that proposals with high localization qualities are prioritized over incomplete ones. Extensive experiments verify the advantages of our methods, where our model records new state-of-the-art results on three benchmarks.*

----------

## Prerequisites
### Recommended Environment
* Python 3.7
* Pytorch 1.9
* Tensorboard 1.15

### Dependencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.
For anaconda setup, please refer to the official [Moment-DETR github](https://github.com/jayleicn/moment_detr).

### Data Preparation
<b>1. Prepare datasets</b>
1. Prepare [QVHighlights](https://github.com/jayleicn/moment_detr) dataset.
2. Extract features with Slowfast and CLIP models.
    - We recommend downloading [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB)
3. Extract it under project root directory.
```
tar -xf path/to/moment_detr_features.tar.gz
```

### Training
Training can be launched by running the command below.
If you want to try other training options, please refer to the config file [bam_detr/config.py](bam_detr/config.py).
```
bash bam_detr/scripts/train.sh 
```

### Evaluation
Once the model is trained, you can use the following command for inference.
```
bash bam_detr/scripts/inference.sh CHECKPOINT_PATH SPLIT_NAME
``` 
where `CHECKPOINT_PATH` is the path to the saved checkpoint, `SPLIT_NAME` is the split name for inference, can be one of `val` and `test`.

## References
We note that this repo is heavily based on the following codebases. We express our appreciation to the authors for sharing their code.

* [Moment-DETR](https://github.com/jayleicn/moment_detr)
* [QD-DETR](https://github.com/wjun0830/QD-DETR)


## Citation
If you find this code useful, please cite our paper.

~~~~
@inproceedings{lee2024bam-detr,
  title={Bam-detr: Boundary-aligned moment detection transformer for temporal sentence grounding in videos},
  author={Lee, Pilhyeon and Byun, Hyeran},
  booktitle={European Conference on Computer Vision},
  pages={220--238},
  year={2024},
  organization={Springer}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Pilhyeon Lee (pilhyeon.lee@inha.ac.kr).
