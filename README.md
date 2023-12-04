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
