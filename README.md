# Testing GLIGEN for my master thesis

<div align="center">
<h1>Training-Free Layout Control with Cross-Attention Guidance</h1>

[[Yuheng Li](https://yuheng-li.github.io/)], [[Haotian Liu](https://hliu.cc/)], [[Qingyang Wu]()], [[Fangzhou Mu](https://pages.cs.wisc.edu/~fmu/)], [[Jianwei Yang](https://jwyang.github.io/)], [[Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)], [[Chunyuan Li](https://chunyuan.li/)], [[Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)],

[[Paper](https://arxiv.org/abs/2301.07093)] [[Project Page](https://gligen.github.io/)] [[Demo](https://huggingface.co/spaces/gligen/demo)]

<h3>Abstract</h3>

Large-scale text-to-image diffusion models have made amazing advances. However, the status quo is to use text input alone, which can impede controllability. In this work, we propose GLIGEN, Grounded-Language-to-Image Generation, a novel approach that builds upon and extends the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on grounding inputs. To preserve the vast concept knowledge of the pre-trained model, we freeze all of its weights and inject the grounding information into new trainable layers via a gated mechanism. Our model achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configuration and concepts. GLIGEN’s zero-shot performance on COCO and LVIS outperforms that of existing supervised layout-to-image baselines by a large margin.

</div>

## Quick start

```bash
conda create --name gligen python=3.10
conda activate gligen
pip install -r requirements.txt
```

## Image generation

The .csv file containing the prompts should be inside a folder named `prompts` that is posiotioned in the root of the project.

The .csv file used is expected to have the following structure (no limits in the number of objects):
`id,prompt,obj1,bbox1,obj2,bbox2,obj3,bbox3,obj4,bbox4`

## Citation

```bibtex
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}

```
