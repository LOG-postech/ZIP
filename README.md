# ZIP: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models

> Official PyTorch Implementation of [**ZIP: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models**](https://openreview.net/forum?id=2OegVbwvY2) (ICLR 2025)  
> **Seonghwan Park, Jaehyeon Jeong, Yongjun Kim, Jaeho Lee, and Namhoon Lee**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Research Highlights](#research-highlights)
3. [Setup](#setup)
   <!-- - [Clone the Repository](#clone-the-repository)
   - [Create and Activate Conda Environment](#create-and-activate-conda-environment)
   - [Install PyTorch and TorchVision](#install-pytorch-and-torchvision)
   - [Install Dependencies](#install-dependencies) -->
4. [Data Preparation](#data-preparation)
   - [Datasets from CoOp](#datasets-from-coop)
   - [Datasets from BlackVIP](#datasets-from-blackvip)
     - [SVHN](#svhn)
     - [Resisc45](#resisc45)
     - [CLEVR](#clevr)
5. [Running the Experiments](#running-the-experiments)
   - [Few-Shot Learning Benchmarks](#few-shot-learning-benchmarks)
   - [Base-to-New Generalization Benchmarks](#base-to-new-generalization-benchmarks)
   - [Cross Dataset Transfer Benchmarks](#cross-dataset-transfer-benchmarks)
   - [Out-of-Distribution Generalization Benchmarks](#out-of-distribution-generalization-benchmarks)
6. [Contact](#contact)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

---

## Abstract

<p align="center">
  <img src="imgs/main results.png" alt="Main Results" height="250">
</p>

> Recent studies have introduced various approaches for prompt-tuning black-box vision-language models, referred to as black-box prompt-tuning (BBPT). While BBPT has demonstrated considerable potential, it is often found that many existing methods require an excessive number of queries (i.e., function evaluations), which poses a significant challenge in real-world scenarios where the number of allowed queries is limited. To tackle this issue, we propose **`Z`eroth-order `I`ntrinsic-dimensional `P`rompt-tuning (`ZIP`)**, a novel approach that enables efficient and robust prompt optimization in a purely black-box setting. The key idea of ZIP is to reduce the problem dimensionality and the variance of zeroth-order gradient estimates, such that the training is done fast with far less queries. We achieve this by re-parameterizing prompts in low-rank representations and designing intrinsic-dimensional clipping of estimated gradients. We evaluate ZIP on 13+ vision-language tasks in standard benchmarks and show that it achieves an average improvement of approximately 6% in few-shot accuracy and 48% in query efficiency compared to the best-performing alternative BBPT methods, establishing a new state of the art. Our ablation analysis further shows that the proposed clipping mechanism is robust and nearly optimal, without the need to manually select the clipping threshold, matching the result of expensive hyperparameter search.

## Research Highlights

<p align="center">
  <img src="imgs/ZIP Overview.png" alt="ZIP Overview" width="90%">
</p>

### Key Contributions

- **Prompt Reparameterization:**  
  To address the challenge of dimensionality dependency (i.e., the number of queries scales with the dimensionality of the problem), we propose a novel low-rank representation. This approach reduces the dimensionality while effectively mitigating the loss of expressive power through feature sharing.

- **Clipped Zeroth-order Optimization:**  
  High variance in zeroth-order information can significantly degrade query efficiency. To tackle this, we propose a threshold-free gradient clipping method, termed **intrinsic dimensional clipping**. Inspired by prior studies on clipping thresholds, we set the clipping threshold to the square root of $\delta$, which corresponds to the standard deviation of the zeroth-order gradient, where $\delta$ is the dimensionality of the problem. This approach not only reduces the variance of zeroth-order information but also achieves near-optimal performance without requiring manual tuning.

- **Empirical Results:**  
  We extensively validate ZIP across **13+ datasets**, demonstrating its outstanding performance in _few-shot adaptability_ and _generalization on unseen distributions_.

---

## Setup
* Run the following commands to install Dassl.
  * [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

### Clone the Repository
Clone the ZIP repository from GitHub and navigate into the directory.
```shell
git clone https://github.com/LOG-postech/ZIP.git
cd ZIP
```

### Create and Activate Conda Environment
Create a new Conda environment named `zip` with Python version 3.9.12 and activate it.
```shell
conda create -y -n zip python=3.9.12
conda activate zip
```

### Install PyTorch and TorchVision

> **Note:** If you require a different CUDA version, please refer to the [PyTorch official website](https://pytorch.org/).

Install specific versions of `torch`, `torchvision`, and `torchaudio` using `pip` with the specified CUDA version.

```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install Dependencies

Install all necessary dependencies listed in the `requirements.txt` file.
```shell
pip install -r requirements.txt
```

---

## Data Preparation

### Datasets from [CoOp](https://github.com/KaiyangZhou/CoOp)

Prepare the following 10 datasets by following the instructions in the [CoOp dataset guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md):

- **Caltech101**
- **OxfordPets**
- **Flowers102**
- **Food101**
- **FGVCAircraft**
- **SUN397**
- **DTD**
- **EuroSAT**
- **UCF101**
- **ImageNet**

**Note:** The same few-shot splits used in CoOp are applied for these datasets.

### Datasets from [BlackVIP](https://github.com/changdaeoh/BlackVIP)

Follow the steps below to prepare the additional datasets:

#### SVHN

1. **Create Directory:** Create a folder named `svhn/` under `$DATA`.

2. **Download Dataset:** Run the dataset download script.

   - **Important:** Replace the `DATA_PATH` in **line 52** of the script with your desired path.

3. **Download Split File:** Download [split_mlai_SVHN.json](https://drive.google.com/file/d/1dnjnMX-sr7FClb6EUywc-tNx6e1YaGpu/view?usp=sharing) and place it in `$DATA/svhn`.

#### Resisc45

1. **Create Directory:** Create a folder named `resisc45/` under `$DATA`.

2. **Download and Extract Dataset:**
   
   - Download [NWPU-RESISC45.rar](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp).
   - Extract the contents into `$DATA/resisc45`.

3. **Download Split File:** Download [split_mlai_Resisc45.json](https://drive.google.com/file/d/1QTThkyN-p58hAxN7wpBntO4CncTf9qtF/view?usp=share_link) and place it in `$DATA/resisc45`.

#### CLEVR

1. **Download and Extract Dataset:**
   
   - Download [CLEVR_v1.0.zip](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip).
   - Extract it into `$DATA`.

2. **Download Split File:** Download [split_mlai_CLEVR.json](https://drive.google.com/file/d/1L4DjruSBez66W_Uezyo2mN7siDORkgeO/view?usp=share_link) and place it in `$DATA/CLEVR_v1.0`.

---

## Running the Experiments

### Few-Shot Learning Benchmarks

1. **Navigate to the Directory:** Go to the `ZIP/scripts/few_shot` directory.
```shell
cd ZIP/scripts/few_shot
```

2. **Run the Commands:** Execute the desired method on the target dataset.
```shell
# Replace {METHOD} with the desired method and {DATASET} with the target dataset
sh {METHOD}.sh {DATASET}
```

- **`METHOD`:** Specify the method name, such as `ZIP`.
- **`DATASET`:** Specify the dataset name, for example, `imagenet` or `caltech101`.

**Note:** Valid names correspond to the filenames located in the `ZIP/configs/` directory.

### Base-to-New Generalization Benchmarks

1. **Navigate to the Directory:** Go to the `ZIP/scripts/base_to_new` directory.
```shell
cd ZIP/scripts/base_to_new
```

2. **Train the Model:** Train the model using the specified method and dataset.
```shell
# Replace {METHOD} with the desired method and {DATASET} with the target dataset
sh train_{METHOD}.sh {DATASET}
```

3. **Test the Model:** Test the trained model using the specified method and dataset.
```shell
# Replace {METHOD} with the desired method and {DATASET} with the target dataset
sh test_{METHOD}.sh {DATASET}
```

### Cross Dataset Transfer Benchmarks

1. **Train ImageNet as Source:** Ensure ImageNet is trained as a source dataset before proceeding.

2. **Navigate to the Directory:** Go to the `ZIP/scripts/cross_dataset_transfer` directory.
```shell
cd ZIP/scripts/cross_dataset_transfer
```

3. **Run the Commands:** Execute the cross-dataset transfer testing using the desired method.
```shell
# Replace {METHOD} with the desired method
sh xd_test_{METHOD}.sh
```

### Out-of-Distribution Generalization Benchmarks

1. **Train ImageNet as Source:** Ensure ImageNet is trained as a source dataset before proceeding.

2. **Navigate to the Directory:** Go to the `ZIP/scripts/out_of_distribution` directory.
```shell
cd ZIP/scripts/out_of_distribution
```

3. **Run the Commands:** Execute the out-of-distribution generalization testing using the desired method.
```shell
# Replace {METHOD} with the desired method
sh xd_test_im_{METHOD}.sh
```

---

## Contact

For any questions, discussions, or proposals, please contact:

📧 **Email:** [shpark97@postech.ac.kr](mailto:shpark97@postech.ac.kr)

---

## Citation

If you use our code in your research, please consider citing:

```bibtex
@inproceedings{
    park2025zip,
    title={{ZIP}: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models},
    author={Seonghwan Park, Jaehyeon Jeong, Yongjun Kim, Jaeho Lee, and Namhoon Lee},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=2OegVbwvY2}
}
```

---

## Acknowledgements

Our experimental pipeline is built upon the following repositories:

- [CoOp, CoCoOp](https://github.com/KaiyangZhou/CoOp)  
- [COMPACTER](https://github.com/rabeehk/compacter)

For baseline construction, we referred to and borrowed code from these repositories:

- [BlackVIP](https://github.com/changdaeoh/BlackVIP?tab=readme-ov-file)  
- [BAR](https://github.com/yunyuntsai/Black-box-Adversarial-Reprogramming)  
- [BPTVLM](https://github.com/BruthYU/BPT-VLM)

We express our gratitude to the authors of these works (Zhou et al., Mahabadi et al., Oh et al., Tsai et al., Yu et al.) for sharing their code and making their contributions available.