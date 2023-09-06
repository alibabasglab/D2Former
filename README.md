# CMGAN: Conformer-based Metric GAN for speech enhancement (https://arxiv.org/abs/2203.15149)

## Abstract:
Recently, convolution-augmented transformer (Conformer) has achieved promising performance in automatic speech recognition (ASR) and time-domain speech enhancement (SE), as it can capture both local and global dependencies in the speech signal. In this paper, we propose a conformer-based metric generative adversarial network (CMGAN) for SE in the time-frequency (TF) domain. In the generator, we utilize two-stage conformer blocks to aggregate all magnitude and complex spectrogram information by modeling both time and frequency dependencies. The estimation of magnitude and complex spectrogram is decoupled in the decoder stage and then jointly incorporated to reconstruct the enhanced speech. In addition, a metric discriminator is employed to further improve the quality of the enhanced estimated speech by optimizing the generator with respect to a corresponding evaluation score. Quantitative analysis on Voice Bank+DEMAND dataset indicates the capability of CMGAN in outperforming various previous models with a margin, i.e., PESQ of 3.41 and SSNR of 11.10 dB. 

[Demo of audio samples](https://ruizhecao96.github.io/) 

The manuscript is accepted in INTERSPEECH2022. Source code is released!

## How to train:

### Step 1:
In src:

```pip install -r requirements.txt```

### Step 2:
Download VCTK-DEMAND dataset with 16 kHz, change the dataset dir:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```

### Step 3:
If you want to train the model, run train.py
```
python3 train.py --data_dir <dir to VCTK-DEMAND dataset>
```

### Step 4:
Evaluation with the best ckpt:
```
python3 evaluation.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

## Model and Comparison:
<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figure/Overview.png" width="600px">

<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figure/Table.png" width="600px">

## Citation:
```
@article{cao2022cmgan,
  title={CMGAN: Conformer-based Metric GAN for Speech Enhancement},
  author={Cao, Ruizhe and Abdulatif, Sherif and Yang, Bin},
  journal={arXiv preprint arXiv:2203.15149},
  year={2022}
}
```
