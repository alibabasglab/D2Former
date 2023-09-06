# D2Former: A Fully Complex Dual-Path Dual-Decoder Conformer Network using Joint Complex Masking and Complex Spectral Mapping for Monaural Speech Enhancement (https://arxiv.org/abs/2302.11832)

## Abstract:
Monaural speech enhancement has been widely studied using real networks in the time-frequency (TF) domain. However, the input and the target are naturally complex-valued in the TF domain, a fully complex network is highly desirable for effectively learning the feature representation and modelling the sequence in the complex domain. Moreover, phase, an important factor for perceptual quality of speech, has been proved learnable together with magnitude from noisy speech using complex masking or complex spectral mapping. Many recent studies focus on either complex masking or complex spectral mapping, ignoring their performance boundaries. To address above issues, we propose a fully complex dual-path dual-decoder conformer network (D2Former) using joint complex masking and complex spectral mapping for monaural speech enhancement. In D2Former, we extend the conformer network into the complex domain and form a dual-path complex TF self-attention architecture for effectively modelling the complex-valued TF sequence. We further boost the TF feature representation in the encoder and the decoders using a dual-path learning structure by exploiting complex dilated convolutions on time dependency and complex feedforward sequential memory networks (CFSMN) for frequency recurrence. In addition, we improve the performance boundaries of complex masking and complex spectral mapping by combining the strengths of the two training targets into a joint-learning framework. As a consequence, D2Former takes fully advantages of the complex-valued operations, the dual-path processing, and the joint-training targets. Compared to the previous models, D2Former achieves state-of-the-art results on the VoiceBank+Demand benchmark with the smallest model size of 0.87M parameters. 

[Demo of audio samples](https://github.com/alibabasglab/D2Former/tree/main/D2Former_enh_VoiceBank_DEMAND/) 

The manuscript is accepted in ICASSP 2023. 

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
If you want to train the model, run run_train.sh
```
sh run_train.sh
```

### Step 4:
Evaluation with the best ckpt:
```
sh run_eval.sh
```

## Citation:
```
@article{cao2022cmgan,
  title={D2{F}ormer: A Fully Complex Dual-Path Dual-Decoder Conformer Network using Joint Complex Masking and Complex Spectral Mapping for Monaural Speech Enhancement},
  author={Shengkui Zhao and Bin Ma},
  journal={arXiv:2302.11832},
  year={2023}
}
```
## Reference code:
CMGAN: https://github.com/ruizhecao96/CMGAN
