
[官网教程](https://pytorch.org/audio/main/)
## 前言

torchaudio.transformers是Pytorch官方提供的有关音频处理的库

### 读写
加载与保存
- torchaudio.info(path)
- torchaudio.load(path)
- torchaudio.save(path,wavform,sample_rate)
### StreamReader & StreamWriter(略)
可用于流式的读写，支持麦克风，网络读写
### transformers
```
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
```
### Resample
要将音频波形从一个freqeeuncy重新采样，您可以使用 torchaudio.transforms.Resample 或者 torchaudio.functional.resample() 
```
resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
```

```
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=6)
```
### [Data Augmentation](https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html)

####  filters滤波器

```python
# Define effects
effect = ",".join(
    [
        "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
        "atempo=0.8",  # reduce the speed
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
        # Applying echo gives some dramatic feeling
    ],
)
effector = torchaudio.io.AudioEffector(effect=effect)
effector.apply(waveform, sample_rate)
```

#### RIR混响模拟
![[rir.wav]]
```python
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)
speech, _ = torchaudio.load(SAMPLE_SPEECH)
augmented = F.fftconvolve(speech, rir)
```
#### 添加噪声
```python
speech, _ = torchaudio.load(SAMPLE_SPEECH)
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : speech.shape[1]]

snr_dbs = torch.tensor([20, 10, 3])
noisy_speeches = F.add_noise(speech, noise, snr_dbs)
```

```python
import torch

import torchaudio

import torchaudio.functional as F

import random

  

# 加载语音和噪声文件

speech, _ = torchaudio.load("/hpc_stor01/home/yangui.fang_sx/workingspace/tools/R8009_M8024-8076-000350_000545.flac")

noise, _ = torchaudio.load("/hpc_stor01/home/yangui.fang_sx/workingspace/tools/rir.wav")

# 确保语音和噪声的采样率相同

assert fs == noise.shape[1], "语音和噪声的采样率必须相同"

  

# 如果语音比噪声长，随机选择噪声的起始点

if speech.shape[1] > noise.shape[1]:

    # 随机选择噪声的起始点

    start_idx = random.randint(0, speech.shape[1] - noise.shape[1])

    # 在语音的随机位置开始添加噪声

    speech_with_noise = speech.clone()

    speech_with_noise[:, start_idx:start_idx + noise.shape[1]] += noise

else:

    # 如果噪声比语音长，从噪声的随机位置开始截取

    start_idx = random.randint(0, noise.shape[1] - speech.shape[1])

    noise = noise[:, start_idx:start_idx + speech.shape[1]]

    # 直接将噪声添加到语音中

    snr_dbs = random.randomint(1, 30)

    noisy_speeches = F.add_noise(speech, noise, snr_dbs)

  

# 保存带噪语音信号

Audio(speech_with_noise, rate=fs)

# output_path = "noisy_speech.wav"

# torchaudio.save(output_path, speech_with_noise, fs)

# print(f"Saved noisy speech to {output_path}")
```
#### 编码

```python
encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
encoder.apply(waveform, sample_rate)	
```
 format,encoder支持下面的，
- "wav","pcm_mulaw"
- "g722"
- "ogg", encoder="vorbis"

[### Feature Extract 特征提取](https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html)

#### Spectrogram

```python
spectrogram = T.Spectrogram(n_fft=512)

# Perform transform
spec = spectrogram(wav)
```
#### GriffinLim 
基于规则的从频谱恢复波形

```python
# Define transforms
n_fft = 1024
spectrogram = T.Spectrogram(n_fft=n_fft)
griffin_lim = T.GriffinLim(n_fft=n_fft)

# Apply the transforms
spec = spectrogram(SPEECH_WAVEFORM)
reconstructed_waveform = griffin_lim(spec)
```
#### Melspectrogram

```python
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(SPEECH_WAVEFORM)
```

#### MFCC

```python
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)
```
#### LFCC
```python
n_fft = 2048
win_length = None
hop_length = 512
n_lfcc = 256

lfcc_transform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=n_lfcc,
    speckwargs={
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
    },
)

lfcc = lfcc_transform(SPEECH_WAVEFORM)
plot_spectrogram(lfcc[0], title="LFCC")
```

#### Picth 音高

```python
pitch = F.detect_pitch_frequency(SPEECH_WAVEFORM, SAMPLE_RATE)
```



#### SpecAugment

```python

spec = spec.permute(1, 0).unsqueeze(0)
stretch = T.Spectrogram(n_freq=通道数)
rate = random.random()*0.2 + 0.9
spec = stretch(spec, rate)

Timemasking = T.TimeMasking(time_mask_param=100)
Frequencymasking = T.FrequencyMasking(freq_mask_param=27)
spec = Timemasking(spec)
spec = Timemasking(spec)
spec = Frequencymasking(spec)
spec = Frequencymasking(spec)  
```

### [CTC强制对齐](https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)

- torchaudio.functional.forced_align()
[多语言对齐](https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html)


### 波形合成

- [振荡波](https://pytorch.org/audio/stable/tutorials/oscillator_tutorial.html)
- [波](https://pytorch.org/audio/stable/tutorials/additive_synthesis_tutorial.html)
### [滤波器设计](https://pytorch.org/audio/stable/tutorials/filter_design_tutorial.html)




### Piplines

#### [基于CTC语言识别](https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html)：
#### [在线的基于RNN-T的ASR](https://pytorch.org/audio/stable/tutorials/online_asr_tutorial.html)
#### [基于麦克风的流式识别](https://pytorch.org/audio/stable/tutorials/device_asr.html)


### [波束成形](https://pytorch.org/audio/stable/tutorials/mvdr_tutorial.html)

```python
import torch
import torchaudio
import torchaudio.transforms as transforms

# 1. 加载多通道音频
audio_path = "path_to_your_audio_file.wav"  # 替换为你的音频路径
waveform, sample_rate = torchaudio.load(audio_path)
specgram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256)(waveform)

# 2. 计算 PSD 矩阵
psd = transforms.PSD()(specgram)

# 3. 定义参考通道
reference_channel = 0

# 4. 使用 SoudenMVDR 进行波束形成
mvdr = transforms.SoudenMVDR(ref_channel=reference_channel)
enhanced_specgram = mvdr(specgram, psd, psd, reference_channel=reference_channel)

# 5. 将增强后的频谱转换回时域信号
enhanced_waveform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256)(enhanced_specgram)

# 6. 保存增强后的音频
torchaudio.save("enhanced_output.wav", enhanced_waveform, sample_rate)
```

```python
import torch
import torchaudio
import torchaudio.transforms as transforms
import torchaudio.functional as functional

# 1. 加载多通道音频
audio_path = "path_to_your_audio_file.wav"  # 替换为你的音频路径
waveform, sample_rate = torchaudio.load(audio_path)
specgram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256)(waveform)

# 2. 计算 RTF 向量
rtf = functional.rtf_evd(specgram, reference_channel=0)

# 3. 计算噪声的 PSD 矩阵
psd_n = functional.psd(specgram, mask=noise_mask)

# 4. 定义 RTFMVDR 模块
rtf_mvdr = transforms.RTFMVDR(reference_channel=0)

# 5. 应用波束形成
enhanced_specgram = rtf_mvdr(specgram, rtf, psd_n, reference_channel=0)

# 6. 将增强后的频谱转换回时域信号
enhanced_waveform = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256)(enhanced_specgram)

# 7. 保存增强后的音频
torchaudio.save("rtf_mvdr_output.wav", enhanced_waveform, sample_rate)
```