

[Whisper的PT文件下载地址](https://gitcode.csdn.net/65ed73ad1a836825ed799909.html)
[在Colab微调Whisper](https://huggingface.co/blog/fine-tune-whisper)

## Whisper简介

Whisper是Openai开发的语音识别工具，通常我们可以用Whisper库或者Transformers来使用Whisper，本文专注于Whisper库的使用，安装方式如下

```python
pip install -U openai-whisper
```

还需要安装ffmpeg
```
conda install ffmpeg(支持非sudo用户)
sudo apt install ffmpeg 
```

Whisper包含encoder和decoder两个部分,encoder接受30s的音频长度的输出，编码成为特征向量，decoder负责解码


## Whisper识别
`transcribe`:
- 最简单的识别方式
```python
import whisper
model = whisper.load("path/name")
text = model.transcribe("wav_path")

```
`decode`：
- 注意到音频会被`pad_or_trim`函数填充或者裁剪为30s长度
- `decode`支持`mel`输入或者`encoder`编码后的特征输入
- `nmels=80/128`,128适合v3，80适合其他版本
```python
import whisper
import numpy as np
model = whisper.load_model("")
audio = whisper.load_audio("")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels).to("cuda").to(model.device)
result = model.decode(mel)
# result = whisper.decode(model, mel)


```

```python
import whisper
import numpy as np
model = whisper.load_model("")
audio = whisper.load_audio("")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels).to("cuda").to(model.device)
result =model.decode(mel)
encoder_output = model.encoder(mel.unsqueeze(0))

result = model.decode(encoder_output)
# result = whisper.decode(model, encoder_output)
# 打印encoder输出的形状
```
## Whisper提取特征
如果采用whisper的encoder提取特征，音频首先要被填充到30s
```python
import whisper
import numpy as np
model = whisper.load_model("")
audio = whisper.load_audio("")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels).to("cuda").to(model.device)
result =model.decode(mel)
encoder_output = model.encoder(mel.unsqueeze(0))

```

可以采用替代forward函数的方法来提取不定长度的特征,因为encoder不支持小于30s长度音频的原因在于
- `x = (x + self.positional_embedding).to(x.dtype)`

```python
import types
import whisper
import torch
import torch.nn as nn
import torch.nn.functional as F
def whisper_encoder_forward_monkey_patch(self, x: torch.Tensor):
	"""
	x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
	the mel spectrogram of the audio
	"""
	x = F.gelu(self.conv1(x))
	x = F.gelu(self.conv2(x))
	x = x.permute(0, 2, 1)
	# assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
	# x = (x + self.positional_embedding).to(x.dtype)
	x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
	for block in self.blocks:
		x = block(x)
		x = self.ln_post(x)
	return x
```

```python

encoder = whisper.load_model("base").encoder
encoder.whisper_encoder_forward_monkey_patch = types.MethodType(whisper_encoder_forward_monkey_patch, encoder)
audio_path = ""
audio = whisper.load_audio(audio_path)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
features = encoder.whisper_encoder_forward_monkey_patch(mel.unsqueeze(0))
```

```python
whisper.model.AudioEncoder.forward = forward
model = whisper.load_model("")
audio = whisper.load_audio("")
mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels).to("cuda").to(model.device).unsquenze(0)
outout = model.encoder(mel)
```
如果需要whisper提取出的该特征进行解码，必须使用options

```python
options = whisper.DecodingOptions(
    task="transcribe",
    language="zh",
    without_timestamps=True,
    beam_size=4,

)
print(whisper.decode(model,mel,options))
```

## Whisper Options
- `task`:默认为`transcribe`，可以设置为`translate`,即为将输出翻译为英语



## Huggingface用法
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
# 加载预训练的Whisper模型和处理器
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
# 假设你有一个输入的语音特征
input_features = ...  # 这里应该是预处理后的语音特征
# 将输入特征移动到模型所在的设备上
input_features = input_features.to(model.device)
# 使用分块算法生成输出
outputs = model.generate(
    input_features=input_features,
    return_dict_in_generate=True,
    output_hidden_states=True,
    chunk_length=30,  # 设置分块长度
    stride_length=15  # 设置步长
)
# 解码生成的序列
transcriptions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
# 打印转录结果
print(transcriptions)

```