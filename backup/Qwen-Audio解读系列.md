# 简介

Qwen-Audio是由阿里的Qwen团队开发的语音多模态大模型，分为Qwen-Audio，Qwen2-Audio,两者均具备强大的听觉能力，具备处理语音、音频事件和音乐歌曲的广泛能力

Paper：[Qwen-Audio](https://arxiv.org/abs/2311.07919)
Github：[Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)
Paper：[Qwen2-Audio](https://arxiv.org/abs/2407.10759)
Github：[Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
（注释：Qwen-Audio的代码比Qwen2-Audio详细很多）
## 数据

Qwen

![Image](https://github.com/user-attachments/assets/2d2e32e9-ff90-45b4-a828-8665cfbd4e71)

- Speech：约89.2k(不足1k按照1k的计算)
- Sound：约16.8k
- Music：约40.5k

Qwen2-Audio
- Spech：37k
- Sound：10k
- Music：140k
## 模型
Qwen-Audio：Qwen-Audio的实际结构为 Whisper-large-v2+MLP（堆叠）+Qwen-7B(具体型号没找到)

![Image](https://github.com/user-attachments/assets/bc90e9ea-03b4-4169-8e5c-2ceb00658cc1)

Qwen2-Audio:实际结构为Whisperlarge-v3+平均池化层(长2)+MLP（没有堆叠）+Qwen-7B(具体型号没找到)
![Image](https://github.com/user-attachments/assets/6319993a-5269-4a2b-82cc-00ed24b690e7)
## 训练
Qwen-audio:Qwen-Audio采用的是类似Whisper的训练框架，即为预测Token，并且Qwen-Audio只训练Whisper+MLP，不对语言模型进行微调，而且只是采用预训练，SFT得到的是Qwen-Audio-Chat

Qwen2-audio:采用Pretrain+SFT+DPO，具体训练的部分没提到
- Pretrain：训练ASR+ACC
- SFT：分为两类任务，语音分析任务：文本指令+语音，聊天任务：语音
- DPO：
## 结果
Qwen-Audio

![Image](https://github.com/user-attachments/assets/781b1d46-34a6-46fb-ab4e-bdc066b3a0fd)

Qwen2-Audio

![Image](https://github.com/user-attachments/assets/ee9a8c6c-42c7-4e6c-a6f9-abff8216c6b1)