
## Vocoders

TTS的工作主要是把文本信息转成音频信息，其大致流程分为前端处理和后端处理两个部分。前端的工作主要是语言领域的处理，主要包括分句、文本正则、分词、韵律预测、拼音预测（g2p)，多音字等等。后端的主要工作是把前端预测的语言特征转成音频的时域波形，大体包括声学模型和声码器，其中声学模型是把语言特征转成音频的声学特征，声码器的主要功能是把声学特征转成可播放的语音波形。声码器的好坏直接决定了音频的音质高低，尤其是近几年来基于神经网络声码器的出现，使语音合成的质量提高一个档次。目前，声码器大致可以分为基于相位重构的声码器和基于神经网络的声码器。基于相位重构的声码器主要因为TTS使用的声学特征（mel特征等等）已经损失相位特征，因此使用算法来推算相位特征，并重构语音波形。基于神经网络的声码器则是直接把声学特征和语音波形做mapping，因此合成的音质更高。目前，比较流行的神经网络声码器主要包括wavenet、wavernn、melgan、waveglow、fastspeech和lpcnet等等。

本部分我们介绍声码器的一些paper,主要是基于神经网络的神声码器的介绍。

### 1. MelGAN

!> https://arxiv.org/abs/1910.06711

<!-- https://blog.csdn.net/qq_28662689/article/details/105971998 -->


### 2. MultiBandMelGAN

!> https://arxiv.org/abs/2005.05106


### 3. ParallelWaveGAN



### 4. GAN-TTS discriminators


### 5. WaveNet


### 6. WaveRNN


### 7. WaveGrad


### 8. HiFiGAN V1/V2



### 9. UnivNet


### 10. WaveGlow


