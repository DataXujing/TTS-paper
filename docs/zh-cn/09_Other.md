## 其他

这一部分我们会以实战的方式详细的介绍如何使用Pytorch, paddlepaddle,elmo和一些开源的语音克隆算法实现从训练TTS模型，语音克隆模型到部署TTS模型的过程。

<!-- 
1.在pytorch下训练一个TTS模型
2.在elmo下训练一个TTS模型
3.粤语模型的训练和部署
4.语音clone的尝试
 -->

### 1.基于PaddleSpeech的粤语TTS在叫号系统中的应用
 
<!-- ToDo: 下周工作 -->

#### 1.前言
##### 1.1 背景知识

为了更好地了解文本转语音任务的要素，我们先简要地回顾一下文本转语音的发展历史。文本转语音，又称语音合成（Speech Sysnthesis），指的是将一段文本按照一定需求转化成对应的音频，这种特性决定了的输出数据比输入长得多。文本转语音是一项包含了语义学、声学、数字信号处理以及机器学习的等多项学科的交叉任务。虽然辨识低质量音频文件的内容对人类来说很容易，但这对计算机来说并非易事。

按照不同的应用需求，更广义的语音合成研究包括：语音转换，例如说话人转换、语音到歌唱转换、语音情感转换、口音转换等；歌唱合成，例如歌词到歌唱转换、可视语音合成等。

##### 1.2 发展历史

在第二次工业革命之前，语音的合成主要以机械式的音素合成为主。1779年，德裔丹麦科学家 Christian Gottlieb Kratzenstein 建造了人类的声道模型，使其可以产生五个长元音。1791年， Wolfgang von Kempelen 添加了唇和舌的模型，使其能够发出辅音和元音。贝尔实验室于20世纪30年代发明了声码器（Vocoder），将语音自动分解为音调和共振，此项技术由 Homer Dudley 改进为键盘式合成器并于 1939年纽约世界博览会展出。

第一台基于计算机的语音合成系统起源于20世纪50年代。1961年，IBM 的 John Larry Kelly，以及 Louis Gerstman 使用 IBM 704 计算机合成语音，成为贝尔实验室最著名的成就之一。1975年，第一代语音合成系统之一 —— MUSA（MUltichannel Speaking Automation）问世，其由一个独立的硬件和配套的软件组成。1978年发行的第二个版本也可以进行无伴奏演唱。90 年代的主流是采用 MIT 和贝尔实验室的系统，并结合自然语言处理模型。

<div align=center>
    <img src="zh-cn/img/ch9/01/p1.png" /> 
</div>

##### 1.3 主流方法

当前的主流方法分为**基于统计参数的语音合成**、**波形拼接语音合成**、**混合方法**以及**端到端神经网络语音合成**。基于参数的语音合成包含隐马尔可夫模型（Hidden Markov Model,HMM）以及深度学习网络（Deep Neural Network，DNN）。端到端的方法包含声学模型+声码器以及“完全”端到端方法。


#### 2.基于深度学习的语音合成技术

##### 2.1 语音合成基本知识

<div align=center>
    <img src="zh-cn/img/ch9/01/p2.png" /> 
</div>

语音合成流水线包含 **文本前端（Text Frontend）** 、**声学模型（Acoustic Model）** 和 **声码器（Vocoder）**三个主要模块:

+ 通过文本前端模块将原始文本转换为字符/音素。
+ 通过声学模型将字符/音素转换为声学特征，如线性频谱图、mel 频谱图、LPC 特征等。
+ 通过声码器将声学特征转换为波形。

<div align=center>
    <img src="zh-cn/img/ch9/01/tts的一般流程.png" /> 
</div>

**文本前端(Text Frontend):**

一个文本前端模块主要包含:

+ 分段（Text Segmentation）
+ 文本正则化（Text Normalization, TN）
+ 分词（Word Segmentation, 主要是在中文中）
+ 词性标注（Part-of-Speech, PoS）
+ 韵律预测（Prosody）
+ 字音转换（Grapheme-to-Phoneme，G2P） （Grapheme: 语言书写系统的最小有意义单位; Phoneme: 区分单词的最小语音单位）
    - 多音字（Polyphone）
    - 变调（Tone Sandhi）
        + “一”、“不”变
        + 三声变调
        + 轻声变调
        + 儿化音
        + 方言
+ ...

（输入给声学模型之前，还需要把音素序列转换为 id）

其中最重要的模块是 **文本正则化 模块**和 **字音转换**（TTS 中更常用 G2P 代指） 模块。

各模块输出示例:

```
• Text: 全国一共有112所211高校
• Text Normalization: 全国一共有一百一十二所二一一高校
• Word Segmentation: 全国/一共/有/一百一十二/所/二一一/高校/
• G2P（注意此句中“一”的读音）:
    quan2 guo2 yi2 gong4 you3 yi4 bai3 yi1 shi2 er4 suo3 er4 yao1 yao1 gao1 xiao4
    （可以进一步把声母和韵母分开）
    q uan2 g uo2 y i2 g ong4 y ou3 y i4 b ai3 y i1 sh i2 er4 s uo3 er4 y ao1 y ao1 g ao1 x iao4
    （把音调和声韵母分开）
    q uan g uo y i g ong y ou y i b ai y i sh i er s uo er y ao y ao g ao x iao
    0 2 0 2 0 2 0 4 0 3 ...
• Prosody (prosodic words #1, prosodic phrases #2, intonation phrases #3, sentence #4):
    全国#2一共有#2一百#1一十二所#2二一一#1高校#4
    （分词的结果一般是固定的，但是不同人习惯不同，可能有不同的韵律）
```

文本前端模块的设计需要结合很多专业的语义学知识和经验。人类在读文本的时候可以自然而然地读出正确的发音，但是这些先验知识计算机并不知晓。 例如，对于一个句子的分词：

```
我也想过过过儿过过的生活
我也想/过过/过儿/过过的/生活

货拉拉拉不拉拉布拉多
货拉拉/拉不拉/拉布拉多

南京市长江大桥
南京市长/江大桥
南京市/长江大桥
```

或者是词的变调和儿化音：

```
你要不要和我们一起出去玩？
你要不（2声）要和我们一（4声）起出去玩（儿）？

不好，我要一个人出去。
不（4声）好，我要一（2声）个人出去。

（以下每个词的所有字都是三声的，请你读一读，体会一下在读的时候，是否每个字都被读成了三声？）
纸老虎、虎骨酒、展览馆、岂有此理、手表厂有五种好产品
```

又或是多音字，这类情况通常需要先正确分词：

```
人要行，干一行行一行，一行行行行行;
人要是不行，干一行不行一行，一行不行行行不行。

佟大为妻子产下一女

海水朝朝朝朝朝朝朝落
浮云长长长长长长长消
```

**用深度学习实现文本前端：**

<div align=center>
    <img src="zh-cn/img/ch9/01/p4.png" /> 
</div>

**声学模型（Acoustic Model）：**

声学模型将字符/音素转换为声学特征，如线性频谱图、mel 频谱图、LPC 特征等，声学特征以 “帧” 为单位，一般一帧是 10ms 左右，一个音素一般对应 5~20 帧左右, 声学模型需要解决的是 “不等长序列间的映射问题”，“不等长”是指，同一个人发不同音素的持续时间不同，同一个人在不同时刻说同一句话的语速可能不同，对应各个音素的持续时间不同，不同人说话的特色不同，对应各个音素的持续时间不同。这是一个困难的“一对多”问题。

```
# 卡尔普陪外孙玩滑梯
000001|baker_corpus|sil 20 k 12 a2 4 er2 10 p 12 u3 12 p 9 ei2 9 uai4 15 s 11 uen1 12 uan2 14 h 10 ua2 11 t 15 i1 16 sil 20
```

声学模型主要分为自回归模型和非自回归模型，其中自回归模型在 $t$ 时刻的预测需要依赖 $t-1$ 时刻的输出作为输入，预测时间长，但是音质相对较好，非自回归模型不存在预测上的依赖关系，预测时间快，音质相对较差。

主流声学模型发展的脉络:

+ 自回归模型:
    - Tacotron
    - Tacotron2
    - Transformer TTS
+ 非自回归模型:
    - FastSpeech
    - SpeedySpeech
    - FastPitch
    - FastSpeech2
    - ...

粤语TTS中使用了FastSpeech2作为声学模型：

<div align=center>
    <img src="zh-cn/img/ch9/01/p5.png" /> 
    <p>FastSpeech2 网络结构图</p>
</div>

PaddleSpeech TTS 实现的 FastSpeech2 与论文不同的地方在于，我们使用的是 phone(音素) 级别的 pitch 和 energy(与 FastPitch 类似)，这样的合成结果可以更加稳定。

<div align=center>
    <img src="zh-cn/img/ch9/01/p6.png" /> 
    <p>FastPitch 网络结构图</p>
</div>

关于FastSpeech2的介绍可以参考:[FastSpeech V2](zh-cn/03_Text_to_spectrogram?id=_9-fastspeech-2-fast-and-high-quality-end-to-end-text-to-speech)

paddlespeech更多模型介绍参考： <https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/tts/models_introduction.md>

**声码器（Vocoder）:**

声码器将声学特征转换为波形。声码器需要解决的是 “信息缺失的补全问题”。信息缺失是指，在音频波形转换为频谱图的时候，存在相位信息的缺失，在频谱图转换为 mel 频谱图的时候，存在频域压缩导致的信息缺失；假设音频的采样率是16kHZ, 一帧的音频有 10ms，也就是说，1s 的音频有 16000 个采样点，而 1s 中包含 100 帧，每一帧有 160 个采样点，声码器的作用就是将一个频谱帧变成音频波形的 160 个采样点，所以声码器中一般会包含上采样模块。

与声学模型类似，声码器也分为自回归模型和非自回归模型, 更细致的分类如下:

+ Autoregression
    - WaveNet
    - WaveRNN
    - LPCNet
+ Flow
    - WaveFlow
    - WaveGlow
    - FloWaveNet
    - Parallel WaveNet
+ GAN
    - WaveGAN
    - Parallel WaveGAN
    - MelGAN
    - Style MelGAN
    - Multi Band MelGAN
    - HiFi GAN
+ VAE
    - Wave-VAE
+ Diffusion
    - WaveGrad
    - DiffWave

PaddleSpeech TTS 主要实现了百度的 WaveFlow 和一些主流的 GAN Vocoder, 在粤语TTS中，我们使用 HiFi-GAN 作为声码器，这也是目前主流的商用的声码器。

<div align=center>
    <img src="zh-cn/img/ch9/01/p7.png" width=70%/> 
    <p>Parallel WaveGAN 网络结构图</p>
</div>

关于Parallel WaveGAN的详细介绍参考：[Parallel WaveGAN](zh-cn/07_Vocoders?id=_3-parallel-wavegana-fast-waveform-generation-model-based-on-generative-adversarial-networks-with-multi-resolution-spectrogram)

各 GAN Vocoder 的生成器和判别器的 Loss 的区别如下表格所示:

<div align=center>
    <img src="zh-cn/img/ch9/01/p8.png" /> 
</div>

#### 3.如何训练粤语TTS系统

PaddleSpeech r1.4.0 版本还提供了全流程粤语语音合成解决方案，包括语音合成前端、声学模型、声码器、动态图转静态图、推理部署全流程工具链。语音合成前端负责将文本转换为音素，实现粤语语言的自然合成。为实现这一目标，声学模型采用了基于深度学习的端到端模型 FastSpeech2 ，声码器则使用基于对抗神经网络的 HiFiGAN 模型。这两个模型都支持动转静，可以将动态图模型转化为静态图模型，从而在不损失精度的情况下，提高运行速度

**0.环境搭建**

```
pip install paddle-gpu
git clone https://gitee.com/paddlepaddle/PaddleSpeech
cd PaddleSpeech
git install . --user
```


**1.构建数据集**

数据集下载地址：

+ https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-daily-use-sentence/
+ https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-in-the-vehicle/

下载后的元数据结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch9/01/p9.png" /> 
</div>


**2.训练MFA模型**

MFA只在训练的时候用到，推理的时候只需要通过预测的duration做对齐就可以了。

+ Kaldi MFA： https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
+ Paddle调用MFA生成对齐的数据：https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa

训练数据的对齐数据如下所示：



<div align=center>
    <img src="zh-cn/img/ch9/01/p10.png" /> 
</div>

有了这些数据我们局可以对这些数据进行处理训练声学模型，在这里我们仅训练了声学模型，声码器使用了在CSMSC 数据集（中文标准女声音库）上训练的HiFi-GAN,而没有对该模型进行重新训练或微调。


+ textgrid: https://textgrid.org/en/download
+ praat: https://www.fon.hum.uva.nl/praat/

使用praat工具打开如下图所示：

<div align=center>
    <img src="zh-cn/img/ch9/01/p13.png" /> 
</div>


**3.数据预处理**

+ 由MFA得到的textgrid生成音素级别的duration

```shell
echo "Generate durations.txt from MFA results ..."
python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
    --inputdir=./canton_alignment \
    --output durations.txt \
    --config=${config_path}
```

+ 音频特征提取

详细的参考`\paddlespeech\t2s\exps\fastspeech2\preprocess.py`

```shell
echo "Extract features ..."
python3 ${BIN_DIR}/preprocess.py \
    --dataset=canton \
    --rootdir=~/datasets/canton_all \
    --dumpdir=dump \
    --dur-file=durations.txt \
    --config=${config_path} \
    --num-cpu=20 \
    --cut-sil=True
```

1. 划分训练集，开发集，测试集
2. 提取音频的mel谱，Pitch, Energy
3. 得到每个音频的如下特征：

```json
 record = {
        "utt_id": utt_id,
        "phones": phones,
        "text_lengths": len(phones),
        "speech_lengths": num_frames,
        "durations": durations,
        "speech": str(mel_path),
        "pitch": str(f0_path),
        "energy": str(energy_path),
        "speaker": speaker
    }

```

+ 特征标准化

详细的参考`\paddlespeech\t2s\exps\fastspeech2\normalize.py`

主要是将mel谱，pitch, energy进行标准化，使用了sklearn中的`StandardScaler`方法实现。

最终处理的用来训练FastSpeech2的数据结构为：

```shell
dump
├── dev
│   ├── norm
│   └── raw
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```

这里的`raw`文件夹中包含了每个样本的speech,pitch,energy的特征，`norm`文件夹存放了标准化的版本， `*_stats.npy`统计了训练集中的对应特征的均值和方差等统计信息，用来归一化`dev`和`test`数据。

**4.声学模型训练**

section 3中的数据处理仅仅是将数据处理为训练模型需要的标注数据格式并不涉及TTS的前端处理，paddlespeech也帮我们实现了一些TTS前端的方法，详细的可以参考：`paddlespeech\t2s\frontend`,这个过程在我们构建MFA时已经完成。

有了训练标注数据，我们就可以训练粤语的TTS声学模型FastSpeech2了！

参考：`\paddlespeech\t2s\exps\fastspeech2\train.py`

```shell
python3 ${BIN_DIR}/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=2 \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt

```

这样就可以正常训练声学模型FastSpeech2了！

!> 注意：这里的实现与原始的FastSpeech2不同的地方时增加了支持Multi-speacker的speaker embedding!

!> 训练好的模型可以转onnx参考： https://www.paddlepaddle.org.cn/documentation/docs/zh/2.5/guides/advanced/model_to_onnx_cn.html

#### 4.粤语TTS系统的流式推断

这里我们做短文本的推断，暂时不介绍流式的TTS推断服务的搭建。我们实现的过程如下图流程图所示，这也是一个常用的完整的TTS服务需要的过程！

我们实现了如下的推理流程：

<div align=center>
    <img src="zh-cn/img/ch9/01/p11.png" /> 
</div>


#### 5.Demo Time


<div align=center>
    <img src="zh-cn/img/ch9/01/p12.png" /> 
</div>


其效果如下：

<div align=center>
    <audio id="audio" controls="" preload="none" >
          <source id="wav" src="zh-cn/img/ch9/01/fa1cc3d2_afde_11ee_8f0e_00d861c69d42.wav">
    </audio>
</div>


------

<!-- elmo 训练TTS, bert-vits,  Real-Time-Voice-Cloning-->