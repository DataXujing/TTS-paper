## Text-to-spectrogram

<!-- # RNN -->

### 1. Tacotron

!> https://arxiv.org/abs/1703.10135

<!-- https://google.github.io/tacotron/ -->
<!-- https://blog.csdn.net/qq_37236745/article/details/108846377 -->
<!-- https://zhuanlan.zhihu.com/p/337042442 -->

Tacotron是谷歌于2017年提出的端到端语音合成系统，该模型可接收字符的输入，输出相应的原始频谱图，然后将其提供给 Griffin-Lim 重建算法直接生成语音。

#### Abstract

一个文本转语音的合成系统通常包含多阶段处理，例如文本分析前端，声学模型和声音合成模块。构建这些组件常常需要大量的领域专业知识，而且设计选择也可能很脆弱。在这篇论文里，我们提出了Tacotron，一种端到端的生成式文本转语音模型，可以直接从字符合成语音。在`<文本,声音>`配对数据集上，该模型可以完全从随机初始化从头开始训练。我们提出了几个可以使seq2seq框架在这个高难度任务上表现良好的关键技术。Tacotron 在美式英语测试里的平均主观意见评分达到了3.82分（总分是5分），在合成自然度方面优于已在生产中应用的参数模型。另外，由于Tacotron是在帧层面上生成语音，所以它大幅度快于样本级自回归方式的模型。

#### 1.Introduction

现代文本转语音（TTS）的流水线比较复杂(Taylor,2009)，举例来说，通常基于统计参数的TTS系统包含：一个用来抽出多种语言特征的文本分析前端，一个音长模型（语音持续时间模型），一个声学特征预测模型，和一个复杂的基于信号处理的声码器(Zen et al.,2009; Agiomyrgiannakis,2015)。这些组件都基于大量的领域专业知识因而其设计很艰难。并且，这些组件都是单独训练的，所以产生自每个组件的错误会有叠加效应。因此，现代TTS系统的设计复杂度导致在构建新系统时需要投入大量的工程努力。

正因如此，集成一个能在少量人工标注的`<文本，语音>`配对数据集上训练的端到端的TTS系统，会带来诸多优势。首先，这样一个系统减少了艰难特征工程的必要，而正是这些特征工程可能会导致启发式错误和脆弱的设计选择。其次，这样的系统允许基于各种属性来进行多样化的调节，比如不同说话人，不同语言，或者像语义这样的高层特征，这是因为调节不只是出现在特定几个组件中，而是在模型的最开始就发生了。类似的，拟合新数据也将变得更容易。最后，相比会出现错误叠加效应的多阶段模型，单一模型倾向于更鲁棒。这些优势意味着，一个端到端的模型能够允许我们在现实世界容易获取的大量的丰富生动的同时也很嘈杂的数据上执行训练。

TTS是一个大规模的逆问题：把一份高度压缩的文本源解压缩成语音。由于同一份文本可以对应到不同的发音或讲话风格，对一个端到端的模型来说，这是一个异常困难的学习任务：给定一个输入它必须处理在信号层面的大量变化。而且，不像端到端的语音识别(Chan et al.,2016)或者机器翻译(Wu et al.,2016)，TTS的输出是连续的，并且输出序列通常比输入序列长很多。这些属性造成了预测错误的快速累积。在这篇论文中，我们提出了Tacotron，一个端到端的基于带注意力范式(Bahdanau et al.,2014)的序列到序列（seq2seq）(Sutskever et al.,2014)的生成式TTS模型。该模型使用几个可以改善普通seq2seq模型能力的技术，输入字符直接输出原始声谱图。给定`<文本，语音>`配对数据，Tacotron可以完全从随机初始化从头开始训练。由于不需要音素层面的对齐，因此它可以很容易的使用大量带有转录文本的声学数据。使用一个简单的波形合成技术，Tacotron在美式英语评估数据集上得到了3.82的平均意见得分（MOS），在合成自然度方面优于已在生产中应用的参数模型（语音样本展示参照：<https://google.github.io/tacotron>）

#### 2.Related Work

WaveNet(van den Oord et al.,2016)是一个强大的声音生成模型。它在TTS中表现良好，但是样本水平自回归的天性导致其速度慢。它还要求在既存TTS前端生成的语言特性上进行调节，因此不是端到端的，它只替换了声码器和声学模型部分。另外一个最近开发的神经模型是DeepVoice (Arik et al.,2017)，它分别用一个神经网络替换了典型TTS系统流水线中的每一个组件，然而它的每个组件都是单独训练的，要把系统改成端到端的方式不那么简单。

据我们所知，Wang et al. (2016)是最早使用带有注意力的seq2seq方法尝试端到端TTS系统的。但是，首先它需要一个预先训练好的隐马尔可夫（HMM）对齐器来帮助seq2seq模型学习如何对齐。所以很难说seq2seq本身学到了多少对齐能力。其次，为了训练模型使用了几个技巧，作者指出这些技巧有损于合成韵律。第三，它预测了声码器参数作为中间特征表达，因此需要一个声码器。最后，该模型训练的输入是音素数据并且实验结果好像有点受限。

Char2Wav (Sotelo et al.,2017)是一个独立开发的可以在字符数据上训练的端到端模型。但是，Char2Wav在使用SampleRNN神经声码器(Mehri et al.,2016)前也预测了声码器参数，而Tacotron直接预测原始声谱图。另外，seq2seq和SampleRNN需要单独进行预训练，但我们的模型可以从头开始训练。最后，我们对普通seq2seq进行了几个关键变更，我们后面会展示，普通的seq2seq模型对字符输入不太奏效。

#### 3.Model Architecture

<div align=center>
    <img src="zh-cn/img/ch3/01/p1.png" /> 
</div>

Tacotron的骨干部分是一个有注意力机制的(Bahdanau et al.,2014; Vinyals et al.,2015)seq2seq模型。上图为该模型架构，它包含一个编码器，一个基于注意力机制的解码器和一个后处理网络。从high-level看，我们的模型把字符作为输入，产生的声谱帧数据随后会被转换成波形。下面详细描述这些组件。

**CBHG Module**

<div align=center>
    <img src="zh-cn/img/ch3/01/p2.png" /> 
</div>

我们首先描述CBHG的模块,如上图所示。CBHG包含一个一维卷积滤波器组，后跟一个Highway网络(Srivastava et al.,2015)和一个双向门控循环单元（GRU）(Chung et al.,2014)循环神经网络（RNN）。CBHG是一个强大的模块以提取序列的特征表达。首先在输入序列上用$K$组一维卷积核进行卷积，这里第$k$组包含$C_k$个宽度是$k$（这里$k=1,2,...,K$）的卷积核。这些卷积核显式地对局部上下文信息进行建模（类似于unigrams, bigrams, 直到K-grams）。卷积输出结果被堆叠在一起然后再沿时间做最大池化以增加局部不变性，注意我们令`stride=1`以维持时间方向的原始分辨率。然后把得到的结果序列传给几个定长一维卷积，其输出结果通过残差连接(He et al.,2016)和原始输入序列相叠加。所有卷积层都使用了批标准化(Ioffe & Szegedy,2015)。卷积输出被送入一个多层高速公路网络来提取高层特征。最上层我们堆叠一个双向GRU RNN用来前后双向提取序列特征。CBHG是受机器翻译(Lee et al.,2016)论文的启发，我们与(Lee et al.,2016)的不同包括使用非因果卷积，批标准化，残差连接以及`stride=1`的最大池化处理。我们发现这些修改提高了模型的泛化能力。

**Encoder**

编码器的目的，是提取文本的鲁棒序列表达。编码器的输入是字符序列，输入的每个字符都是个一个one-hot向量并被嵌入一个连续向量中。然后对每个字符向量施加一组非线性变换，统称为“pre-net”。在这次工作中，我们使用带dropout的瓶颈层（bottleneck layer）作为pre-net以帮助收敛并提高泛化能力。CBHG模块将pre-net的输出变换成编码器的最终表达，并传给后续的注意力模块。我们发现基于CBHG的编码器不仅减少了过拟合，它还能比标准的多层RNN编码器产生更少的错音（参看链接网页上的合成语音样本）

**Decoder**

我们使用基于内容的tanh注意力解码器（参照Vinyals et al. (2015)），在这个解码器中，一个有状态的循环层在每个时间步骤上都产生一次注意点查询。我们把上下文向量和Attention RNN单元的输出拼接在一起，作为解码器RNN的输入。同时我们还使用了带有纵向残差连接的GRUs堆栈(Wu et al., 2016)，它能加速收敛。选择什么作为解码器的目标输出非常重要。因为我们可以直接预测原始声谱图，这对于学习语音信号和原始文本对齐的目标（这是在这个任务上使用seq2seq的真正动机）是一个高度冗余的表示。因为这个冗余，我们为seq2seq解码和波形合成选择了一个不同的目标。语音合成作为一个可训练或者可确定的逆向过程，只要能够提供足够的语音可理解性和足够的韵律信息，seq2seq的目标输出就可以被大幅度压缩。尽管类似倒谱这样更小的带宽或者更简洁的目标输出也可行，但我们采用带宽为80的梅尔刻度声谱图作为解码器的目标输出。我们使用了一个后处理网络（接下来讨论）把seq2seq的目标输出转化为波形。

我们使用一个简单的全连接输出层来预测解码器的目标输出。我们发现一个重要的技巧是，每一步解码处理可以同时预测多个非重叠的输出帧，一次预测$r$帧使得全体解码步骤缩小了$r$倍，结果是减小了模型大小，训练时间和推断时间。更重要的，我们发现这个技巧会大幅度加快收敛速度，试验中注意力模块非常迅速（且非常稳定）的学到了如何对齐。这可能是因为每个字符通常对应了多个语音帧而相邻的语音帧具有相关性。强制每次输出一帧使得模型对同一个输入字符进行多次重复关注，而同时输出多帧允许注意力在训练中更早向前移动。Zen et al. (2016)也使用了类似的技巧，但目的主要是用来加速推断。

解码器的第一步是在一个“全零帧”上开始调节，模型结构图中标示的`"<GO>"frame`。在推断时，解码器的第$t$步处理，预测结果的最后一帧被作为解码器第$t+1$步的输入。注意这里选择最后一帧输入到下一步处理中只是一种选择而已，也可以选择一组$r$帧的全部作为下一步的输入。在训练中，我们取每个第$r$帧输入给解码器。像编码器中的处理一样，输入帧传给一个pre-net。因为没有使用scheduled sampling（Bengio et al.,2015）（我们发现这样做会损害声音质量）那样的技术，所以pre-net中的dropout对模型泛化很关键，因为dropout为解决输出分布中的多形态问题提供了噪声源。

**Post-Processing Net and Waveform System**

上面也提到了，后处理网络的任务是，把seq2seq的输出转化成可以被合成为波形的目标表达。因为使用Griffin-Lim做合成器，后处理网络要学习的是如何预测在线性频率刻度上采样的频谱幅度。构建后处理网络的另外一个动机是它可以看到全体解码结果序列，对比普通seq2seq总是从左到右运行，它可以获得前后双向信息用以纠正单帧预测错误。在这次工作中，我们使用CBHG模块作为后处理网络，尽管一个更简单的架构可能也会工作的很好。后处理网络的概念是高度通用的，它可以用来预测不同的目标输出如声码器参数，也可以作为像WaveNet那样的神经声码器(van den Oord et al.,2016; Mehri et al.,2016; Arik et al.,2017)来直接合成波形样本。

我们使用Griffin-Lim算法(Griffin & Lim,1984)从预测出的声谱图合成波形。我们发现把预测频谱的振幅提高1.2倍再输入到Griffin-Lim可以减少人工痕迹，可能是归功于其谐波增强效果。我们观察到Griffin-Lim在50次迭代后收敛（实际上大约30次迭代好像就足够了），这个速度相当快。我们在Tensorflow中实现了Griffin-Lim算法，所以它也成为了整个模型的一部分。尽管Griffin-Lim是可导的（它没有训练参数），但我们没有在其上设计任何损失。我们强调选择Griffin-Lim是为了简单，尽管它已经生成了很好的结果，我们也在开发一个快速的高品质的可训练的声谱-波形转换器。


#### 4.Model Details

<div align=center>
    <img src="zh-cn/img/ch3/01/p3.png" /> 
</div>

上表列出了超参数和模型架构。我们使用对数幅度谱，汉明窗，帧长50毫秒，帧移12.5毫秒，2048点傅里叶变换，我们还发现预加重(0.97)也有用。对所有的试验我们使用24k赫兹采样率。

在论文的MOS评分中使用`r=2`（解码器输出层的缩小因子），更大的$r$也运行的很好（例如`r=5`）。我们使用Adam优化器(Kingma & Ba,2015)，学习率从0.001开始500K步后降低到0.0005，1M步后降到0.0003，2M步后降到0.0001。我们采用简单的L1损失同时囊括seq2seq解码器（梅尔刻度声谱图）和后处理网络（线性刻度声谱图）。两部分损失的权重相同。

训练的批大小设定为32，所有的序列都补齐到最大长度。在训练中使用损失屏蔽是一个常用做法，在补零的数据帧上屏蔽损失。然而我们发现这样训练出的模型不知道何时停止输出，导致靠近结尾会有重复的声音。解决这个问题的一个简单技巧是对补零的数据帧也进行波形重建。

#### 5.Experiments

我们在一个内部北美英语数据集上训练Tacotron，这个数据集包含大约24.6小时的语音数据，由一个专业女性播讲。所有单词都进行了标准化处理，如"16"被转换成"sixteen"

<div align=center>
    <img src="zh-cn/img/ch3/01/p4.png" /> 
</div>

为理解模型的关键组件我们实施了几个剥离研究。对于生成式模型，基于客观度量的模型比较是很困难的，这些客观度量不能与感知很好地匹配(Theis et al.,2015)。相反的我们主要依赖视觉比较。我们强烈推荐读者听一下我们提供的语音样本。

首先，与普通的seq2seq模型比较。编码器和解码器都是用2层残差RNN，每层包含256个GRU单元（我们也尝试了LSTM，结果类似），不使用pre-net和后处理网络，解码器直接预测线性对数幅度声谱图。我们发现，预排程采样（采样率0.5）对于这个模型学习对齐和泛化是必要的。我们在上图中展示了学到的注意力对齐，上图(a)揭示了普通的seq2seq学到的对齐能力很微弱，一个问题是其中有一段注意力卡住了，这导致了语音合成结果的可理解度很差，语音自然度和整体音长也都被摧毁了。相对的，我们的模型学到了清晰平滑的对齐，上图(c)所示。

其次，我们比较了用一个2层残差GRU编码器替换CBHG后的模型，包括编码器的pre-net在内模型的其余部分不变。比较上图(b)和上图(c)可以看到，GRU编码器的对齐有些噪声。试听语音合成结果，我们发现这些对齐噪声会导致发音错误。CBHG编码器减少了过拟合并且在长的复杂短语上泛化能力很好。


<div align=center>
    <img src="zh-cn/img/ch3/01/p5.png" /> 
</div>

上图(a)和(b)展示了使用后处理网络的好处。我们训练了一个除了不包含后处理网络其余部分都一样（解码RNN修改成预测线性声谱图）的模型，拥有了更多的上下文信息，后处理网络的预测结果包含更好的谐波（比如在100~400之间的高频谐波）和高频共振峰结构，这会减少合成的人工痕迹。

<div align=center>
    <img src="zh-cn/img/ch3/01/p6.png" /> 
</div>

我们做了平均意见得分（MOS）测试，由测试者对合成语音的自然程度进行 5 分制的李克特量表法（Likert scale score）评分。MOS 的测试者均为参与众包的母语人群，共使用 100 个事先未展示的短语，每个短语获得 8 次评分。当计算MOS评分时，只有佩戴耳机时打出的评分被计算在内。我们对Tacotron与参数式（parametric）系统（基于LSTM（Zen et al.,2016））和拼接式（concatenative）系统（Gonzalvo et al.,2016）做了对比，后两者目前均已投入商业应用。测试结果如下表显示：Tacotron 的 MOS 分数为 3.82，优于参数系统。由于参照基准已经非常强大，以及由Griffin-Lim 引起的人工痕迹，这一新方法具有非常好的前景。

#### 6.Discussions

我们提出了Tacotron，一个集成的端到端的生成式TTS模型，它以字符序列作为输入，输出对应的声谱图。后接一个简单的波形合成模块，模型在美式英语上的MOS得分达到了3.82，在自然度方面超越了已经投入生产的参数式系统。Tacotron是基于帧数据的，因此推断要大大快于样本水平的自回归方法。Tacotron不像之前的研究工作那样需要人工工程的语言特征或者像HMM对齐器这样复杂的组件，它可以从随机初始化开始从头进行训练，只是进行了简单的文本标准化处理，但是最近在文本标准化学习的进步(Sproat & Jaitly,2016)表明这一步处理未来也可以去掉。

我们的模型的很多方面还有待调查，很多早期的设计决定一直保持原样。输出层，注意力模块，损失函数，以及Griffin-Lim波形合成器的改善时机都已经成熟。例如，Griffin-Lim的输出听起来含有人工合成的痕迹已经广为人知，我们现在正在开发一个快速的高品质的基于神经网络的声谱图逆变换网络。

------

### 2. Tacotron2

!> https://arxiv.org/abs/1712.05884

!> NVIDIA/Tacotron2：https://github.com/NVIDIA/tacotron2/

<!-- https://github.com/JasonWei512/Tacotron-2-Chinese -->

<!-- https://www.bilibili.com/video/BV1uh411m7Y2/?spm_id_from=333.337.search-card.all.click&vd_source=def8c63d9c5f9bf987870bf827bfcb3d
 -->

 <!-- NVIDIA/Tacotron2：https://github.com/NVIDIA/tacotron2/ -->

 <!-- https://wmathor.com/index.php/archives/1478/ -->

 <!-- https://zhuanlan.zhihu.com/p/103521105 -->
 <!-- https://www.bilibili.com/video/BV1tb4y1y7H9/?spm_id_from=333.788&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

 <!-- https://blog.csdn.net/qq_37236745/article/details/108846686 -->

#### 1.概述

Tacotron2是由Google Brain在2017年提出来的一个End-to-End语音合成框架。模型从下到上可以看作由两部分组成：

+ 声谱预测网络：一个Encoder-Attention-Decoder网络，用于将输入的字符序列预测为梅尔频谱的帧序列
+ 声码器（vocoder）：一个WaveNet的修订版，用于将预测的梅尔频谱帧序列产生时域波形

<div align=center>
    <img src="zh-cn/img/ch3/02/p3.png" /> 
</div>

详细的结构如下图：

<div align=center>
    <img src="zh-cn/img/ch3/02/p1.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch3/02/p2.png" /> 
</div>

#### 2.编码器

<div align=center>
    <img src="zh-cn/img/ch3/02/p4.png" /> 
</div>

Encoder的输入是多个句子，每个句子的基本单位是character，例如

+ 英文`"hello world"`就会被拆成`"h e l l o w o r l d"`作为输入
+ 中文`"你好世界"`则会先把拼音标识出来得到`"ni hao shi jie"`，然后进一步按照声韵母的方式来分割成`"n i h ao sh i j ie"`，或者直接按照类似英文的方式分割成`"n i h a o s h i j i e"`

Encoder的具体流程为：

1. 输入的数据维度为`[batch_size, char_seq_length]`
2. 使用512维的Character Embedding，把每个character映射为512维的向量，输出维度为`[batch_size, char_seq_length, 512]`
3. 3个一维卷积，每个卷积包括512个kernel，每个kernel的大小是`5*1`（即每次看5个characters）。每做完一次卷积，进行一次BatchNorm、ReLU以及Dropout。输出维度为`[batch_size, char_seq_length, 512]`（为了保证每次卷积的维度不变，因此使用了pad）
4. 上面得到的输出，扔给一个单层BiLSTM，隐藏层维度是256，由于这是双向的LSTM，因此最终输出维度是`[batch_size, char_seq_length, 512]`

```python
class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,x
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

```

```python
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
                                    
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
```

#### 3.注意力机制

<div align=center>
    <img src="zh-cn/img/ch3/02/p5.png" width=70% /> 
</div>

上图描述了第一次做attention时的输入和输出。其中，$y_0$是PreNet初始输入`<S>`的编码表示，$c_0$是当前的"注意力上下文"。初始第一步时，$y_0$和$c_0$都被初始化为全0向量，然后将$y_0$和$c_0$拼接起来，得到一个768维的向量$y_{0,c}$，将该向量与attention_hidden和attention_cell一起作为LSTMcell的输入（attention_hidden其实就是LSTMcell的hidden_state，attention_cell其实就是LSTMcell的cell_state）。得到的结果是$h_1$和attention_cell，这里没有给attention_cell单独起名字，主要考虑其是"打酱油"的，因为除了attention_rnn之外，其它地方没有用到attention_cell.

<div align=center>
    <img src="zh-cn/img/ch3/02/p6.png" width=70% /> 
</div>

Attention_Layer一共接受五个输入：

1. $h_1$是和mel谱相关的变量
2. $m$来自source character sequence编码的"记忆"
3. $m^{'}$是$m$通过一个Linear后得到的
4. attention_weights_cat是将历史（上一时刻）的attention_weights和attention_weights_cum拼接得到的
5. mask全false，基本没用

计算细节如下：

<div align=center>
    <img src="zh-cn/img/ch3/02/p7.png" width=70% /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch3/02/p8.png" /> 
</div>

get_alignment_energies函数图示如下：

<div align=center>
    <img src="zh-cn/img/ch3/02/p9.png" width=70% /> 
</div>

图中Location_Layer的代码如下：

```python
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, # 32, 31
                 attention_dim): # 128
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2) # padding=15
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat): # [1, 2, 151]
        processed_attention = self.location_conv(attention_weights_cat) # [1, 32, 151]
        processed_attention = processed_attention.transpose(1, 2) # [1, 151, 32]
        processed_attention = self.location_dense(processed_attention) # [1, 151, 128]
        return processed_attention

```

Attention的代码如下：

```python
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
```


#### 4.解码器

<div align=center>
    <img src="zh-cn/img/ch3/02/p10.png" width=50%/> 
</div>

解码器是一个自回归结构，它从编码的输入序列预测出声谱图，一次预测r帧

1. 上一步预测出的频谱首先传入一个PreNet，它包含两层神经网络，PreNet作为一个信息瓶颈层（bottleneck），对于学习注意力是必要的
2. PreNet的输出和Attention Context向量拼接在一起，传给一个含有1024个单元的两层LSTM。LSTM的输出再次和Attention Context向量拼接在一起，然后经过一个线性投影来预测目标频谱
3. 最后，目标频谱帧经过一个5层卷积的PostNet（后处理网络），再将该输出和Linear Projection的输出相加（残差连接）作为最终的输出
4. 另一边，LSTM的输出和Attention Context向量拼接在一起，投影成标量后传给sigmoid激活函数，来预测输出序列是否已完成预测

PreNet层的图示及代码如下所示：

<div align=center >
    <img src="zh-cn/img/ch3/02/p11.png" width=30%/> 
</div>

```python
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
```

PostNet层的图示及代码如下所示：

<div align=center>
    <img src="zh-cn/img/ch3/02/p12.png" width=50%/> 
</div>

```python
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x
```

从下面Decoder初始化部分可以看出Decoder由prenet，attention_rnn，attention_layer，decoder_rnn，linear_projection，gate_layer组成

```python
class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')
```

#### 5.总结

Tacotron2模型的完整网络结构：

```python
Tacotron2(
  (embedding): Embedding(148, 512)
  (encoder): Encoder(
    (convolutions): ModuleList(
      (0): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (lstm): LSTM(512, 256, batch_first=True, bidirectional=True)
  )
  (decoder): Decoder(
    (prenet): Prenet(
      (layers): ModuleList(
        (0): LinearNorm(
          (linear_layer): Linear(in_features=80, out_features=256, bias=False)
        )
        (1): LinearNorm(
          (linear_layer): Linear(in_features=256, out_features=256, bias=False)
        )
      )
    )
    (attention_rnn): LSTMCell(768, 1024)
    (attention_layer): Attention(
      (query_layer): LinearNorm(
        (linear_layer): Linear(in_features=1024, out_features=128, bias=False)
      )
      (memory_layer): LinearNorm(
        (linear_layer): Linear(in_features=512, out_features=128, bias=False)
      )
      (v): LinearNorm(
        (linear_layer): Linear(in_features=128, out_features=1, bias=False)
      )
      (location_layer): LocationLayer(
        (location_conv): ConvNorm(
          (conv): Conv1d(2, 32, kernel_size=(31,), stride=(1,), padding=(15,), bias=False)
        )
        (location_dense): LinearNorm(
          (linear_layer): Linear(in_features=32, out_features=128, bias=False)
        )
      )
    )
    (decoder_rnn): LSTMCell(1536, 1024, bias=1)
    (linear_projection): LinearNorm(
      (linear_layer): Linear(in_features=1536, out_features=80, bias=True)
    )
    (gate_layer): LinearNorm(
      (linear_layer): Linear(in_features=1536, out_features=1, bias=True)
    )
  )
  (postnet): Postnet(
    (convolutions): ModuleList(
      (0): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(80, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (4): Sequential(
        (0): ConvNorm(
          (conv): Conv1d(512, 80, kernel_size=(5,), stride=(1,), padding=(2,))
        )
        (1): BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
)
```

------

<!-- #SPP -->
### 3. Neural HMM TTS: Neural HMMS are All You Need

!> https://arxiv.org/abs/2108.13320

!> https://github.com/shivammehta25/Neural-HMM

!> https://shivammehta25.github.io/Neural-HMM/

#### Abstract

基于seq2seq的神经网络的TTS模型已经表现比基于统计的（SPSS）的HMM模型要好很多了。但是基于神经网络的模型不是概率模型，使用了非单调的Attention对齐。这种Attention的方式导致训练时间正常并且有可能出现合成出来的语音是语无伦次胡言乱语的。本论文结合了传统的基于HMM的概率模型和神经网络模型解决这些问题，在神经网络中通过一个从左向右的自回归是的HMM代替Attention。我们调整了Tacotron2得到一个HMM-based Neural TTS Model使用单调的对齐方式，使用全序列的对数似然函数训练模型而不是近似方法，在较少的训练数据和训练步数上由于Tacotron2,并且在不需要PostNet也可以生成有竞争力自然度的语音，并且可以容易的调整语速。

#### 1.Introduction

过去十年TTS技术进步巨大，随着该领域的发展，输出语音质量出现了许多阶跃变化。统计参数语音合成
（SPSS）基于隐马尔可夫模型（HMM），现在已经
很大程度上被神经TTS所取代。与基于信号处理的声码器相比，波形级深度学习极大地提高了分段质量，而具有注意力的序列到序列模型，例如Tacotron，显示出极大地改善了韵律。结合起来，如Tacotron 2所述，这些创新产生了合成语音，其自然度有时可以与录音语音相媲美。

然而，并不是TTS系统的所有方面都有所改进。将具有位置特征的深度学习集成到基于HMM的TTS中提高了自然度，但牺牲了同时学习说话和对齐的能力，而需要外部强制对齐器。基于注意力的神经TTS系统重新引入了学习对齐的能力，但不以概率为基础，需要更多的数据和时间才能开始说话。此外，它们的非单调注意力机制并不能强制执行语音的一致顺序。因此，合成容易出现跳跃和结结巴巴的伪影，并可能灾难性地崩溃，导致难以理解的胡言乱语。

在这篇文章中，我们1）提出了基于HMM的和神经网络的
TTS方法可以结合起来，以获得两个领域的优点。我们2）通过描述神经TTS架构来支持这一说法
基于Tacotron 2，但注意力机制被HMM状态取代，以获得持续时间和声学的完全概率联合模型。模型开发利用了基于HMM和seq2seq的TTS的设计原则。实验表明。该模型在1k次的模型更新训练后产生了可理解的语音，语音质量类似于Tacotron2,但是训练速度提高了15倍，与标准的Tacotron 2不同，它还允许控制语速。代码和样例语音参考：<https://shivammehta25.github.io/Neural-HMM/>


#### 2.Background

这项工作的起点是《Where do the improvements come from in sequence-tosequence neural TTS》，反应基于HMM的SPSS方法与Seq2Seq的Attention方法的四个关键点是：

1. Neural vocoder with mel-spectrogram inputs
2. Learned front-end (the encoder)
3. Acoustic feedback (autroregression)
4. Attention instead of HMM-based alignment

其中，第1-3项提高了语音质量，而Attention有时会使输出明显变差。本文将1-3纳入TTS系统，该系统利用神经HMM，而不是关注seq2seq的建模。下文第2.1节描述了如何在先前工作的基础上将1-3添加到HMM中，并在第2.2节中讨论了注意事项（4）。

##### 2.1 Adding neural TTS aspects to HMM-based TTS

关于1，现在大多数的Vocoder依然采用频谱作为输入，因此本方法依然采用该技术；改进韵律的另一个因素是第2项
前端（即编码器）。再说一遍，没有什么可以阻止的在利用HMM的系统中使用这一思想。基于HMM我们介绍的系统都使用与Tacotron2相同的编码器架构
，没有添加额外的语言特征。对于3，在本文中，我们描述了HMM，它与Tacotron一样，使用由神经网络定义的更强的非线性AR模型。

##### 2.2 Attention in TTS

在一个典型的基于序列到序列的TTS系统中，Attention机制负责持续时间建模和学习
在训练期间将输入符号与输出帧对齐。但是一些研究表明Attention机制在TTS中并不是均有效的。一些研究表明，好的注意力机制必须满足：local(output frame要对齐唯一的input symbol);monotonic; complete(not skip any speech sounds).大多数的基于神经网咯的TTS的Attention机制不满足这3点。

#### 3.Method

3.1节描述了如何将Tacotron2魔改为Neural HMM。 3.2节描述了Neural HMM TTS中的一些Trick。

<div align=center>
    <img src="zh-cn/img/ch3/03/p1.png" /> 
</div>

##### 3.1 Replacing attention with neural HMMs

Tactron2中的核心技术：location-sensitive attention，使用了之前生成的acoustic frames: $x_{1:t-1}$,用以选择出encoder output中的哪个$h_n$用来发送给decoder产生下一帧$x_t$. 注意力也有一种内部状态，以先前注意力权重$\alpha_{1:t-1,n}$的形式存在。 上图中(a)展示了Tacotron2如何生成第$t$帧：

<div align=center>
    <img src="zh-cn/img/ch3/03/p2.png" width=40%/> 
</div>

这里的$a_{t-1}$表示第一个decoder LSTM层的hidden和cell state，OutputNet表示decoder的上半部分（包含了第二层LSTM),$\tau_t \in [0,1]$表示stop token.

Neural HMM模型中，移除了上述公式（1）中的$g_{t-1}$,替换了attention,通过一个概率OutputNet使用$a_t$和HMM state $s_t \in \\{1,...N\\}$来估计第$t$帧$x_t$,输出HMM的输出分布$o(\theta)$的参数$\theta_t$。 stop token变成了转移概率$\tau_t \in [0,1]$对应$s_t,s_1=1$,上式中的（2）-（4）变为：

<div align=center>
    <img src="zh-cn/img/ch3/03/p3.png" width=40%/> 
</div>

这里的$Bernoulli(p)$是取值为$\\{0,1\\}$的二项分布，其中取1的概率为$p$.Tacotron2中的attention state $\alpha_{t,n}$被替换为single, integer state variable $s_t$基于$\tau_t$随机演化得到。转移概率基于$s_t$状态的h-vector ($g_t$)和之前的帧$x_{1:t-1}$($a_t$),因此对于不同时刻$t$即使是相同的状态也是不一样的。

<div align=center>
    <img src="zh-cn/img/ch3/03/model_video.gif" /> 
</div>

如上图所示最终neural HMM是一个left-right no-skip的AR-HMM模型。encoder将每一个input sequence变成一个独一无二的HMM,$h_n$表示状态，decoder输入$h_n$这个状态向量和AR input $x_{1:t-1}$产生输出分布$o(\theta_t)$和时间$t$下一个状态$n$的转移概率$\tau_t$。

为了满足马尔可夫性，$(\theta_t,\tau_t)$只能依赖当前的状态$s_t$和过去的观测$x_{1:t-1}$,因此将tacotron2中的OutputNet中的LSTM结构替换为feedforward layer,这也大大降低了参数量。

最后Tacotron2还设计了一个基于卷积的post-net用来增强频谱的生成，这类似于经典SPSS中的后滤波和全局方差补偿。Tacotron2损失函数最小化post-net前后的频谱的MSE。这种不可逆的post-net不适合我们这种基于对数似然的模型。对于这里post-net可以单独训练或使用一个可逆的方式实现。这一部分将作为我们未来探索的工作。

##### 3.2 Practical considerations

+ Numerical stability: HMM中大量使用了“log-sun-exp trick”，但是会出现$ln0=-\infty$的情况，会导致在pytorch下计算梯度是NaN的情况。类似于经典的HMM方法，我们选择使用对角Gaussian 输出分布$o(\mu,\sigma)$作为输出分布。使用$softplus=log(1+exp^{x})$非线性方式拟合$\sigma$，具有非零最小值，此处为0.001，这在其他生成模型中是重要的。

+ Architecture enhancements: Tacotron 2可以使用软注意力来表示中间状态，因为$\alpha_{t;n}$值有许多自由度。取而代之的是，主要的基于HMM的合成器每个输入音素使用5个子状态，并以200帧/秒的速度运行。Tacotron 2以80帧/秒的速度运行，即40%的帧速率，因此我们每个音素使用2个状态来获得与这些HMM相同的时间分辨率。
这是通过将解码器输出层的大小加倍并将其输出解释为每个音素的两个级联状态向量$h$来实现的。

经典的基于HMM的TTS包括几个相邻帧之间的依赖关系模型，以促进时间上平滑的输出。尽管本文中的Tacotron2和神经HMM只将最新的帧$x_{t-1}$作为AR输入，但Eq.（1）意味着它们可以任意记忆很久以前的信息，这有利于对话语级韵律进行建模。我们还将$x_0$，即初始AR上下文（“go token”）视为可学习的参数。

+ Initialisation: HMM通常使用平启动进行初始化，其中所有状态都具有相同的统计数据。通过将解码器输出层中的所有权重归零，但将其他层初始化为normal，所有状态将具有相同的输出（零），但梯度不同且为非零，从而实现学习。选择最后一层偏置值,以便在训练开始时，每个z状态的的$\mu=0$和$\sigma=1$与我们归一化数据的全局统计数据相匹配。

+ Training: 神经HMM训练是新旧的混合：我们使用经典的（按比例）前向算法来计算精确的序列对数似然，但随后利用反向传播和自动微分来使用Adam进行优化。这些部分分别对应于（广义的）EM算法的E步骤和M步骤。训练期间的计算在状态上是平行的，但与Tacotron 2一样，由于时间的重复，在时间上是连续的。

线性AR HMM的最大似然估计可能导致模型不稳定。非线性自回归神经TTS也存在类似的问题。Tacotron 2通过在pre-net中添加dropout来解决这个问题，我们在这里保留了这个解决方案。

+ Synthesis: 我们可以迭代使用第3.1节中的方程，并随机采样新的帧$x_t \sim o(\theta_t)$。然而，基于HMM的TTS通常受益于确定性地生成典型输出，而不是随机采样。对于声学模型，这是通过生成最可能的输出序列来实现的，该输出序列高斯分布的$o(\theta_t)$的均值$\mu_t$相同，迭代的取$x_t=\mu_t$(模型结构图中的红色箭头)。这与Tacotron 2的输出生成密切相关，因为它是使用MSE进行训练的，MSE通过平均$E[X_t]$最小化。

SSNT-TTS发现，在合成时随机采样转变导致较差的暂停持续时间，并且经典的基于HMM的系统通常将每个状态中的时间基于该状态的平均持续时间。这个平均值很难用通过转移概率$\tau_t$隐式定义的持续时间分布来计算，如这里所示。相反，我们使用[24，32]中的简单算法来确定
基于持续时间分位数的持续时间生成（例如，中位数而不是平均值）。分位数阈值控制说话速率，可以根据各状态进行调整。对于本文评估的模型，非正式听力表明声学和持续时间的确定性生成都导致了清晰质量改进,网页上提供了示例。


#### 4.Expriments

<div align=center>
    <img src="zh-cn/img/ch3/03/p4.png" /> 
</div>

上图展示了Neural HMM的模型要比Tacotron2模型大小要小，这里的T2+P表示Tacotron2添加post-net,T2-P标表示Tacotron2不添加post-net,NH2表示Neural HMM每个音素有2个状态，NH1表示Neural HMM每个音素有1个状态。

<div align=center>
    <img src="zh-cn/img/ch3/03/p5.png" /> 
</div>

训练过程中Tactron2要在14.5K步迭代后才能合成连贯的语音，而NH2仅需要1K步的更新训练就可以，上图展示了合成100个验证话语的谷歌ASR单词错误率（WER）在训练过程中的演变（包括对数据的一小部分（500个话语）进行训练的结果）。


#### 5.Conclusion and Future Work

我们描述了经典的和现代的TTS范式，可以将其组合得到完全 probabilistic, attention-free的seq2seq的基于neural HMM的TTS模型。
我们的示例系统比Tacotron2更小，但达到了相当的自然度，更快地学会说话和对齐，需要更少的数据，而且不会胡言乱语。据我们所知这是第一个基于HMM的模型在语音质量上优于神经网络的TTS模型。神经HMM还允许容易地控制合成语音的说话速率。

未来的工作包括更强的网络架构，例如，基于transformer和单独训练的post-net。将Neural HMM与强大的分布族结合比如normalising flows，或者替换Gaussion假设或者替换为一个类似于post-net的网络来拟合分布。这可以允许采样语音的自然度超过确定性输出生成的自然度。

------

<!-- #CNN -->

### 4. DeepVoice v1,v2,v3

!> v1: https://arxiv.org/abs/1702.07825

!> v2: https://arxiv.org/abs/1705.08947

!> v3: https://arxiv.org/abs/1710.07654


#### 1.Deep Voice: Real-time Neural Text-to-Speech

<!-- https://blog.csdn.net/weixin_42721167/article/details/113062548 -->

##### 1.简介

这篇文章介绍了 Deep Voice ，一个完全由深度神经网络构建的生产质量的文本到语音系统，为真正的端到端神经语音合成奠定了基础。
该系统包括五个主要的模块：定位音素边界的分割模型、字素-音素转换模型、音素时长预测模型、基频预测模型和音频合成模型。
在音素分割模型中，论文提出了一种基于深度神经网络的音素边界检测方法。
对于音频合成模型，论文实现了一个不同的 WaveNet ，它需要的参数更少，比原来的训练更快。
       
通过为每个组件使用神经网络，论文提出的系统比传统的文本到语音系统更简单、更灵活；在传统的系统中，每个组件都需要费力的特征工程和广泛的领域专业知识。最后，论文提出的系统可以比实时更快地执行推理，并描述了 CPU 和 GPU 上优化的 WaveNet 推理内核，比现有的实现可达到400倍的速度。

Deep Voice 是受传统的文本-语音管道的启发，采用相同的结构，但用神经网络取代所有组件，使用更简单的特征：首先将文本转换为音素，然后使用音频合成模型将语言特征转换为语音。与之前的工作不同(使用手工设计的特征，如光谱包络、光谱参数、非周期参数等)，系统中唯一的特征是：带有重音标注的音素、音素持续时间和基频(F0)。这种特性的选择使系统更容易适用于新的数据集、声音和领域，而不需要任何手动数据注释或额外的特性工程。
       
论文们通过在一个全新的数据集上重新训练整个管道，而不修改任何超参数来演示这一声明，该数据集只包含音频和未对齐的文本副本，并生成相对高质量的语音。在传统的 TTS 系统中，这种调整需要几天到几周的时间，而 Deep Voice 只需要几个小时的人工工作和模型训练时间。
       
实时推理是生产质量TTS系统的要求；如果没有它，系统就不能用于大多数TTS的应用。先前的工作已经证明，WaveNet 就可以产生接近人类水平的语音。然而，由于 WaveNet 模型的高频、自回归特性，WaveNet推理提出了一个令人生畏的计算问题，迄今为止还不知道这种模型是否可以用于生产系统。
我们肯定地回答了这个问题，并演示了高效、实时的 WaveNet 推断内核，产生高质量的 16khz 音频，并实现了比以前的 WaveNet 推断实现 400 倍的加速。

##### 2.相关研究

之前的研究使用神经网络替代多个TTS系统组件，包括字素-音素转换模型，音素持续时间预测模型，基础频率预测模型和音频合成模型。然而，与 Deep Voice 不同的是，这些系统都不能解决 TTS 的全部问题，许多系统使用专门为其领域开发的手工工程特性。最近，在参数音频合成方面有很多工作，特别是 WaveNet 、 SampleRNN 和 Char2Wav。虽然 WaveNet 可以用于条件和无条件音频产生，但 SampleRNN 只用于无条件音频产生。 Char2Wav 用一个基于注意力的音素持续时间模型和等效的 F0 预测模型扩展了 SampleRNN ，有效地为基于 SampleRNN 的声码器提供了本地条件信息。

Deep Voice 在几个关键方面与这些系统不同，显著地增加了问题的范围。首先， Deep Voice 是完全独立的；训练新的 Deep Voice 系统不需要预先存在的 TTS 系统，可以使用短音频剪辑数据集和相应的文本文本从头开始。相反，复制上述两个系统都需要访问和理解已存在的 TTS 系统，因为它们在训练或推理时使用来自另一个 TTS 系统的特性。
其次， Deep Voice 最大限度地减少了人工工程功能的使用;它使用独热编码字符进行字素到音素的转换、独热编码的音素和重音、音素持续时间（毫秒）和标准化对数基频（可以使用任何 F0 估计算法从波形计算）。
所有这些都可以很容易地从音频和文本以最小的努力获得。相比之下，以前的工作使用了更复杂的特性表示，如果没有预先存在的 TTS 系统，就不可能有效地复制系统。

WaveNet 从 TTS 系统使用多个特性，包括如一个词的音节数量，位置词的音节，当前帧的音素，语音频谱的和动态特征频谱和激发参数，以及它们的衍生品。
Char2Wav 依赖世界 TTS 系统的声码器特征来预训练他们的对齐模块，其中包括 F0 、频谱包络和非周期参数。

最后，我们关注于创建一个可用于生产的系统，这要求我们的模型实时运行以进行推理。 Deep Voice 可以在几分之一秒内合成音频，并提供了合成速度和音频质量之间的可调平衡。相比之下，以前的 WaveNet 结果需要几分钟的运行时间才能合成一秒钟的音频。 SampleRNN 原始版本中描述的 3 层架构在推理过程中需要的计算量大约是我们最大的 WaveNet 模型的 4-5 倍，所以实时运行模型可能会很有挑战性。

##### 3.TTS 系统组件

TTS系统由五个主要的构建模块组成：
1. **字素到音素模型**：将书面文本(英语字符)转换为音素(使用像ARPABET这样的音素字母表编码)。
2. **分割模型**：对语音数据集中的音素边界进行定位。给定一个音频文件和音频的一个音素逐音素转录，分割模型确定每个音素在音频中的起始和结束位置。
3. **音素持续时间模型**：预测音素序列(一句话)中每个音素的时间持续时间。
4. **基频模型**：预测一个音素是否被发声；如果是，该模型预测整个音素持续时间的基频（F0）。
5. **音频合成模型**：将字素到音素、音素持续时间和基频预测模型的输出组合，并以与所需文本相对应的高采样率合成音频。

<div align=center>
    <img src="zh-cn/img/ch3/04/p1.png" /> 
</div>

描述（a）训练程序和（b）推理程序的系统图，输入在左边，输出在右边；在系统中，持续时间预测模型和 F0 预测模型是由一个经过联合损失训练的单一神经网络来实现的；字素到音素模型用作音素字典中不存在的单词的后备；虚线表示非学习组件。
       
在推理过程中：通过字素-音素模型或音素字典输入文本来生成音素；接下来，将音素作为音素持续时间模型和 F0 预测模型的输入提供，以便为每个音素分配持续时间并生成 F0 轮廓；最后，将音素、音素时长和 F0 作为语音合成模型的局部条件输入特征，生成最终的语音。

与其他模型不同的是，在推理过程中不使用分割模型；相反，它用于用音素边界注释训练语音数据；音素边界包含音素持续时间，可以用来训练音素持续时间模型；在语音合成模型中，使用带有音素、音素持续时间和基频的音频进行训练。

###### 3.1 字素到音素模型(Grapheme-to-Phoneme Model)

字素到音素模型，是基于 “字素到音素转换的序列到序列神经网络模型” 开发的编码器-解码器架构。使用了具有门控递归单元（GRU）非线性的多层双向编码器和同样深度的单向 GRU 解码器。每个解码器层的初始状态初始化为对应编码器的隐藏状态。
       
该体系结构采用teacher forcing训练，采用beam search进行解码。我们在编码器中使用 3 个双向GRU(每个层 1024 个隐层单元)，在解码器中使用 3 个大小相同的单向GRU，beam search size为5。在训练过程中，我们在每个循环层后使用概率为 0.95 的 Dropout 算法。
训练过程中,我们使用 Adam 优化算法与 `β1 = 0.9, β2 = 0.999, ε= 10^-8`, 批处理大小为 `64`，学习速率为 `10^-3` ，退火速率为 `0.85` 每`1000`次迭代。

###### 3.2 分割模型

分割模型经过训练，输出对齐的给定的对话和目标音素序列。
这个任务类似于语音识别中将语音与文本输出对齐的问题。在该领域，CTC损失函数已被证明专注于字符对齐，以学习声音和文本之间的映射。
我们从最先进的语音识别系统中采用了卷积递归神经网络体系结构，用于音素边界检测。
用 CTC 训练生成音素序列的网络将为每个输出音素产生简短的峰值。尽管这足以粗略地将音素与音频对齐，但还不足以检测到精确的音素边界。
为了克服这一点，我们训练预测音素对的序列而不是单个音素。然后，该网络将倾向于在接近一对音素之间边界的时间步输出音素对。
       
输入音频以 10 毫秒步长计算 20 个梅尔频率倒谱系数（ MFCCs ）为特点。在输入层之上，有两个卷积层(时间和频率的 2D 卷积)，三个双向循环的 GRU 层，最后是一个 softmax 输出层。
卷积层使用单位步幅、高度 9 （频率）和宽度 5 （时间）的核，循环层使用 512 个 GRU 单元(每个方向)。
在最后一次卷积和递归层之后，应用概率为 0.95 的 Dropout 。为了计算音素对错误率，我们采用波束搜索进行解码。
为了解码音素边界，我们在相邻的音素对至少有一个音素重叠的约束下进行宽度为 50 的波束搜索，并跟踪每个音素对在对话中的位置。
在训练中,我们使用 Adam 优化算法与 `β1 = 0.9 ， β2 = 0.999 ， ε= 10^-8` ，批处理大小为 `128` ， `10^-4 `的学习速率和退火速率 `0.95` 每 `500` 次迭代。


###### 3.3 音素持续时间模型与基频模型

用一个模型来同时预测音素持续时间和随时间变化的基频。
该模型的输入是一个带有重音的音素序列，每个音素和重音都被编码为一个one-hot vector。
该模型包括两个全连接层，每个层有 256 个隐藏单元，然后是两个单向循环层GRU，每个层有 128 个GRU单元，最后是一个全连接的输出层。在初始全连接层和最后一个循环层之后，应用概率为 0.8 的 Dropout 。
最后一层对每个输入音素产生三种估计:`音素持续时间`、`音素清音的概率(即具有基频)`和 `20 个时间相关的 F0 值`，这些值在预测持续时间内均匀采样。
       
该模型通过最小化音素持续时间误差、基频误差、音素清音概率的负对数似然以及与 F0 相对于时间的绝对变化成比例的惩罚项来实现平滑，从而使音素持续时间误差、基频误差、音素清音概率的负对数似然以及与 F0 相对于时间的绝对变化成比例的联合损失达到最优。
       
在训练过程中,我们使用 Adam 优化算法与 `β1 = 0.9 ， β2 = 0.999 ， ε= 10^-8` ， 批处理大小为 `128` ， `3*10^-4`的学习速率和退火速率 `0.9886` 每 `400` 次迭代。

###### 3.4 音频合成模型

<div align=center>
    <img src="zh-cn/img/ch3/04/p2.png" /> 
</div>

音频合成模型是 WaveNet 的一个变种。 WaveNet 由一个调节网络（conditioning network）和一个自回归网络组成，该网络将语言特征upsample到所需的频率，并在离散音频样本 $y \in \\{0,1,...,255\\}$上生成概率分布 $P(y)$ 。
我们改变层数$l$，残差通道数$r$ (每层隐藏状态的维数)，以及skip层通道数$s$ (在输出层之前层输出被投影到的维数)。
       
WaveNet 包括一个上采样和调节网络，然后是带有$r$ 的残差输出通道和 tanh 非线性门控的 `2*1` 的卷积层。
我们用 $W_{prev}$ 和 $W_{cur}$ 将卷积分解为每个时间步的两个矩阵乘法。这些层通过残差连接连接起来。
每一层的隐藏状态连接到一个 $l_r$ 向量，用 $W_{skip}$ 投影到 s skip层通道，然后用 relu 非线性进行两层 `1*1` 的卷积(权值 $W_{relu}$ 和 $W_{out}$ )。
       
WaveNet 使用转置卷积进行上采样和调节。我们发现，如果我们先用一叠双向quasi-RNN（Quasi-recurrent neural networks. arXiv
preprint arXiv:1611.01576, 2016.）（ QRNN ）层对输入进行编码，然后通过重复到所需频率进行上采样，我们的模型会表现得更好，训练得更快，需要的参数更少。
       
我们最高质量的最终模型使用 40 层网络， `r = 64` 个残差通道， `s = 256` 个skip层通道。在训练过程中，我们使用了 `β1 = 0.9 ， β2 = 0.999 ， ε= 10^-8` 的 Adam 优化算法，批量大小为 `8` ，学习率为 `10^-3` ，每 `1000` 次迭代退火率为 `0.9886` 。

其最终的网络结构如上如所示。

##### 4.结果

我们在一个内部英语语音数据库上训练我们的模型，该数据库包含约 20 小时的语音数据，这些数据被分割为 13079 个话语。
此外，我们还提供了基于Blizzard 2013 数据子集训练的模型的音频合成结果。
这两个数据集都是由一位职业女性演讲者说的。所有的模型都是使用 TensorFlow 框架实现的

###### 4.1 分割结果

我们使用 8 个 Titanx Maxwell GPU 进行训练，将每个批处理在 GPU 之间平均分配，并在不同 GPU 上使用环形全局归约平均梯度的计算，每次迭代大约花费 1300 毫秒。
经过大约 14000 次迭代，模型收敛到 7% 的音素错误率。我们还发现音素边界不一定是精确的，随机移动音素边界 10-30 毫秒对音素质量没有影响，因此我们怀疑音素质量对超过某一点的音素错误率不敏感。

###### 4.2 字素到音素结果

 我们在 CMUDict 获得的数据上训练了一个字素-音素模型。我们去掉了所有不以字母开头、包含数字或有多个发音的单词，这样在原来的 133854 个字素-音素序列对中就剩下了 124978 个字素-音素序列对。
       
我们使用一个 Titanx Maxwell GPU 进行训练，每次迭代大约花费 150 毫秒。经过约 20000 次迭代，该模型的音素错误率为 5.8% ，单词错误率为 28.7% ，与之前的结果基本一致。
与之前的工作不同，我们在解码过程中不使用语言模型，在我们的数据集中不包括具有多种发音的单词。

###### 4.3 音素持续时间与基频结果

我们使用一个 Titanx Maxwell GPU 进行训练，每次迭代大约需要 120 毫秒。在大约 20000 次迭代后，模型收敛到 38 毫秒(音素持续时间)和 29.4 Hz (基频)的平均绝对误差。

###### 4.4 音频合成结果

我们将音频数据集中的话语划分为一个一秒块，每个块有 1/4 秒的上下文，在开始时用 1/4 秒的静默填充每句语。我们过滤掉主要为静默的块，最后总共剩下74348块。
       
我们对模型进行了不同深度的训练，包括残差层堆栈中的 10 、 20 、 30 和 40 层。我们发现低于 20 层的模型会导致较差的音频质量。 20 层、 30 层和 40 层模型都能产生高质量的可识别语音，但 40 层模型的噪声比 20 层模型小，可以通过高质量的耳机检测到。
       
先前的研究已经强调了感受野大小对决定模型质量的重要性。事实上， 20 层模型的感受野只有 40 层模型的一半。然而，当在 48khz 运行时， 40 层的模型只有 83 毫秒的感受野，但仍然可以产生高质量的音频。这表明 20 层模型的接收场是充分的，我们推测音频质量的差异是由于其他因素而不是接收场大小。
       
我们在 8 个 Titanx Maxwell GPU 上训练，每个 GPU 有一个数据块，使用环形全局归约在不同的 GPU 上计算平均梯度。每次迭代大约需要 450 毫秒。我们的模型在大约 300000 次迭代后趋于收敛。我们发现单个 1.25s 的数据块就足以使 GPU 的计算饱和，而批处理并不能提高训练效率。
       
与高维生成模型一样，模型丢失与个体样本的感知质量在一定程度上无关。当模型具有异常高的损失声音明显嘈杂，模型优化低于某一阈值没有损失指示其质量。此外，模型架构中的变化（如深度和输出频率）可能会对模型损耗产生重大影响，而对音频质量的影响则很小。
       
为了评估我们的 TTS 管道中各个阶段的感知质量，我们使用 CrowdMOS 工具包和方法从 Mechanical Turk 众包了平均意见评分（ MOS ）评分。
为了单独的音频预处理的效果， WaveNet 模型质量，和音素持续时间和基频模型质量,我们提出 MOS 得分对于各种话语类型,包括合成结果的 WaveNet 输入（持续时间和 F0 ）从地面实况音频中提取，而不是由其他模型合成。
结果如下表所示。我们有目的地在评分者评估的每批样本中包含地面实况样本，以突出人类语言的增量，并允许评分者区分模型之间更细粒度的差异；这种方法的缺点是，产生的 MOS 分数将显著低于如果评分者只提供合成音频样本。

<div align=center>
    <img src="zh-cn/img/ch3/04/p3.png" /> 
</div>

首先，我们发现 MOS 显著下降时只需将采样音频流从 48 kHz至 16 kHz,特别是结合$\mu-law$压缩和量化,可能是因为 48 kHz 样本呈现给评级机构作为基准为 5 分，一个低质量的嘈杂的合成结果是 1 分。

当使用Ground truth的持续时间和 F0 时，我们的模型得分很高，我们的模型的 95% 置信区间与Ground truth样本的置信区间相交。然而，使用合成频率会降低 MOS ，进一步包括合成时间会显著降低 MOS 。

结论是，朝向自然 TTS 的进展的主要障碍在于持续时间和基频预测，而我们的系统在这方面并没有有意义地进步超过目前的技术水平。

最后，我们的最佳模型运行速度略慢于实时模型，因此我们通过对运行速度比实时模型快 1 倍和 2 倍的模型获得分数来调整模型大小，从而证明综合质量可以换取推理速度。
       
我们也测试了基于原始 WaveNet 的完整特征集训练的 WaveNet 模型，但是没有发现这些模型和基于我们的简化特征集训练的模型之间有知觉上的差异。

<div align=center>
    <img src="zh-cn/img/ch3/04/p4.png" /> 
</div>

###### 4.5 Blizzard数据集结果

为了演示我们系统的灵活性，我们在Blizzard 2013 数据集上用相同的超参数重新训练了所有模型。在我们的实验中，我们使用了一个 20.5 小时的数据集子集，该子集被分割为 9741 个话语。我们评估了模型，该过程鼓励评价者直接比较合成音频与地面真实情况。显示设置， 16 kHz 组合和扩展音频接收 MOS 评分 4.65±0.13 ，而我们的合成音频接收 MOS 评分 2.67±0.37 。

##### 5.优化推理

虽然 WaveNet 在生成高质量的合成语音方面表现出了很大的潜力，但最初的实验报告显示，短语音的生成时间可达数分钟或数小时。由于模型的高频、自回归特性，波网推理提出了一个难以置信的具有挑战性的计算问题，它比传统的递归神经网络需要更多数量级的时间步长。

当生成音频时，单个样本必须在大约 60 秒（ 16khz 音频）或 20 秒（ 48khz 音频）内生成。对于我们的 40 层模型，这意味着单层（由几个矩阵乘法和非线性组成）必须在大约 1.5 秒内完成。相比之下,访问驻留在内存的值在一个CPU可以 0.1 毫秒。

为了实时执行推断，我们必须非常小心，永远不要重新计算任何结果，将整个模型存储在处理器缓存（而不是主存）中，并优化利用可用的计算单元。这些相同的技术可以用于使用 PixelCNN 加速图像合成到每幅图像不到一秒。
       
用我们的 40 层 WaveNet 模型合成一秒钟的音频大约需要 55109 个浮点运算（ FLOPs ）。任何给定层中的激活都依赖于前一层和前一时间步中的激活，因此必须一次一个时间步和一层地进行推理。

单个层只需要 `42*10^3` 次 FLOPs ，这使得实现有意义的并行性非常困难。除了计算要求，该模型有大约 `1.6*10^6`个参数，如果用单一精度表示，这相当于约 6.4 MB 。
       
在 CPU 上，单个 Haswell 或 Broadwell 核的单精度吞吐量峰值约为 `77*10^9` 次 FLOPs ， l2 - l1 缓存带宽约为 140 GB/s 。模型必须在每个时间步从缓存加载一次，这需要 100GB/s 的带宽。即使模型适合 L2 缓存，实现也需要利用最大带宽的 70% 和峰值 FLOPS 的 70% ，以便在单个核心上实时进行推断。
跨多个核分割计算减少了问题的难度，但它仍然具有挑战性，因为推理必须在最大内存带宽和峰值 FLOPs 的很大一部分下运行，同时保持线程同步。
       
与 CPU 相比， GPU 具有更高的内存带宽和峰值 FLOPs ，但它提供了更专门的、因此更有限制的计算模型。为每一层或每个时间步启动一个内核的幼稚实现是站不住脚的，但是基于持久性 RNN 技术的实现可能会利用 GPU 提供的吞吐量。
       
我们在 CPU 和 GPU 上实现了高速优化的推理内核，并证明了可以实现比实时速度更快的波网推理。上表列出了不同型号的 CPU 和 GPU 推断速度。在这两种情况下，基准只包括自回归、高频音频生成，而不包括语言条件作用特征的生成（这可以在整个话语中并行完成）。

我们的 CPU 内核是实时运行的，或者在某些模型上运行得比实时还要快，而 GPU 模型还没有达到这样的性能。

###### 5.1 CPU实现

我们通过避免任何重计算、进行缓存友好的内存访问、通过多线程实现高效同步并行工作、最小化非线性故障、通过线程固定避免缓存抖动和线程争用，以及使用定制的硬件优化例程进行矩阵乘法和卷积来实现实时 CPU 推理。
       
对于CPU的实现，我们将计算分为以下步骤:
1. 样本嵌入
2. 层推理
3. 输出

如下图所示，我们在两组线程之间并行处理这些内容。

<div align=center>
    <img src="zh-cn/img/ch3/04/p5.png" /> 
</div>

 将线程固定到物理内核（或禁用超线程）对于避免线程争用和缓存抖动非常重要，并可将性能提高约 30% 。
       
根据模型大小的不同，非线性（ tanh 、 sigmoid 和 softmax ）也会占用推断时间的很大一部分，因此我们用高精度近似代替所有非线性。

这些近似产生的最大绝对误差为 tanh 的 `1.5*10^3` ， sigmoid 的 `2.5*10^3` ， e^x 的 `2.4*10^5` 。使用近似而不是精确的非线性，性能增加了大约 30% 。
       
我们也实现了将权重矩阵量化到 int16 的推理，并发现使用量化时感知质量没有变化。对于更大的模型，量化在使用更少的线程时提供了显著的加速，但线程同步的开销使它在使用更大的线程时不起作用。
       
最后，我们使用 PeachPy 专门为我们的矩阵大小编写定制的 AVX 汇编内核，用于矩阵向量乘法。在使用 float32 时，使用自定义汇编内核的推断速度比 Intel MKL 快 1.5 倍，比 OpenBLAS 快 3.5 倍。两个库都不提供等效的 int16 操作。

###### 5.2 GPU实现

由于它们的计算强度，许多神经模型最终部署在 gpu 上， gpu 的计算吞吐量比 cpu 高得多。因为我们的模型是基于内存带宽和失败的，所以在 GPU 上运行推断似乎是一个很自然的选择，但结果却带来了一系列不同的挑战。
       
通常，代码在 GPU 上以一系列内核调用的方式运行，每个矩阵乘或向量操作都是它自己的内核。然而， CUDA 内核启动的延迟（可能高达 50 毫秒）和从 GPU 内存加载整个模型所需的时间对于这样的方法来说是非常大的。这种风格的推理内核最终会比实时内核慢大约 1000 倍。
       
为了在 GPU 上接近实时性，我们转而使用持久 RNNs 技术构建一个内核，它在一个内核启动中生成输出音频中的所有样本。模型的权值只加载到寄存器一次，然后在整个推理过程中不卸载它们而使用。

由于 CUDA 编程模型和这些持久内核之间的不匹配，生成的内核专门针对特定的模型大小，编写起来非常费力。

虽然我们的 GPU 推断速度不是很实时，但我们相信通过这些技术和更好的实现，我们可以在 GPU 和 CPU 上实现实时的 WaveNet 推断


##### 6.结论

在这项工作中，我们通过构建一个完整的神经系统，证明了当前的深度学习方法对于一个高质量的文本到语音引擎的所有组件都是可行的。我们将推理优化到比实时更快的速度，表明这些技术可以应用于以流媒体的方式实时生成音频。我们的系统是可训练的，无需任何人力参与，极大地简化了创建 TTS 系统的过程。
       
我们的工作为探索开辟了许多新的可能方向。通过仔细优化，在 GPU 上进行模型量化，在 CPU 上进行 int8 量化，以及在 Xeon Phi 等其他架构上进行试验，可以进一步提高推理性能。
       
另一个方向自然是去除阶段之间的分离和合并的分割,持续时间预测,预测模型和基本频率直接到音频合成模型,从而将问题转化为一个完整的 sequence-to-sequence 模型,创建一个端到端可训练的 TTS 系统,整个系统允许我们训练没有中间监督。
为了代替融合模型，通过更大的训练数据集或生成建模来改进持续时间和频率模型技术可能会对声音的自然性产生影响。

------

#### 2.Deep Voice 2: Multi-Speaker Neural Text-to-Speech

<!-- https://blog.csdn.net/weixin_42721167/article/details/113681731 -->

##### 摘要

本文介绍了一种利用低维可训练说话人嵌入增强神经文本到语音的技术，以从单个模型产生不同的声音。

作为起点，我们展示了针对单说话人神经 TTS 的两种最先进方法的改进： Deep Voice 1 和 Tacotron 。
我们引入了 Deep Voice 2 ，它基于与 Deep Voice 1 类似的管道，但构建了更高的性能构建块，并表现出较 Deep Voice 1 更显著的音频质量改进。

我们通过引入后处理神经声码器来改进 Tacotron ，并展示了显著的音频质量改进效果。然后，我们在两个多说话人 TTS 数据集上演示了用于 Deep Voice 2 和 Tacotron 的多说话人语音合成技术。

##### 1.介绍

人工语音合成，通常称为文本到语音( TTS )，在技术接口、可访问性、媒体和娱乐等方面有多种应用。
大多数 TTS 系统都是用单个讲话者语音构建的，而通过拥有不同的语音数据库或模型参数来提供多个讲话者语音。
因此，与只支持单一语音的系统相比，开发支持多个语音的 TTS 系统需要更多的数据和开发工作。

在这项工作中，我们证明我们可以建立全神经的多说话人 TTS 系统，在不同的说话人之间共享绝大多数的参数。
我们证明，一个单一的模型不仅可以从多个不同的声音产生语音，而且与训练单说话者系统相比，每个说话者所需要的数据也明显较少

具体而言，我们的贡献如下:
1. 我们提出了一种基于 Deep Voice 1 (《Deep voice: Real-time neural text-to-speech》)的改进架构： Deep Voice 2 ；
2. 我们介绍了一种基于 WaveNet (《Wavenet: A generative model for raw audio》)的声谱图到音频神经声码器，并将其与 Tacotron (《Tacotron: Towards end-to-end speech synthesis.》) 一起使用，作为 Griffin-Lim 音频生成器的替代品；
3. 以这两个单说话人模型为基线，我们通过在 Deep Voice 2 和 Tacotron 中引入可训练的说话人嵌入来演示多说话人神经语音合成。

本文的其余部分组织如下：
第二节讨论了相关的工作，以及使本文与之前工作做出不同贡献的原因；
第三节介绍了 Deep Voice 2 ，并凸显了与 Deep Voice 1 的区别；
第四节解释了神经 TTS 模型的说话人嵌入技术，并展示了 Deep Voice 2 和 Tacotron 架构的多说话人变体；
第五节第一部分通过平均意见评分( MOS )评估量化了单说话人 TTS 的改进，第二部分通过 MOS 评估和多说话人鉴别器精度度量给出了 Deep Voice 2 和 Tacotron 的综合音频质量；
第六节给出结论并对结果和未来工作进行讨论。

##### 2.相关工作

我们按顺序讨论我们在第一节提出的每个相关工作，从单说话人神经语音合成开始，然后转向多说话人语音合成和生成模型质量度量。
       
关于单说话人的语音合成，深度学习已被用于各种各样的子组件，包括持续时间预测(《Fast, compact, and high quality LSTM-RNN based statistical parametric speech synthesizers for mobile devices》)，基本频率预测(《Median-based generation of synthetic speech durations using a non-parametric approach》)，声学建模(《Unidirectional long short-term memory recurrent neural network with recurrent output layer for low-latency speech synthesis》)，以及自回归逐样本音频波形生成器(《SampleRNN: An unconditional end-to-end neural audio generation model》)

我们的贡献建立在最近的完全神经 TTS 系统方面的工作基础上，包括 Deep Voice 1 (《Deep voice: Real-time neural text-to-speech》)、 Tacotron(《Tacotron: Towards end-to-end speech synthesis.》)和 Char2Wav(《Char2wav: End-to-end speech synthesis》) 。

这些工作集中在构建单说话人 TTS 系统，而我们的论文则集中在扩展神经 TTS 系统，以在每个说话人的数据更少的情况下处理多个说话人。

我们的工作并不是第一次尝试多说话人 TTS 系统。例如，在传统的基于 HMM 的 TTS 合成(《Robust speaker-adaptive hmm-based text-to-speech synthesis》)中，一个平均语音模型使用多个说话者数据进行训练，然后对其进行适配以适应不同的说话者。
基于 DNN 的系统(《On the training of DNN-based average voice model for speech synthesis》)也被用于构建平均语音模型， i-vector 表示说话人作为每个目标说话人额外的输入层和单独的输出层。类似地，Fan等人(《Multi-speaker modeling and speaker adaptation for DNN-based TTS synthesis》)在不同说话人之间使用带有说话人相关的输出层预测声码器参数(如线谱对、非周期性参数等)的共享隐藏表达。为了进一步研究，Wu等人(《A study of speaker adaptation for DNN-based speech synthesis》)对基于 DNN 的多说话人建模进行了实证研究。最近，生成对抗网络( GANs )(《Voice conversion from unaligned corpora using variational autoencoding wasserstein generative adversarial networks》)解决了说话人适应问题。

相反，我们使用可训练说话人嵌入多说话人 TTS 。该方法被研究在语音识别(《Fast speaker adaptation of hybrid NN/HMM model for speech recognition based on discriminative learning of speaker code》)，但也是语音合成的一种新技术。与之前依赖于固定嵌入(例如 i-vector )的工作不同，本工作中使用的说话人嵌入是与模型的其他部分一起从零开始训练的，因此可以直接学习与语音合成任务相关的特征。此外，这项工作不依赖于每个说话人的输出层或平均语音建模，那会导致更高质量的合成样本和更低的数据需求(因为每个说话人需要学习的唯一参数更少)。

为了以一种自动的方式评估生成的声音的区别性，我们建议使用说话人鉴别器的分类精度。类似的指标，如初始分数，已被用于图像合成的 GANs 定量质量评估(《Improved techniques for training gans》)。说话人分类研究既有传统的基于高斯均值的方法(《* Speaker verification using adapted gaussian mixture models*》)，也有最近的深度学习方法(《Deep speaker: an end-to-end neural speaker embedding system》)。

##### 3.单说话人 Deep Voice 2

在本节中，我们介绍了 Deep Voice 2 ，一种基于 Deep Voice 1 (《Deep voice: Real-time neural text-to-speech》)的神经 TTS 系统。我们保留了 Deep Voice 1 (《Deep voice: Real-time neural text-to-speech》)的一般结构，如下图所示。

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p1.png" /> 
</div>

Deep Voice 2 和 Deep Voice 1 的一个主要区别是音素持续时间和频率模型的分离。Deep Voice 1 有一个单一的模型来联合预测音素持续时间和频率模型(浊音和随时间变化的基频，F0)。在 Deep Voice 2 中，首先预测音素持续时间，然后作为频率模型的输入。

在随后的小节中，我们介绍了在 Deep Voice 2 中使用的模型。我们将在第五节第一部分对 Deep Voice 2 和 Deep Voice 1 进行定量比较。

###### 3.1 分割模型

与 Deep Voice 1 类似，在 Deep Voice 2 中，音素位置估计被视为一个无监督学习问题。分割模型是卷积—循环体系结构，带有连接主义时间分类( CTC )缺失(《Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks》)，用于音素对分类，然后使用音素对提取它们之间的边界。

Deep Voice 2 的主要架构变化是在卷积层中添加了批处理规范化和残差连接。其中， Deep Voice 1 的分割模型将每一层的输出计算为：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p2.png" width=20%/> 
</div>

其中 $h^{(l)}$ 是第 $l$ 层的输出， $W^{(l)}$ 为卷积滤波器组， $b^{(l)}$ 是偏置向量，`*`是卷积算子。
相反， Deep Voice 2 的分割模型层代替计算为：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p3.png" width=20%/> 
</div>

式中BN为批归一化(《Batch normalization: Accelerating deep network training by reducing internal covariate shift》)。
此外，我们发现该分割模型对沉默音素和其他音素之间的边界存在错误，这将显著降低在某些数据集上的分割精度。我们引入了一个小的后处理步骤来纠正这些错误:每当分割模型解码沉默边界时，我们调整边界的位置与沉默检测启发式。

###### 3.2 持续时间模型

在 Deep Voice 2 中，我们没有预测一个连续值的持续时间，而是将持续时间预测制定为一个序列标记问题。
我们将音素持续时间离散化为对数尺度的桶，并将每个输入音素分配给与其持续时间对应的桶标签。
我们通过在输出层具有成对电位的条件随机场( CRF )对序列进行建模(《Neural architectures for named entity recognition》)
在推理过程中，我们使用维特比前向后算法从 CRF 中解码离散时间。
我们发现量化持续时间预测和引入 CRF 隐含的两两依赖可以提高综合质量。

###### 3.3 频率模型

根据时长模型解码后，预测的音素时长将从每个音素输入特征中升级到频率模型的每个帧输入中。Deep Voice 2 频率模型由多个层次组成:首先，双向GRU层(《Learning phrase representations using rnn encoder-decoder for statistical machine translation》)通过输入特征生成隐藏状态。从这些隐藏的状态，仿射投影紧随一个 sigmoid 非线性产生每个帧被表达的概率。隐藏状态也被用来做两个单独的标准化 F0 预测。

首先用单层双向 GRU 进行 $f_{GRU}$ 预测，然后再用仿射投影进行预测。第二种预测 $f_{conv}，是通过将不同卷积宽度的多个卷积和单个输出通道的贡献相加得出的。最后，将隐态与仿射投影和 sigmoid 非线性相结合来预测混合比率 $ω$ ，并将两者归一化后的频率预测合并为：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p4.png" width=20%/> 
</div>

然后，将归一化预测 $f$ 转换为真实频率 F0 预测：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p5.png" width=20%/> 
</div>

其中， `F0` 和 `σF0` 分别为训练模型的说话者 F0 的均值和标准差。我们发现，混合卷积和循环层来预测 F0 的效果比单独预测任一层的效果都好。我们将此归因于一种假设，即在有效处理整个上下文信息的同时，广泛的卷积减少了循环层在大量输入帧中维护状态的负担。

###### 3.4 声码器

Deep Voice 2 语音模型基于一个 WaveNet 架构(《Wavenet: A generative model for raw audio》)，带有一个双层双向 QRNN (《Quasi-recurrent neural networks》)调节网络，类似于 Deep Voice 1 。然而，我们去掉了门控 tanh 非线性和残差连接之间的 `1 × 1` 卷积。此外，我们对WaveNet的每一层都使用相同的调节器偏差，而不是像在 Deep Voice 1 中那样为每一层产生单独的偏差。

##### 4.带有可训练说话人嵌入的多说话人模型

为了从多个说话人合成语音，我们在我们的每个模型中增加了一个低维说话人，并为每个说话人嵌入向量。
与以前的工作不同，我们的方法不依赖于每个说话人的权重矩阵或层。
与说话人相关的参数存储在一个非常低维的向量中，因此在说话人之间有几乎完全的权重共享。
我们使用说话人嵌入来产生递归神经网络( RNN )初始状态、非线性偏差和乘法门控因子，并在整个网络中使用。
说话人嵌入随机初始化，均匀分布在`[ – 0.1, 0.1]`，并通过反向传播联合训练。每个模型都有自己的一套扬声器嵌入。

为了鼓励每个说话人的独特声音特征影响模型，我们将说话人嵌入到模型的多个部分。
根据经验，我们发现仅仅提供扬声器嵌入输入层不适用于任何模型，除了发音模型。这可能使由于高度的残余 WaveNet 连接存在，以及学习高品质说话人嵌入的难度导致。
我们注意到有几种模式趋向于产生高性能：
+ 特定位置说话人嵌入：对于模型架构中的每个使用地点，通过仿射投影和非线性变换嵌入到适当的维度和形式的共享说话人。
+ 循环初始化：初始化循环层隐藏状态与特定位置的说话人嵌入。
+ 输入增加：连接一个特定位置的说话人嵌入到输入在每个时间步循环层。
+ 功能控制：多层激活，嵌入一个特定位置的说话人，以呈现可适应的信息流。

接下来，我们将描述如何在每个体系结构中使用说话人嵌入。

###### 4.1 多说话人 Deep Voice 2

Deep Voice 2 的每个模型都有单独的扬声器嵌入。然而，它们可以被看作是一个更大的独立训练的说话人块嵌入。

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p6.png" /> 
</div>

**4.1.1 分割模型**

在多说话人分割模型中，我们在卷积层的剩余连接中使用特征门控。我们将批归一化激活乘上一个特定位置的说话人嵌入：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p7.png" width=30%/> 
</div>

$g_s$ 是一种特定位置的说话人嵌入。在所有卷积层中，相同的特定位置的嵌入是共享的。此外，我们初始化每个循环层与第二个特定位置嵌入。类似地，每一层共享相同的特定位置的嵌入，而不是每一层有一个单独的嵌入。

**4.1.2 持续时间模型**

多说话人持续时间模型使用与说话人相关的循环初始化和输入增强。一种特定于站点的嵌入用于初始化 RNN 隐藏状态，另一种特定位置的嵌入通过连接到特征向量作为第一 RNN 层的输入。

**4.1.3 频率模型**

多说话人频率模型使用循环初始化，它用一个特定位置的说话人嵌入来初始化循环层(循环输出层除外)。如 3.3 节所述，单说话人频率模型中的递归输出层和卷积输出层预测归一化频率，然后通过固定的线性变换将其转换为真 F0 。线性变换依赖于说话人 F0 的均值和标准偏差。
这些值在不同的演讲者之间差别很大：例如，男性演讲者的 F0 均值往往要低得多。
为了更好地适应这些变化，我们使平均值和标准偏差可训练的模型参数，并将它们乘以依赖于说话人嵌入的缩放项。具体而言，我们计算 F0 预测为:

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p8.png" width=40%/> 
</div>

$g_f$ 是一种特定位置说话人嵌入， `F0` 和 `σF0` 是可训练的标量参数，初始化为数据集上 F0 的均值和标准偏差， $V_{\mu}$ 和 $V_{σ}$ 是可训练的参数向量。

$$softsign=\frac{x}{1+|x|}$$

**4.1.4 语音模型**
       
多说话人的语音模型只使用输入增强，特定位置的说话人嵌入连接到控制器的每个输入框上。
这与Oord等人(《Wavenet: A generative model for raw audio》)提出的全局条件反射不同，并允许说话者嵌入影响局部条件反射网络。
       
在没有说话人嵌入的情况下，由于频率和持续时间模型提供了独特的特征，语音模型仍然能够产生听起来比较清晰的声音。
然而，在语音模型中嵌入说话人可以提高音频质量。我们确实观察到嵌入收敛到一个有意义的潜在空间。

###### 4.2 多说话人 Tacotron

除了通过扬声器嵌入扩展 Deep Voice 2 ，我们还扩展了 Tacotron (《Tacotron: Towards end-to-end speech synthesis》)，一种序列到序列的字符到波形模型。

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p9.png" /> 
</div>

当训练多说话人的 Tacotron 变体时，我们发现模型的性能高度依赖于模型的超参数，并且一些模型常常不能学习一小部分说话人的注意机制。我们还发现，如果每个音频片段中的语音不在同一时间步长开始，模型就不太可能收敛到有意义的注意力曲线和可识别的语音。因此，我们在每个音频剪辑中削减了所有最初和最终的沉默。

由于模型对超参数和数据预处理的敏感性，我们认为可能需要额外的调整来获得最大的质量。因此，我们的工作重点是证明 Tacotron ，就像 Deep Voice 2 一样，能够通过说话人嵌入来处理多个说话人，而不是比较两种架构的质量。

**4.2.1 特征到谱图模型**

Tacotron 字符—频谱图体系结构由一个 convolution-bank-highway-GRU ( CBHG )编码器、一个注意力解码器和 CBHG 后处理网络组成。由于体系结构的复杂性，我们省略了完整的描述，而将重点放在我们的修改上。
       
我们发现将说话人嵌入到 CBHG 后处理网络中会降低输出质量，而将说话人嵌入到字符编码器中则是必要的。
如果没有与说话人相关的 CBHG 编码器，该模型就无法学习其注意机制，也无法产生有意义的输出。
为了使说话人编码器适应，我们在每个时间步用一个特定位置的嵌入作为每个高通图层的额外输入，并用第二个特定位置的嵌入初始化 CBHG RNN 状态。
       
我们还发现，增加说话人嵌入解码器是有帮助的。
我们使用一个特定位置嵌入解码器前置网络作为一个额外的输入，一个额外的特定位置嵌入作为初始上下文向量注意力 RNN 的关注，一个特定位置嵌入作为初始解码器格勒乌隐藏状态，和一个特定位置嵌入的偏倚 tanh 基于内容的注意机制。

**4.2.2 频谱图到波形模型**

在(《Tacotron: Towards end-to-end speech synthesis》)中的原始 Tacotron 实现使用 Griffin-Lim 算法通过迭代估计未知相位将谱图转换为时域音频波形。我们观察到，输入谱图中的小噪声会导致 Griffin-Lim 算法中明显的估计误差，产生的音频质量下降。
       
为了使用 Tacotron 而不是使用 Griffin-Lim 来产生更高质量的音频，我们训练了一个基于 WaveNet 的神经声码器，将线性声谱图转换为音频波形。
所使用的模型相当于 Deep Voice 2 的发声模型，但采用线性比例的对数幅度谱图代替音素身份和 F0 作为输入。
       
合成的 Tacotron–WaveNet 模型如上图所示。正如我们将在第5.1节中展示的那样，基于 WaveNet 的神经声码器确实也显著改善了单说话人 Tacotron 加速器。

##### 5.结果

在本节中，我们将展示使用所述架构进行单说话人和多说话人语音合成的结果。

###### 5.1 单说话人语音合成

我们在一个包含大约20小时的单说话人数据的内部英语语音数据库上训练 Deep Voice 1 、 Deep Voice 2 和 Tacotron 。
我们使用 crowdMOS 框架(《Crowdmos: An approach for crowdsourcing mean opinion score studies》)进行 MOS 评估，比较样本的质量，比较结果如下：

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p10.png" /> 
</div>

结果表明，在结构上的改进在 Deep Voice 2 产生了比 Deep Voice 1 显著的质量上的提高。使用 WaveNet 将 Tacotron 生成的声谱图转换为音频比使用迭代的 Griffin-Lim 算法更好。

###### 5.2 多说话人语音合成

我们在 VCTK 数据集上用 44 小时的语音训练所有上述模型，该数据集包含 108 个说话人，每个人大约有 400 个语音。
我们还在 audibooks 的内部数据集上训练所有模型，该数据集包含 477 个扬声器，每个说话人有 30 分钟的音频(总计 238 小时)。
从我们的模型中观察到的一致的样本质量表明，我们的架构可以很容易地学习数百种不同的声音，这些声音具有各种不同的口音和节奏。

如下图所示，学习到的嵌入位于一个有意义的潜在空间中。

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p11.png" /> 
</div>

(a) 80 层语音模型和(b) VCTK 数据集的字符-声谱图模型的学习说话人嵌入的主成分

为了评估合成音频的质量，我们使用 crowdMOS 框架进行 MOS 评估，结果显示在下表。

<div align=center>
    <img src="zh-cn/img/ch3/04-1/p12.png" /> 
</div>

我们有意将真实情况的样本包含在被评估的集合中，因为数据集中的口音对于我们的北美众包评分者来说很可能是陌生的，因此会因为口音而不是模型质量而被评为低评级。
通过包含真实情况样本，我们可以将模型的 MOS 值与真实情况 MOS 值进行比较，从而评估模型质量而不是数据质量。
然而，由于与真实情况样本的隐式比较，得到的 MOS 值可能较低。
总的来说，我们观察到，在考虑低采样率和扩展/扩展的情况下， Deep Voice 2 模型可以接近一个接近真实情况的 MOS 值。
       
高采样质量但无法区分声音的多说话人 TTS 系统会产生高 MOS 值，但不能满足准确再现输入声音的预期目标。
为了证明我们的模型不仅能产生高质量的样本，还能产生可区分的声音，我们还在我们生成的样本上测量了说话人判别模型的分类精度。
speaker discriminative 是一种卷积网络，训练它对说话人的话语进行分类，在与 TTS 系统本身相同的数据集上训练。
如果声音无法区分(或者音频质量较低)，那么合成样本的分类精度将远远低于真实情况样本。
根据上表分类精度可知，从我们的模型生成的样本与真实情况样本具有同样的可区分性。
只有使用 WaveNet 的 Tacotron 的分类精度显著较低，而且我们怀疑 WaveNet 会加剧谱图中的生成误差，因为它是用真实情况谱图训练的。

##### 6.结论

在这项工作中，我们探索如何通过低维可训练的说话人嵌入，将完全神经化的语音合成管道扩展到多说话人的文本到语音。
我们首先介绍一种改进的单扬声器模型— Deep Voice 2 。
接下来，我们通过训练多说话人 Deep Voice 2 和多说话人 Tacotron 模型来演示我们技术的适用性，并通过 MOS 评估它们的质量。
总之，我们使用说话人嵌入技术来创建高质量的文本–语音系统，并最终表明神经语音合成模型可以有效地从散布在数百个不同的说话人中的少量数据中学习。
       
本研究的结果为未来的研究提供了许多方向。
未来工作可能测试这种技术的局限性和探讨这些模型可以概括，多少人多少数据确实是每个说话人需要高质量的合成、新的说话人是否可以添加到一个系统通过修正模型参数和单独培训新的说话人嵌入,以及说话者嵌入是否能像词嵌入一样，可以作为一个有意义的向量空间。

------

#### 3.Deep Voice 3: Scaling Text-to-Speech with Convolutional Sqeuence Learning

<!-- 
https://blog.csdn.net/qq_37175369/article/details/81476473?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-7-81476473-blog-113681731.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-7-81476473-blog-113681731.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&utm_relevant_index=8 -->

!> https://github.com/r9y9/deepvoice3_pytorch

!> https://r9y9.github.io/deepvoice3_pytorch/


##### 摘要

我们现在的deepvoice3，是一个基于注意力机制的全卷积神经元TTS系统。它匹配最先进的神经元自然语音合成系统并且训练更快。我们用前所未有的TTS训练集，在超过800小时的音频2000多个人上训练deepvoice3。另外的，我们识别了基于注意力机制的语音合成网络的常见问题，证明了怎么去减轻他们，并且比较了一些不同的语音合成方法。我们还描述了如何在一个GPU服务器上对每天一千万个查询的推理进行衡量。

##### 1.引言

TTS系统转换文字到人类语音，TTS系统被用于各种各样的应用。传统TTS基于复杂的多阶段工程流水线。通常，这些系统首先将文本转换为压缩的音频特征，然后使用称为声码器(vocoder)的音频波形合成方法将该特征转换为音频。

最近关于神经网络TTS的研究已经取得了令人印象深刻的结果，产生了具有更简单的特征、更少的组件和更高质量的合成语音的流水线。关于TTS的最优神经网络结构还没有达成共识。然而，序列到序列模型(seq2seq)已经显现出有希望的结果。

在文本中，我们提出了一个新颖的，全卷积结构的语音合成，扩展到非常大的数据集，并且演示了部署一个基于注意力机制的TTS系统时出现的几个现实问题。具体来说，我们做了以下贡献：

1. 我们提出了一个全卷积的字符到声谱的结构，它使完全并行计算成为可能，并且比相似的循环神经网络结构快几个数量级。
2. 我们展示了我们的结构可以在包含了820小时的音频和2484个人的 LibriSpeech ASR数据集上训练得很快。
3. 我们证明了可以产生单调的注意力行为，避免通常的错误模式影响到sequence-to-sequence模型。
4. 我们对比了几个声波合成方法的质量，包括WORLD，Griffin-Lim和WaveNet。
5. 我们描述了一个用于Deep Voice 3的推理核心的实现，它可以在一个GPU服务器上每天提供多达一千万次查询。

##### 2.相关工作

我们的工作建立在最先进的神经网络语音合成和注意力机制的sequence-to-sequence学习之上。

最近一些作品解决了神经网络语音合成的问题，包括Deep Voice 1，Deep Voice 2，Tacotron，Char2Wav，VoiceLoop，SampleRNN和WaveNet。Deep Voice 1 & 2保持了传统的TTS流水线结构，分离的字型和音素的转换，持续时间和频率的预测，和声波的合成。对比Deep Voice 1 & 2,Deep Voice 3使用了一个基于注意力机制的sequence-to-sequence模型，使用更紧凑的体系结构。相似与Deep Voice 3， Tacotron和Char2Wav提出了用于TTS的sequence-to-sequence神经网络结构。Tacotron是一个seq2seq的文本到声谱转化模型，使用了Griffin-Lim做声谱到声波的合成。Char2Wav预测了全局声码器的参数，并且在全局参数之上，使用了一个SampleRNN做声波生成。对比Char2Wav和Tacotron, Deep Voice 3为了加快训练，没有使用循环神经网络。Deep Voice 3使基于注意力机制的TTS系统在生产环境上可行，它避免了一些通常的注意力机制带来的问题来使系统的正确率不会降低。最后，WaveNet和SampleRNN是神经声码器的声波合成系统。在文献上也有很多高质量的手工工程声码器可以被替代，像STRAIGHT和WORLD。Deep Voice 3没有添加新的声码器，但是可以在稍作修改后，与其他不同的波形合成方法集成。

自动语音识别（ASR）数据集通常比TTS数据集大很多，但往往不那么干净，因为他们通常包含多重的扩音器和背景噪声，虽然先前的方法已经将TTS应用于ASR数据集，但是，Deep Voice 3是我们所知的最好的，第一个扩展到上千个说话的人的独立TTS系统。

Sequence-to-sequence模型编码了一个可变长度的输入到隐藏状态中，然后用解码器去处理并生成一个目标序列。注意力机制使解码器可以在生成目标序列时自适应的关注编码器的隐藏状态中不同的位置。基于注意力机制的sequence-to-sequence模型非常广泛的用于机器翻译，语音识别和文本摘要。最近的有关Deep Voice 3的注意力机制上的优化包括在训练时强制单调注意力，完全注意力的非循环结构和卷积的sequence-to-sequence模型。Deep Voice 3证明了TTS训练中单调注意的效果，TTS是单调性的一个全新领域。另外，我们证明了在推理中使用一个简单的强制单调，一个标准的注意力机制可以工作的更好。Deep Voice 3建立了一个卷积sequence-to-sequence结构，引入了一个位置编码，更好的适应input和output之间的对齐（这句不太好翻译，原文：Deep Voice 3 also builds upon the convolutional sequence-to-sequence architecture from Gehring et al. (2017) by introducing a positional encoding similar to that used in Vaswani et al. (2017), augmented with a rate adjustment to account for the mismatch between input and output domain lengths.）


##### 3.模型结构

在这一节中，我们介绍了我们的全卷积的sequence-to-sequence TTS结构。我们的结构有能力转换一个文本特征到各种声码器参数，比如梅尔频谱，线性标度对数幅谱，基频，频谱包络和非周期性参数。这些声码器参数可以被用作声波合成模型的输入。

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p1.png" /> 
    <p align=center>Deep Voice 3使用残差卷积层把text编码为每个时间点的key和value向量给基于注意力机制的解码器使用。解码器使用它们去预测对应输出音频的梅尔标度对数幅谱。（淡蓝色点状箭头描绘了推理过程中的自回归过程）将解码器的隐藏状态馈送给转换网络去预测声波合成所使用的声码器参数。更多细节详见附录A。</p>
</div>


Deep Voice 3的结构由三部分组成：

+ 编码器：一个全卷积的编码器，它将文本特征转化为内部的学习表示。
+ 解码器：一个全卷积的因果解码器，使用了注意力机制的空洞卷积，以一个自回归的方法将学习表示解码为低维的音频表示（梅尔标度谱）
+ 转换器：一个全卷积的后处理网络，从解码器的隐藏状态中预测了最终的声码器参数（取决于声码器的选择）。区别于解码器，转换器是非因果的因此可以依赖于未来的上下文信息。

要优化的总体目标函数是来自解码器（第3.5节）和转换器（第3.7节）的损失的线性组合。我们分别的将解码器和转换器用于多任务的训练，因为它在实践中可以使注意力的训练更加容易。具体来说，对梅尔谱图预测的损失可以训练注意力机制，因为注意力机制不仅和声码器参数预测相关，还和梅尔谱图的梯度预测相关。

在多人的会话场景中，训练的说话人的嵌入，在编码器，解码器和转换器中都会使用。然后，我们详细的描述了每一个组件和数据预处理。模型的超参数可在附录C的表4中得到，如下图所示:

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p2.png" /> 
</div>

###### 3.1 文本预处理

文本预处理是良好性能的关键。直接使用原始文本（具有间隔和标点的字符）在许多话语中能产生可以接受的表现。然而，一些话中的稀有词汇会产生许多错发音，或者可能跳过和重复单词。我们通过以下方式对文本进行正则化来减轻这些问题：

1. 我们将所有文本中的字母变成大写
2. 我们删除了中间所有的标点符号
3. 我们用句号或问好结束每一句话
4. 我们用特殊的分隔符替代单词当中的空格，它代表了说话人在单词中间停顿是时长。我们使用了4中不同的分隔符：
    - （i）粘在一起的词
    - （ii）标准的发音和字符间隔
    - （iii）在单词中间的短停顿
    - （iv）单词中间的长停顿

```
For example, the sentence “Either way, you should shoot very slowly,” with a long pause after “way” 
and a short pause after “shoot”, would be written as “Either way%you should shoot/very slowly%.” 
with % representing a long pause and / representing a short pause for encoding convenience.
```

###### 3.2 字符和音素的联合表示

有效的TTS系统需要拥有一个修改发音来改正通常错误（通常涉及适当的名词、外来词和特定领域的术语）的方法。通常的方法是维护一个字典将卷积块转化为语音表达。

我们的模型可以直接将字符（带有标点和间隔）转换为声学特征，学习隐含的字素到音素的模型，这种隐式转换在模型产生错误时很难改正，因此，除了字符模型，我们同样训练了一个只有音素的模型和混个字符音素模型以供选择。他们在只有字符的模型上相同，除了编码器有时候接受音素和重音嵌入而不是字符嵌入。

一个只有音素的模型需要一个将词转化为音素表达的预处理步骤（通过使用外部音素字典或单独训练的字形到音素模型），混合音素字符模型除了在音素词典中的词之外需要相同的预处理步骤。这些在词典之外中的词以字符输入，并且被允许使用隐式学习到的字素转音素模型。当使用字符音素混合模型时，在每轮训练过程中每一个词会有固定的几率被转换为它的音素表达。我们发现这会提高发音和准确率并且减少注意力错误，尤其是当推广到比训练集更长的句子中时。更重要的是，支持音素表达的模型允许使用音素字典纠正错发音，这是一个生产系统的重要需求属性。

###### 3.3 用于序列处理的卷积块

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p3.png" /> 
    <p align=center>一个卷积块包含了带有门控线性单元的1-D卷积和残差连接。这里的 c 指代了输入的维度。大小为 2 * c 的卷积输出被分成了大小相等的两部分：门控向量和输入向量</p>
</div>

通过提供足够大的接收字段，堆叠的卷积层可以利用序列中长期的上下文信息并且不会在计算中引入任何的顺序依赖。我们使用了在上图中描绘的卷积块作为主要的时序处理单元去编码文本和音频的隐藏表示。这个卷积块包含了一个1-D的卷积核，一个门控线性单元作为非线性学习，一个对输入的残差连接，和一个 $\sqrt{0.5}$
的换算系数。这个门控的线性单元为梯度流提供了线性路径，它会减轻堆叠卷积块的梯度消失问题，并且保持非线性。为了引入说话人元素，在卷积输出上，softsign函数之后加入了一个说话人的嵌入作为偏置。我们使用softsign函数因为它限制了输出的范围，同时也避免了指数为基础的非线性有时表现出的饱和问题。（这句翻译有点渣，贴上原文：We use the softsign nonlinearity because it limits the range of the output while also avoiding the saturation problem that exponentialbased nonlinearities sometimes exhibit.）我们在整个网络中用零均值和单位激活方差初始化卷积核的权值。这个卷积在结构中可以是非因果的（例如编码器）或者因果的（例如解码器）。为了保证序列的长度，输入数据在因果卷积的左边被填充了 `k - 1` 个 0 的时间点，在非因果卷积的左右各填充了 `( k - 1 ) / 2` 个 0 的时间点，其中 `k` 是奇数卷积核宽度，在卷积正则化之前会引入 Dropout 层。

###### 3.4 Encoder

编码网络（在网络结构图中描绘的）以一个嵌入层开始，将字符或者音素转换成可训练的向量表达 $h_e$。嵌入的$h_e$首先通过全连接层来从嵌入维度转化为目标维度，然后，他们通过一系列在3.3中描述的卷积块来处理，以提取文本信息的时间依赖性。最后，他们被投影到嵌入维度，来创建注意力的 key 向量 $h_k$。注意力的 value 向量从注意力 key 向量和文本嵌入中计算:
$$h_v=\sqrt{0.5} \times (h_k+h_e)$$

来共同考虑局部的信息$h_e$和长期上下文信息 $h_k$。 key 向量 $h_k$被每个注意力块使用来计算注意力权重，而最终的上下文向量由 value 向量 $h_v$的加权平均计算 （见 3.6）

###### 3.5 Decoder

解码器（在网络结构图中描绘的）通过由过去的音频帧来预测未来的一组音频帧$r$ ，以这种方式来实现自回归的音频生成。因为解码器是自回归的，所以它必须使用因果卷积块。 我们选择梅尔对数幅谱作为紧凑的音频帧表示。我们根据经验观察发现，一起解码多个帧 （例如  $r>1$）能产生更好的音频质量。解码器由多个带有ReLU去线性化的全连接层开始来处理输入的梅尔谱图（在网络结构图中表示为 "PreNet"）。然后，它跟随着一系列的因果卷积和Attention，这些卷积块生成了用于关注编码器隐藏状态的变量 Queries 。最后，一个全连接层输出了下一组音频帧$r$ 和一个二元"最终帧"预测（用来指示这句话的最后一帧是否被合成）。Dropout用在每一个Attention之前的全连接层之前，除了第一个。对梅尔谱图计算 L1 损失，并使用最终帧预测来计算二进制交叉熵损失。

###### 3.6 Attention Block

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p4.png" /> 
    <p align=center>位置编码被添加到 Query 和 Value 中，并且对应了各自的 $w_{query}$和 $w_{value}$。通过将一个较大的负值掩码添加到logit中，可以在推理中实现强迫单调性。 在训练中，注意力权重会被dropped out。</p>
</div>

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p5.png" /> 
    <p align=center>（a）为训练之前的注意力分布，（b）为训练之后但是没有强制约束，（c）在第三层和第一层使用了强制约束</p>
</div>

我们使用了点积注意力机制（如注意力机制示意图所示）。注意力机制使用了 query 向量（解码器的隐藏状态）和编码器每个时间点的 key 向量来计算注意力权重，然后输出了一个上下文向量作为计算全局权重的 Value 向量。我们引入了Attention在时间上的单调性作为诱导偏置，并观察他的经验收益。因此，我们向 key 和 query 向量添加了位置编码。位置编码 $h_p$,从以下公式中选择：

$i$为偶数时：
$$h_{p}(i) = sin(w_{s}i/10000^{k/d})$$ 

$i$为奇数时：
$$h_{p} (i)= cos(w_{s}i/10000^{k/d})$$

其中$i$是时间戳的编号，$k$是位置编码中的通道编号，$d$是位置编码中的通道总数，$w_s$
是编码的 position rate 。position rate 决定了注意力分布线的平均斜率，大体上对应与语音的速度。对于单个说话人，$w_s$
被设置为 query 中的一个，并且固定与输入时长与输出时长的时间比率（在整个数据集上计算）。对多人会话的数据集，$w_s$
使用每一个说话人嵌入的 key 和 value 向量计算（如注意力机制示意图所示），在正弦余弦函数生成正交基时，该初始化产生一个对角线形式的注意力分布，我们初始化了全连接层的权重被用来计算隐藏Attention向量对 key 和 value 的映射。位置编码被用在所有的Attention块上，一个全连接层被用于上下文向量来生成Attention块的输出。总之，位置编码提高了卷积注意力机制。

生产质量的TTS系统对注意力错误的容忍度极低。因此，除了位置编码，我们考虑额外的策略来消除重复或者跳过单词的情况。一种方法是用经典的注意力机制代替拉菲尔等人引入的单调注意机制，利用期望的训练逼近soft-monotonic注意的hard-monotonic随机译码。尽管改进的单调性，这种策略可能会产生更分散的注意力分布。在某些情况下，同时出现了几个字符，无法获得高质量的语音。我们提出了另一种注意力权重策略，它只适用于单调推理，保持训练过程没有任何限制。替代在所有input上计算softmax，**我们在间隔为8的时间窗口上采用了softmax**。最初的权重设为0，然后计算当前窗口中最高权重的序号，这种策略也在上图所示的推理中实施单调的注意，并且产生优异的语音质量。

###### 3.7 Converter

转换器将解码器的最后一个隐藏层作为输入，使用了多个非因果卷积块，然后预测提供给下游声码器的参数。不像解码器，转换器是非卷积并且非自回归的，所以可以使用解码器中未来的上下文去预测自己的输出。

转换器网络的损失函数取决于使用的声码器类型。

!> 略过声码器和4.Result

##### 5.Conclusion

我们介绍了Deep Voice 3，一个具有位置增强注意力机制的基于全卷积sequence-to-sequence声学模型的神经网络TTS系统。我们描述了sequence-to-sequence语音生成模型中常见的错误，然后表明了我们用Deep Voice 3成功的避免了这些错误。我们的模型不能直接生成声波，但可以适用于Griffin-Lim，WaveNet和WORLD做音频合成。我们还证明了我们的结构在多人会话语音合成中表现良好，通过增强讲话人的嵌入。最后，我们描述了Deep Voice 3生产系统的特点，包括文本规范化和特征表现，并通过广泛的MOS评价展示最先进的质量。未来的工作将涉及改进隐式学习的字形到音素模型，与神经声码器联合训练，以及对更干净和更大的数据集进行训练，以模拟人类声音和来自成千上万个扬声器的口音的完全可变性。

##### 附录A: Deep Voice 3 的详细结构

<div align=center>
    <img src="zh-cn/img/ch3/04-2/p6.png" /> 
    <p align=center>Deep Voice 3使用了深度残差卷积网络，把文本和音素转换为每个时间点的 key 和 value向量，将它们交给注意力机制的解码器。解码器使用他们去预测对应输出音频的梅尔对数幅谱，（浅蓝色虚线箭头描绘了推断过程中的自回归合成过程）将解码器的隐藏状态馈送到转换器网络，以生成最后声码器的参数，最终生成波形。权重归一化（Salima＆KimMA，2016）被应用到模型中的所有卷积滤波器和全连接层权重矩阵。</p>
</div>

------

### 5. SpeedySpeech: Efficient Neural Speech Synthesis

!> https://arxiv.org/abs/2008.03802

#### Abstract

最近Seq2Seq的模型在TTS合成的质量上有很大提升，但是不能同时满足训练快，推理快和高质量的音频合成。我们设计了一个student-teacher模型满足高质量的更快的实时的频谱合成，满足低计算资源占用和训练速度快的要求。我们发现self-attention层在合成高质量的音频的过程中不是必须的。我们使用简单的卷积块和残差连接在我们的student和teacher网络中并在teacher网络中使用单个Attention层。与MelGAN Vocodeer耦合，我们的模型的合成质量显著高于Tacotron2。我们的模型可以在单个GPU上进行高效的训练，可以在CPU上进行实时推断，我们提供了合成的样例和源码在GitHub:<https://github.com/janvainer/speedyspeech>

#### 1.Introduction

最近像Tactron2这样的Seq2Seq的TTS系统对语音合成的质量有显著提升，但是需要大量的训练数据和计算资源才可以完成训练。一些工作试图减少计算负担，但是依然是对训练时间，推断速度，合成质量上的tradeoff。

这篇paper考虑高效的TTS系统的设计，在保证合成质量的前提下，提高推断速度和硬件要求。我们提供了一个全卷积的，non-sequential的语音合成系统包括一个teacher和student网络，和FastSpeech类似。teacher网络是一个自回归的卷积网络用来提取音素和语音帧的对齐，student网络是一个非自回归的全卷积网络，用来编码输入的音素，预测每个音素的duration(语音帧的帧数是需要的)，基于音素和duration解码出梅尔尺度的频谱图，student网络和与训练的声码器MelGAN得到高质量的音频波形预测。

该方法在LJSpeech数据集上训练，40个小时的训练数据可以在单张8GB的显卡上进行训练。最终可以在GPU和CPU上实现高质量TTS推断。

本paper的贡献如下：
1. 我们简化了FastSpeech的teacher-student架构，提供了一个快速稳定的训练过程。我们使用一个简单且参数量更少的卷积teacher网络配合一个单层的attention layer代替了FastSpeech的Transformer结构。
2. 我们发现self-attentin结构在student网络上对于提升语音合成质量是非必须的。
3. 我们提供了一种简单的数据增强策略使得teacher网络的训练更快更稳健。
4. 我们发现我们的模型在保持高效训练和推断的前提下合成质量明显优于基线模型。

#### 2.Related Work

像Deep Voice 3和DCTTS这样的TTS系统尝试使用卷积网络代替Tacotron2的encoder-decoder架构进行加速训练。这些模型训练速度快，但是推理依然是sequential的，要比卷积网络慢。WaveRNN通过硬件加速和剪枝的方式提高sequentail推断的速度，但是训练过程是sequential的，会很慢。为了避免使用sequential inference,FastSpeech使用了Transformer架构，可以并行的生成频谱图，但是需要训练大量的attetion layer，这会导致很难训练并且会花费大量的训练时间。一些方法比如Parallel WaveNet和ClariNet提供了一些提升推断速度的方法，但是需要花费大量的计算资源训练teacher模型。

#### 3.Our Model

我们的模型的输入是音素，输出是对数尺度的Mel频谱图，首先我们讨论teacher网络，用来对齐音素和频谱图的帧，student网络使用这个对齐作为额外的监督训练合成频谱图。

##### 3.1 Teacher network – Duration extraction

teacher网络提取音素的duration,基于Deep Voice 3和DCTTS。主要包含四个部分：音素encoder,频谱图encoder,attention和decoder如下图所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p1.png" /> 
</div>

**训练该模型用来input音素和历史的帧预测下一个spectrogram帧**；生成过程通过attention跟踪音素。attention value用来对齐音素和spectrogram帧，提取出音素的durations。

**Phoneme encoder:** 音素encoder开始于embedding和一个ReLU激活的全连接层。进而，一些gated residual block,使用了逐步的空洞非因果卷积，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p2.png" /> 
</div>

block的skip connection 对所有的层的encoder的输出进行相加。替换了DCTTS中的highway block，我们使用简单的convolutional residual block 基于WaveNet，其性能没有显著的下降。

**Spectrogram encoder:** 频谱图编码器提供了对上下文频谱帧进行编码，将过去的频谱帧考虑进去。首先，每个input的频谱经过有个包含ReLU激活的全连接层。接着，拼接了一些gated residual blocks使用了逐步的空洞门控因果卷积，skip connection累加最终的output。

**Attention:** 我们使用；额dot-product attention。音素encoder的输出是key,音素encoder的输出加上音素的embedding作为value(与Deep Voice 3相似)，频谱的encoder输出作为query。key和query通过位置编码和相同的线性层进行处理，以使注意力偏向单调性。attention score根据value与query的匹配程度是value向量的加权平均。这种方式，模型学习选择和下一个频谱帧相关的音素。

**Decoder:** decoder将encoder与attention score进行加和作为input,接着进入入关个gated residual blocks使用了逐步的空洞因果卷积和一些卷基层使用了ReLU激活用来调整channel数，最后经过sigmoid预测层。

**Training:** 目标spectrograms 左移一个位置作为输入，模型强制预测下一个spectrogram帧基于input的音素和之前的帧。和Tactron2不同，模型是并行运算的。最后一层使用sigmoid激活，我们将对数尺度的梅尔频谱缩放到[0,1]。损失函数最小化目标和预测的频谱的MAE和guided attention loss用以帮助单调对齐。guided attention loss对于attention matrix：$A\in R^{N \times T}$
$$GuideAtt(A)=\frac{1}{NT}\sum^{N}_ {n=1}\sum^{T}_ {t=1}A_{n,t}W_{n,t}$$
这里的$W_{n,t}=1-exp(-\frac{(n/N-t/T)^2}{2g^2})$是惩罚矩阵，$N$是音素的个数，$T$是频谱的帧数。参数$g$控制了$A_{n,t}$对损失的贡献。

**Data augmentation:** 我们提供了3个数据增强办法增加训练的稳健性。（1） 我们在每个频谱图的像素上增加了Gaussian噪声。（2）We simulate the model outputs by
feeding the input spectrogram through the network without gradient
update in parallel mode (not sequentially). The resulting
spectrogram is slightly degraded compared to the ground-truth
spectrogram. We repeat this process multiple times to get an
approximation of a sequentially generated spectrogram. We
could simply generate the degraded spectrogram sequentially,
but using the parallel mode several times is still faster than sequential
generation. Moreover, in early stages of training, the
model is virtually unable to sequentially generate more than just
a few frames correctly. We observe that this method improves
the robustness of sequential generation drastically and the model
is able to generate long sentences well with just minor mistakes.（3）我们经input的频谱图选取随机帧进行随机值替换，这样做是为了鼓励模型在时间上使用更远的帧。否则，模型往往会过拟合到输入上的最新帧，并忽略较旧的信息，这使其不太稳定。

**Inference/duration extraction:** 和Deep Voice 3类似我们使用局部掩码attention的位置来避免音素跳跃增强单调对齐，然而我们通过teacher-forceing的方式运行推断，我们input真实的帧来避免误差累积提取更可靠的对齐。attention matrix被用来提取每个音素的duration，通过在每个时间步长计算最有可能的音素的索引并对每个索引在时间上的出现次数进行计数。

##### 3.2 Student network – Spectrogram synthesis

student网络使用teacher网络对齐预测的频谱图。提供input的音素，基于duration和全部的梅尔频谱预测每个音素的durations,如下图所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p3.png" /> 
</div>

student网路由音素encoder,duration predictor和decoder组成。三个模块由逐步的空洞残差卷积块构成，每一个bloack由1D卷积，ReLU激活和基于时间的batch normalization构成。由音素encoder生成的音素编码被馈送到duration predictor。duration predictor最终通过一个卷积和线性层 预测每个音素的duration。

音素的编码向量会基于预测的每个音素的duration进行扩展，从而使得decoder的input的大小和output的大小是一致的。类似于FastSpeech
，我们为音素编码向量添加了位置编码。我们假设网络在单个音素的上下文中而不是整个句子的上下文中区分帧位置更有益。decoder将扩展的音素编码与位置嵌入到Mel谱图的各个帧中。

student网络灵感来源于FastSpeech,但是我们将attention替换为了残差卷积块，使用了基于时间的batch normalization代替layer normalization。

**Training:** 对数尺度的Mel频谱使用MAE和structural similarity index (SSIM) losses，duration的预测使用Huber loss。我们使用从teacher网络提取得到的真实的duration训练student网络音素编码扩展。我们发现对对数尺度的Mel频谱的归一化是有必要的。不同于FastSpeech,我们detach了duration predictor和音素encoder之间得梯度流。

#### 4.Experimental Setup

这一部分介绍训练数据和训练过程的一些参数。

##### 4.1 Dataset

训练数据是LJ Speech。音素的转化使用了g2p python package:<https://github.com/Kyubyong/g2p>, We transform linear spectrograms to mel
scale and a log transformation is applied on the amplitudes(振幅）。

##### 4.2 Teacher network parameters

encoder:包含10个residual block,decoder包含14个residual block,kernel size是3，空洞卷积的dilation rate是1,3,9,27,1,3,9,27对于前8个block,dilation rate是1对于剩下的block。40个channel用于skip connection,80个channel用于gate。优化器使用的是Adam。guided attention loss和增加了位置编码加快了attention的学习，这两项措施都是针对近乎单调的注意力。

##### 4.3 Student network parameters

student网络的生成能力受teacher网络duration提取精度的影响很大。如果没有精确的音素duration,模型会不收敛。我们观察到网络深度和dilation factor必须足够高，以跨越多个单个音素。我们使用了26个encoder block,dilation 重复如下pattern:1,1,2,2,4,4; 3个duration predictor block，dilation rate为4,3,1; 34个decoder block,dilation rate 重复如下pattern:1,1,2,2,4,4,8,8；所有卷积层有128个channel。

使用了batch normalization，过程中尝试了layer normalization,channel normalization;尝试消融实验去掉SSIM loss; 对比了local position encoding,global postion encoding 和no position encoding,最终选择local position encoding。

#### 5.Evaluation

本节在合成质量，推断速度，训练速度上进行了评估。

合成质量上，指标MUSHRA，人工打分，满分100分，视觉上分为5类：“优秀”，“一般”、“好”、“差”和“坏”，我们采用的指标和MUSHRA不同，we did not use anchor recordings. We discarded any participants
who rated the reference under 90 in 8 or more cases out of 10.其评测结果如下表所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p4.png" /> 
</div>

推断速度,如下表所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p5.png" /> 
</div>

训练速度，如下表所示：

<div align=center>
    <img src="zh-cn/img/ch3/05/p6.png" /> 
</div>


#### 6.Conclusion

我们设计了一个基于卷积的TTS模型，input是音素，output是频谱图，综合考虑了训练和推断的速度以及合成语音的质量。我们的源码和合成样例可以在<https://github.com/janvainer/speedyspeech>获取。未来我们计划将我们的模型扩展到多说话人的训练数据的支持上。

------
<!-- #Transformer -->
### 6. TransformerTTS

!> https://arxiv.org/abs/1809.08895

!> https://neuraltts.github.io/transformertts/

!> https://github.com/soobinseo/Transformer-TTS

<!-- https://zhuanlan.zhihu.com/p/512240545 -->


#### 1.先来回顾一下Tactron 2

Tactron 1存在如下问题：

+ CBHG模块的去与留？：虽然在实验中发现该模块可以一定程度上减轻过拟合问题，和减少合成语音中的发音错误，但是该模块本身比较复杂，能否用其余更简单的模块替换该模块？
+ Attention出现错误对齐的现象：Tacotron中使用的Attention机制能够隐式的进行语音声学参数序列与文本语言特征序列的隐式对齐，但是由于Tacotron中使用的Attention机制没有添加任何的约束，导致模型在训练的时候可能会出现错误对齐的现象，使得合成出的语音出现部分发音段发音不清晰、漏读、重复、无法结束还有误读等问题。
+ r值如何设定？：Tacotron中一次可生成r帧梅尔谱，r可以看成一个超参数，r可以设置的大一点，这样可以加快训练速度和合成语音的速度，但是r值如果设置的过大会破坏Attention RNN隐状态的连续性，也会导致错误对齐的现象。
+ 声码器的选择：Tacotron使用Griffin-Lim作为vocoder来生成语音波形，这一过程会存在一定的信息丢失，导致合成出的语音音质有所下降（不够自然）。

因此设计了Tactron 2,结构如下图所示：

<div align=center>
    <img src="zh-cn/img/ch3/06/p1.png" /> 
</div>

+ CBHG模块的去与留？：在Tacotron2中，对于编码器部分的CBHG模块，作者采用了一个`3*Conv1D+BiLSTM`模块进行替代；对于解码器部分的CBHG模块，作者使用了Post-Net（`5*Conv1D`）和残差连接进行替代。
+ Attention出现错误对齐的现象:在Tacotron2中，作者使用了Location-sensitive Attention代替了原有的基于内容的注意力机制，前者在考虑内容信息的同时，也考虑了位置信息，这样就使得训练模型对齐的过程更加的容易。一定程度上缓解了部分合成的语音发音不清晰、漏读、重复等问题。对于Tacotron中无法在适当的时间结束而导致合成的语音末尾有静音段的问题，作者在Tacotron2中设计了一个stop token进行预测模型应该在什么时候进行停止解码操作。
+ r值如何设定？:在Tacotron2中，r值被设定为1，发现模型在一定时间内也是可以被有效训练的。猜测这归功于模型整体的复杂度下降，使得训练变得相对容易。
+ 声码器的选择:在Tacotron2中，作者选择了wavenet作为声码器替换了原先的Griffin-Lim，进一步加快了模型训练和推理的速度，因为wavenet可以直接将梅尔谱转换成原始的语音波形。（Tacotron2合成语音音质的提升貌似主要归功于Wavenet替换了原有的Griffin-Lim）。


#### 2.再看Transformer结构

<div align=center>
    <img src="zh-cn/img/ch3/06/p2.png" /> 
</div>

+ Encoder: N blocks
+ Decoder: N blocks
+ Positional embedding: input embedding + positional embedding(PE)

<div align=center>
    <img src="zh-cn/img/ch3/06/p3.png" /> 
</div>

+ (Masked) Multi-head attention: 
    - Splits each Q, K and V into 8 heads
    - Calculates attention contexts respectively
    - Concatenates 8 context vectors
+ FFN: feed forward network, 2 fully connected layers.
+ Add & Norm: residual connection and layer normalization.

#### 3.TransformerTTS

Tacotron2仍然存在的问题:

虽然Tacotron2解决了一些在Tacotron中存在的问题，但是Tacotron2和Tacotron整体结构依然一样，二者都是一个自回归模型，也就是每一次的解码操作都需要先前的解码信息，导致模型难以进行并行计算。其次，二者在编码上下文信息的时候，都使用了LSTM进行建模。理论上，LSTM可以建模长距离的上下文信息，但是实际应用上，LSTM对于建模较长距离的上下文信息能力并不强。

针对以上问题，研究人员陆续提出了相应的解决方案。基于Tacotron2模型：

+ 训练和推理过程中的效率低下；
+ 使用循环神经网络（RNN）难以建立长依赖性模型。

最终设计了TransformerTTS,如下图所示：

<div align=center>
    <img src="zh-cn/img/ch3/06/p4.png" /> 
</div>

如果对Tacotron2和Transformer比较熟悉的话，可以从上图中看出，其实Transformer TTS就是Tacotron2和Transformer的结合体。其中，一方面，Transformer TTS继承了Transformer Encoder，MHAttention，Decoder的整体架构；另一方面，Transformer TTS的Encoder Pre-net、Decoder Pre-net、Post-net、stop Linear皆来自于Tacotron2，所起的作用也都一致。换句话说，
+ 将Tacotron2: `Encoder BiLSTM ——>Transformer: Multi-head Attention（+positional encoding）`;
+ Tacotron2: `Decoder Location-sensitive Attention + LSTM ——>Transformer: Mask Multi-head Attention （+positional encoding）`;
+ 其余保持不变，就变成了Transformer TTS。

也正是Transformer相对于LSTM的优势，使得Transformer TTS解决了Tacotron2中存在的训练速度低下和难以建立长依赖性模型的问题。

其中值得一提的是，Transformer TTS保留了原始Transformer中的scaled positional encoding信息。为什么非得保留这个呢？原因就是Multi-head Attention无法对序列的时序信息进行建模。可以用下列公式表示：

<div align=center>
    <img src="zh-cn/img/ch3/06/p5.png" width=30%/> 
</div>

其中，$\alpha$是可训练的权重，使得编码器和解码器预处理网络可以学习到输入音素级别对梅尔谱帧级别的尺度适应关系。

结合Tacotron2和Transformer提出了Transformer TTS，在一定程度上解决了Tacotron2中存在的一些问题。但仍然存在一些问题：如1）在训练的时候可以并行计算，但是在推理的时候，模型依旧是自回归结构，运算无法并行化处理；2）相比于Tacotron2，位置编码导致模型无法合成任意长度的语音；3）Attention encoder-decoder依旧存在错误对齐的现象。

#### 4.TransformerTTS slide

<object data="zh-cn/img/ch3/06/(AAAI19-3124)Neural Speech Synthesis with Transformer Network.pdf" type="application/pdf" width=100% height="530px">
    <embed src="http://www.africau.edu/images/default/sample.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="zh-cn/img/ch3/06/(AAAI19-3124)Neural Speech Synthesis with Transformer Network.pdf">Download PDF</a>.</p>
    </embed>
</object>

------

### 7. Glow-TTS：A Generative Flow for Text-to-Speech via Monotonic Alignment Search

!> https://arxiv.org/abs/2005.11129

!> https://github.com/jaywalnut310/glow-tts

!> https://jaywalnut310.github.io/glow-tts-demo/index.html

<!-- https://www.cnblogs.com/Edison-zzc/p/17589837.html -->
<!-- https://blog.csdn.net/zzfive/article/details/126554154 -->
<!-- https://blog.csdn.net/qq_40168949/article/details/126948597 -->

#### 摘要

最近，文本到语音（Text-to-Speech，TTS）模型，如FastSpeech和ParaNet，被提出以并行方式从文本生成Mel频谱图（Mel-Spectrograms）。尽管并行TTS模型具有优势，但它们不能在没有自回归TTS模型作为外部对齐器的指导下进行训练。在本论文中，我们提出了Glow-TTS，一种用于并行TTS的基于流的生成模型，不需要任何外部对齐器。我们引入了单调对齐搜索（Monotonic Alignment Search，MAS），一种用于训练Glow-TTS的内部对齐搜索算法。通过利用流的特性，MAS搜索文本和语音的潜在表示之间最可能的单调对齐关系。与自回归TTS模型Tacotron 2相比，Glow-TTS在合成时获得了数量级的速度提升，并且语音质量相当，端到端仅需要1.5秒来合成一分钟的语音。我们进一步展示了我们的模型可以轻松扩展到多说话人环境。

**关键词**： Glow-TTS，生成流，文本到语音，并行TTS，单调对齐，流的特性，自回归TTS，Tacotron 2，合成语音，多说话人。

#### 1.引言

文本到语音（Text-to-Speech，TTS）是从文本生成语音的任务，基于深度学习的TTS模型已成功产生了与人类语音无法区分的自然语音。在神经网络TTS模型中，自回归模型如Tacotron 2（Shen等，2018）或Transformer TTS（Li等，2019）展现了最先进的性能。在这些自回归模型的基础上，已经取得了许多进展，以在模拟不同说话风格或不同韵律（Wang等，2018；Skerry-Ryan等，2018；Jia等，2018）方面生成多样化的语音。

尽管自回归TTS模型具有高质量，但在实时服务中直接部署端到端自回归模型存在一些困难。由于模型的合成时间与输出长度呈线性增长，生成长篇语音会导致不必要的延迟，而未经过复杂框架设计的TTS系统可能会在多个管道中传播这种延迟（Ma等，2019）。此外，大多数自回归模型在某些情况下显示出缺乏鲁棒性（Ren等，2019）。例如，当输入文本包含重复单词时，自回归TTS模型通常会产生严重的注意力错误。

为了克服自回归TTS模型的这些限制，提出了并行TTS模型，如FastSpeech。这些模型可以比自回归TTS模型更快地合成Mel频谱图。除了快速采样外，FastSpeech通过强制对齐单调性来减少极难句子的失败情况。

然而，这些并行TTS模型的优势来自于文本和语音之间良好对齐的注意力图，这些图是从它们的外部对齐器中提取的。最近提出的并行模型通过从预训练的自回归模型中提取注意力图来解决这些挑战。因此，并行TTS模型的性能严重依赖于自回归TTS模型的性能。此外，由于并行TTS模型假设在训练过程中给定了这种对齐，所以它们不能在没有外部对齐器的情况下进行训练。

在本文中，我们的目标是消除并行TTS模型训练过程中对任何外部对齐器的需求。我们提出了Glow-TTS，一种用于并行TTS的基于流的生成模型，可以内部学习自己的对齐关系。Glow-TTS的训练目标是最大化给定文本情况下语音的对数似然，并且由于生成流的性质，其采样过程是完全并行的。为了消除对其他网络的依赖，我们引入了单调对齐搜索（Monotonic Alignment Search，MAS），一种新颖的方法，仅利用文本和语音的潜在表示来搜索最可能的单调对齐关系。这种内部对齐搜索算法简化了并行TTS模型的整个训练过程，只需在两个GPU上进行3天的训练。

没有任何外部对齐器，我们的并行TTS模型可以比自回归TTS模型Tacotron 2快15.7倍地生成Mel频谱图，同时保持相当的性能。与其他TTS模型只有在Dropout操作中具有随机性的情况不同，Glow-TTS提供了多样化的语音合成。通过改变归一化流的潜在变量，我们可以控制Glow-TTS合成样本的一些特性。我们进一步展示了我们的模型可以通过只进行少量修改来扩展到多说话人环境。

#### 2.相关工作

文本到语音（TTS）模型： TTS模型是一类生成模型，从文本合成语音。TTS模型中的一个子类包括Tacotron 2（Shen等，2018）、Deep Voice 3（Ping等，2017）和Transformer TTS（Li等，2019），它们从文本生成Mel频谱图，这是一种音频的压缩表示。它们产生的语音质量媲美人类声音。另一个子类，也称为声码器（vocoder），已经开发出来将Mel频谱图转换为高保真音频波形（Shen等，2018；Van Den Oord等，2016），并具有快速合成速度（Kalchbrenner等，2018；Van Den Oord等，2017；Prenger等，2019）。研究还致力于增强TTS模型的表现力。通过辅助嵌入方法，已提出生成多样化语音的方法，可以控制一些因素，如语调和节奏（Skerry-Ryan等，2018；Wang等，2018）。此外，一些研究旨在合成不同说话人的语音（Jia等，2018；Gibiansky等，2017）。

注意： 以上是关于TTS模型的相关工作的简要介绍。由于篇幅有限，该部分只提及了一些代表性的TTS模型，而没有详细阐述各种模型的技术细节。在实际论文中，相关工作部分通常会更加详尽地讨论相关领域内的研究，并对现有的方法和技术进行全面的回顾和对比。

并行解码模型： 对于序列到序列（seq2seq）模型来说，以并行方式解码输出序列存在一些挑战。其中一个挑战是缺乏关于每个输入标记需要生成多少输出标记的信息。例如，大多数TTS数据集不包含语音中每个音素的持续时间值。另一个挑战是在并行方式下建模输出标记之间的条件依赖关系，这使得并行模型难以与自回归模型的性能相匹配。

为了应对这些挑战，已经提出了各种领域的并行解码模型。在自然语言处理领域，Gu等（2017）通过使用外部对齐器根据每个输入标记来分割输出标记，来解决这些挑战。此外，他们使用序列级知识蒸馏（Kim和Rush，2016）来减小自回归教师网络和并行模型之间的性能差距。在TTS领域，Ren等（2019）同样从自回归TTS模型Transformer TTS中提取对齐信息，并利用序列级知识蒸馏来提升性能。另一个并行TTS模型ParaNet（Peng等，2019）则利用其教师网络的软注意力图。

我们的Glow-TTS与所有先前的方法不同，它不依赖于自回归教师网络或外部对齐器，而是在并行TTS中内部学习自己的对齐关系。

基于流的生成模型： 基于流的生成模型因其优点而受到广泛关注（Hoogeboom等，2019；Durkan等，2019；Serrà等，2019）。它们通过应用一些可逆变换来估计数据的准确似然。生成流模型简单地通过训练来最大化这种似然。除了高效的密度估计外，（Dinh等，2014；2016；Kingma＆Dhariwal，2018）中提出的变换保证了快速高效的采样。Prenger等（2019）和Kim等（2018）将这些变换引入原始音频语音合成中，以克服自回归声码器WaveNet（Van Den Oord等，2016）的缓慢采样速度。他们提出的模型WaveGlow和FloWaveNet都比WaveNet更快地合成原始音频。通过应用这些变换，Glow-TTS可以在并行方式下合成给定文本的Mel频谱图。

与我们的工作并行的是Flowtron（Valle等，2020）和Flow-TTS（Miao等，2020）。Flowtron是一个基于流的TTS模型，其展示了模型的多样应用，例如通过使用流的属性进行风格转换和控制语音变化。与我们的工作的主要区别在于Flowtron使用自回归流，其目标不是快速语音合成。Flow-TTS是一个并行TTS合成器，使用基于流的解码器和带有位置编码的多头软注意力，与我们的工作相比，**我们通过估计每个输入标记的持续时间来预测硬单调对齐**。

<div align=center>
    <img src="zh-cn/img/ch3/07/p1.png" /> 
</div>

#### 3.Glow-TTS

在本节中，我们描述了一种新型的并行TTS模型，即Glow-TTS，它直接通过最大化似然来进行训练，无需其他网络的参与。Glow-TTS通过生成流的逆变换实现并行采样。在第3.1节中，我们阐述了模型的训练和推理过程，并在上图中对这些过程进行了说明。我们在第3.2节中介绍了我们的新颖对齐搜索算法，它消除了训练Glow-TTS所需的其他网络，并在第3.3节中涵盖了Glow-TTS的所有组件，包括文本编码器、持续时间预测器和基于流的解码器。

##### 3.1 Glow-TTS的训练和推断

一般来说，用于条件密度估计的归一化流将给定条件融入到每个流中，并将数据映射到具有条件流的已知先验。然而，Glow-TTS将文本条件融入到先验分布的统计量中，而不是融入到每个流中。

给定一个Mel频谱图$x$，Glow-TTS使用基于流的解码器$f_{dec}:x \rightarrow z$将Mel频谱图$x$转换为潜在变量$z$，而不需要任何文本信息，并且潜在变量$z$遵循某种各向同性高斯分布$P_Z(z|c)$。然后，文本编码器$f_{enc}$将文本条件$c$映射到文本的高级表示$h$，并将$h$投影到高斯分布的统计量$µ$和$σ$中。因此，文本序列的每个标记都有其相应的分布，潜在变量$z$的每个帧都遵循文本编码器预测的这些分布之一。

我们将潜在变量和分布之间的这种对应关系定义为对齐（alignment）$A$。因此，如果潜在变量$z_j$遵循第$i$个文本标记的预测分布$N(z_j;µ_i，σ_i)$，则我们定义$A(j) = i$。这种对齐可以解释为序列到序列建模中的硬注意力。因此，给定一个对齐$A$，我们可以计算数据的准确对数似然，如下所示：

<div align=center>
    <img src="zh-cn/img/ch3/07/p2.png" /> 
</div>

其中，$T_{mel}$是Mel频谱图的帧数。由于文本和语音是单调对齐的，我们假设对齐$A$是单调递增的。

注意：$N(z_j;µ_ {A(j)},σ_ {A(j)})$ 表示在给定文本条件和对齐的情况下，潜在变量$z$的第$j$个维度（即$z$的第$j$个元素）遵循一个高斯分布，其均值由$µ_ {A(j)}$预测，标准差由$σ_ {A(j)}$预测。这个高斯分布描述了潜在变量$z$在第$j$个维度上的不确定性，即在这个维度上的取值可能会在均值附近波动，并且波动程度由标准差决定。

我们的目标是找到最大化对数似然的参数$θ$和对齐$A$，如公式3所示。然而，寻找公式3的全局最优解在计算上是不可行的。为了解决这个难题，我们通过将目标从公式3修改为公式4，减少了参数$θ$和单调对齐$A$的搜索空间。

最大化$θ$和$A$的目标函数为：

<div align=center>
    <img src="zh-cn/img/ch3/07/p3.png" /> 
</div>

为了解决计算难题，我们将目标函数改写为：

<div align=center>
    <img src="zh-cn/img/ch3/07/p4.png" /> 
</div>

因此，训练Glow-TTS可以分解为两个连续的问题：
1. 在给定当前参数$θ$的情况下，通过公式5和6搜索与潜在表示$z$和预测分布最可能对齐的$A^{∗}$
2. 更新当前参数$θ$，使得在给定$A^{∗}$的情况下最大化对数概率$log P_X(x|c; θ, A^{∗})$。

在实践中，我们使用迭代的方法来处理这两个问题。在每个训练步骤中，我们首先找到$A^{∗}$，然后使用梯度下降更新参数$θ$。虽然我们修改后的目标不能保证得到公式3的全局解，但它仍然提供了全局解的一个很好的下界。

<div align=center>
    <img src="zh-cn/img/ch3/07/p5.png" /> 
</div>

为了解决对齐搜索问题（问题i），我们引入了一种新颖的对齐搜索算法，称为单调对齐搜索（Monotonic Alignment Search，MAS），详细描述在第3.2节。需要注意的是，Glow-TTS也可以像FastSpeech一样通过最大化$L(θ, A_{ext})$进行训练，其中$A_{ext}$是从外部对齐器提取得到的。但是，MAS完全消除了我们训练过程中对外部对齐器的依赖。

除了最大化对数似然之外，我们还训练了一个持续时间预测器$f_{dur}$，用于预测每个文本标记对应的Mel频谱图中的帧数。为了训练持续时间预测器，我们需要为每个文本标记提供一个持续时间标签。我们通过从最可能的对齐$A^{∗}$（MAS的输出）中提取这个标签来获得持续时间标签，尽管在训练初期MAS可能提供了一个较差的对齐。

根据对齐$A^{∗}$，我们可以计算每个文本标记对应的语音帧数，如公式7所示，并将帧数$d_j$作为第$j$个输入标记的持续时间标签。给定文本的高级表示$h$，我们的持续时间预测器$f_{dur}$通过均方误差（MSE）损失进行学习，如公式8所示。与FastSpeech（Ren等人，2019）一样，我们在对数域中训练$f_{dur}$的持续时间$d_j$。我们还对持续时间预测器的输入应用停止梯度运算符$sg[·]$（在反向传播中移除输入的梯度）（Oord等人，2017），以避免影响最大似然目标。因此，我们的最终目标函数如公式9所示。

<div align=center>
    <img src="zh-cn/img/ch3/07/p7.png" /> 
</div>

在推断（inference）过程中，如图1b所示，Glow-TTS通过文本编码器$f_{enc}$和持续时间预测器$f_{dur}$来预测先验分布的统计信息以及每个文本标记的持续时间。我们将这些预测的持续时间向上取整为整数，并根据每个持续时间来复制相应数量的分布。这样扩展后的分布就成为Glow-TTS在推断过程中的先验分布。然后，Glow-TTS从这个先验分布中对潜在变量$z$进行采样，并通过将逆转换$f^{−1}_ {dec}$应用到潜在变量$z$上，以并行方式合成Mel频谱图。


##### 3.2 单调对齐搜索 (Monotonic Alignment Search)

<div align=center>
    <img src="zh-cn/img/ch3/07/p6.png" /> 
    <p>(a) 对齐的示例，覆盖了文本表示$h$的所有标记。(b) 计算最大对数似然$Q$的过程。(c) 搜索最可能的对齐$A^{∗}$的过程。</p>
    <p>图2. 单调对齐搜索的图示说明。</p>
</div>

如在第3.1节中提到的，单调对齐搜索 (Monotonic Alignment Search, MAS) 旨在寻找潜在变量 $z$ 和文本表示 $h$ 之间最可能的单调对齐 $A^{∗}$。由于有大量的对齐方式可以探索，我们根据我们的假设对它们进行限制。我们假设文本和语音是单调对齐的，并且所有文本标记都在语音中有对应。因此，我们只考虑单调对齐并且不跳过任何 $h$ 元素的对齐方式。**图2a**显示了一个我们关心的可能的对齐方式。

我们在**算法1**中呈现了我们的对齐搜索算法。我们首先在 $h$ 的第 $i$ 个元素和 $z$ 的第 $j$ 个元素之间推导出递归解，并且通过使用这个推导来找到最可能的对齐 $A^{∗}$。算法1中的计算过程模拟了这个推导的过程，用于寻找最可能的单调对齐 $A^{∗}$。

<div align=center>
    <img src="zh-cn/img/ch3/07/p8.png" /> 
</div>

让 $Q_{i,j}$ 表示在 $h$ 和 $z$ 部分给定的情况下，分别到第 $i$ 个元素和第 $j$ 个元素的最大对数似然。由于我们的假设 $h_{:i}$ 应该被单调对齐，即 $z_j$ 对应着 $h_i$，而 $z_{j−1}$ 对应着 $h_{i−1} 或 $h_{i}$。这意味着 $Q_{i,j}$ 可以基于可能的部分对齐方式来计算：

<div align=center>
    <img src="zh-cn/img/ch3/07/p9.png" /> 
</div>

这个过程在**图2b**中有所说明。我们迭代地计算所有的$Q$值，直到$Q_{T_{text},T_{mel}}$。需要注意的是，$Q_{T_{text},T_{mel}}$是所有可能的单调对齐的最大对数似然值。

同样地，最可能的对齐$A^{∗}$可以通过确定在递推关系（方程式11）中哪个$Q$值较大来检索得到。因此，我们从对齐的末尾开始回溯，$A^{∗}(T_{mel}) = T_{text}$，以找到所有的$A^{∗}$值（如图2c所示）。

我们算法的时间复杂度是$O(T_{text} × T_{mel})$。虽然我们的方法难以并行化，但它在CPU上运行速度很快，无需使用GPU执行。在我们的实验中，它每次迭代所需的时间不到20毫秒，仅占总训练时间的不到2%。此外，在推断阶段我们不需要使用MAS，因为有一个持续预测器来估计每个输入标记的持续时间。

##### 3.3 模型架构

Glow-TTS的整体架构可在附录A.1中查看。我们还在附录A.2中列出了模型的配置。

**解码器 (Decoder)**：Glow-TTS的核心部分是基于流的解码器。在训练过程中，我们需要高效地将梅尔频谱图转换为潜在表示，以进行最大似然估计和内部对齐搜索。在推断过程中，需要将先验分布高效地反演成梅尔频谱图分布，以进行并行解码。因此，我们的解码器由一系列流组成，可以在并行中执行前向和反向变换。其中包括仿射耦合层 (Affine Coupling Layer) (Dinh et al., 2014; 2016)、可逆`1x1`卷积 (Invertible 1x1 Convolution) 和激活标准化 (Activation Normalization) (Kingma & Dhariwal, 2018)。

<div align=center>
    <img src="zh-cn/img/ch3/07/p10.png" /> 
</div>

具体来说，我们的解码器是多个块的堆叠，每个块包含激活标准化、可逆`1x1`卷积和仿射耦合层。我们遵循WaveGlow (Prenger et al., 2019)中的仿射耦合层架构，但不使用局部条件 (Local Conditioning) (Van Den Oord et al., 2016)。

为了提高计算效率，在解码器操作之前，我们将80通道的梅尔频谱图帧沿时间维度分成两半，并将它们组合成一个160通道的特征图。我们还修改了`1x1`卷积，以减少计算其雅可比行列式对数的耗时。在每次`1x1`卷积之前，我们沿通道维度将特征图分成40个组，并单独对它们进行`1x1`卷积。

**编码器（Encoder）**：我们遵循Transformer TTS（Li等人，2019）的编码器结构，并进行了两个小修改。我们去掉了位置编码，并将相对位置表示（Shaw等人，2018）添加到自注意力模块中。我们还为编码器预处理层（pre-net）添加了残差连接。为了估计先验分布的统计信息，我们在编码器的最后添加了一个线性层。持续时间预测器由两个带有ReLU激活函数的卷积层组成，之后是层归一化和随机失活（dropout），最后是一个投影层。我们的持续时间预测器的架构和配置与FastSpeech（Ren等人，2019）相同。

<div align=center>
    <img src="zh-cn/img/ch3/07/p11.png" /> 
</div>

#### 4.实验

为了评估我们提出的方法，我们在两个不同的数据集上进行实验。对于单一发音人的TTS，我们在广泛使用的单一女性发音人数据集LJSpeech（Ito，2017）上训练我们的模型，该数据集包含13100个短音频片段，总时长约为24小时。我们将数据集随机划分为训练集（12500个样本）、验证集（100个样本）和测试集（500个样本）。对于多发音人TTS，我们使用LibriTTS语料库（Zen等人，2019）的train-clean-100子集，其中包含约247位发音人的约54小时音频记录。我们首先剪辑了数据中所有音频片段的开头和结尾静音部分，然后过滤掉所有文本长度大于190的数据，并将其分成三个数据集进行训练（29181个样本）、验证（88个样本）和测试（442个样本）。此外，我们还收集了用于鲁棒性测试的非分布文本数据。类似于（Battenberg等人，2019），我们从书籍《哈利·波特与魔法石》的前两章中提取了227个话语。

为了将Glow-TTS与自回归TTS模型进行比较，我们将Tacotron 2设置为基准模型，它是最广泛使用的模型，并且遵循（Valle，2018）的配置。在所有实验中，我们将音素作为输入文本标记。我们在音素序列的条件下同时训练Glow-TTS和Tacotron 2。我们将除了基线预训练的文本嵌入层之外的所有参数初始化为与预训练基线相同。我们按照（Valle，2019）的配置进行Mel频谱图的训练，然后将两个模型生成的所有Mel频谱图通过预训练的声码器WaveGlow转换为原始波形。

在训练过程中，我们将Glow-TTS和Tacotron 2生成的所有Mel频谱图通过预训练的声码器WaveGlow转换为原始波形。在训练中，我们简单地将学习到的先验分布的方差$σ$设置为常数1。使用Adam优化器进行240,000次迭代训练Glow-TTS，学习率调度与（Vaswani等人，2017）中相同。这仅需在2个NVIDIA V100 GPU上进行混合精度训练的3天时间。

为了在多发音人环境中训练Glow-TTS，我们添加了**发音人嵌入**并增加了文本编码器和解码器的所有隐藏维度。通过将全局条件（Van Den Oord等人，2016）应用于解码器的所有仿射耦合层，我们将模型与发音人嵌入进行条件设置。其余设置与单发音人环境相同。在这种情况下，我们将Glow-TTS训练了480,000次以达到收敛。

#### 5.结果

##### 5.1 音频质量

我们使用Amazon Mechanical Turk（AMT）测量平均意见分数（MOS）来比较所有音频的质量，包括真实（GT）和我们通过Glow-TTS合成的样本。结果如表1所示。我们随机选择了50个句子用于评估，这些句子来自于我们的测试数据集。从真实的Mel频谱图转换得到的原始音频质量（4.19±0.07）是TTS模型的上限。我们测量了Glow-TTS在不同先验分布标准差（即温度）下的性能；其中温度为0.333时显示出最佳性能。对于任何温度，我们的Glow-TTS都展现出与强大的自回归基线Tacotron 2相当的性能。

<div align=center>
    <img src="zh-cn/img/ch3/07/p12.png" /> 
</div>

##### 5.2 采样速度

我们使用包含500个句子的测试数据集来测量TTS模型的采样速度。图3展示了我们的并行TTS模型的推理时间几乎保持在40毫秒，不论句子长度如何，而Tacotron 2的推理时间随着长度的增加而线性增加，这是由于顺序采样导致的。基于平均语音长度的推理时间，Glow-TTS比Tacotron 2快15.7倍合成Mel频谱图。

<div align=center>
    <img src="zh-cn/img/ch3/07/p13.png" /> 
</div>

我们还测量了从文本合成一分钟语音的总推理时间，这是在端到端的方式下进行的。对于这个测量，Glow-TTS合成一个长度超过5000帧的Mel频谱图，然后WaveGlow将Mel频谱图转换为一分钟语音的原始波形。总推理时间仅为1.5秒来合成一分钟的语音，其中Glow-TTS和WaveGlow的推理时间分别占总推理时间的4%和96%。也就是说，Glow-TTS的推理时间仍然只需要55毫秒来合成一个非常长的Mel频谱图，并且与声码器WaveGlow相比可以忽略不计。

##### 5.3 多样性

在样本多样性方面，大多数之前的TTS模型如Tacotron 2或FastSpeech仅在推理时依赖于Dropout。然而，由于Glow-TTS是一种基于流的生成模型，它可以在给定输入文本的情况下合成各种各样的语音。这是因为从输入文本中采样的每个潜在表示$z$都会转换成不同的Mel频谱图$f^{−1}_ {dec}(z)$。

我们可以用从标准正态分布中采样的噪声$ε$表示这个潜在表示$z ∼ N (µ, T)$，其中先验分布的均值$µ$和周期$T$如下所示：
$$z=\mu+ε \times T$$

因此，我们可以通过改变噪声$ε$和温度$T$来合成多样性的语音。下图a展示了在相同的噪声$ε$下，通过改变温度可以控制语音的音高，同时保持基频轮廓的趋势。此外，下图b展示了Glow-TTS可以通过仅改变噪声$ε$来生成具有不同基频轮廓形状的各种语音。

<div align=center>
    <img src="zh-cn/img/ch3/07/p14.png" /> 
    <p>(a) 图中展示了从相同的句子生成的语音样本，其高斯噪声$ε$相同，但是温度$T$不同，对应的音高轨迹也不同。

(b) 图中展示了从相同的句子生成的语音样本，其温度T相同（$T = 0.667$），但是高斯噪声$ε$不同，对应的音高轨迹也不同。</p>
</div>

##### 5.4 长度稳健性和可控性

长度稳健性。为了研究TTS模型处理长文本的能力，我们从书籍《哈利·波特与魔法石》中提取了一些语音片段，并对其进行合成。这些语音片段的最大长度超过了800个字符，远远大于LJ数据集中输入字符的最大长度（小于200个字符）。我们通过Google语音识别API（Google Cloud Speech-To-Text）对合成的样本进行字符错误率（CER）测量。

Figure3 b显示了与(Battenberg等，2019)类似的结果。Tacotron 2的CER在输入字符长度超过约260个时开始增长。然而，即使我们的模型在训练中没有见过这么长的文本，它也表现出对输入长度的稳健性。

除了长度稳健性的结果，我们还对特定句子的注意力错误进行了分析。结果显示在附录B.1中。
长度可控性。由于Glow-TTS与FastSpeech共享相同的持续时间预测器架构，我们的模型也能够控制输出语音的说话速率。我们通过将正标量值乘以持续时间预测器的预测结果来实现控制。我们在下图中可视化了结果。我们分别将不同的值（1.25、1.0、0.75和0.5）乘以持续时间的预测值。如下图所示，我们的模型生成了不同长度的Mel频谱图。尽管我们的模型在训练中没有见过如此极端快速或缓慢的语音，但该模型可以在不降低质量的情况下控制语音的速度。

<div align=center>
    <img src="zh-cn/img/ch3/07/p15.png" /> 
</div>

##### 5.5 多说话者 TTS

**音频质量**：我们使用类似于第5.1节的方法来测量MOS。我们随机选择了50个句子用于评估。我们比较了两种不同设置下的音频质量：GT音频和从GT梅尔频谱合成的音频。结果如表2所示。从GT梅尔频谱转换的原始音频质量（4.06±0.07）是TTS模型的上限。我们的模型在最佳配置下实现了约3.5的MOS，这表明Glow-TTS可以建模多样的说话者风格。

<div align=center>
    <img src="zh-cn/img/ch3/07/p16.png" /> 
</div>

**说话者相关的语音持续时间**：下图a显示了使用不同说话者身份生成的相同句子的梅尔频谱。由于持续时间预测器的唯一不同输入是说话者嵌入，该结果表明我们的模型会根据说话者身份不同来预测每个输入标记的持续时间。换句话说，模型根据不同的说话者身份对每个输入标记的持续时间进行个性化预测。

**语音转换**：由于我们没有将说话者身份提供给编码器，先验分布被迫与说话者身份无关。换句话说，Glow-TTS学会了解耦潜在表示$z$和说话者身份。为了调查解耦程度，我们将一个具有正确说话者身份的GT梅尔频谱转换为潜在表示，然后用不同的说话者身份进行逆转换。结果显示在下图b中。它显示转换后的语音保持了类似的基频趋势，但具有不同的音高。这表明Glow-TTS学会了将声音特征与说话者身份分离开来，使得它可以实现声音的转换。

<div align=center>
    <img src="zh-cn/img/ch3/07/p17.png" /> 
    <p> (a) 从不同说话者身份生成的相同句子的语音样本的音高轨迹。 (b) 不同说话者身份的声音转换样本的音高轨迹。 </p>
<p>图6. 从在LibriTTS数据集上训练的Glow-TTS中合成的语音样本的基频（F0）轮廓</p>
</div>

#### 6.结论

我们提出了Glow-TTS，一种新型的并行TTS模型，它提供了快速和高质量的语音合成。Glow-TTS是一种基于流的生成模型，直接通过最大似然估计进行训练，并可以并行地生成给定文本的梅尔频谱。通过引入我们的新型对齐搜索算法Monotonic Alignment Search（MAS），我们简化了整个并行TTS模型的训练过程，仅需要3天的训练时间。除了简单的训练过程外，我们还展示了Glow-TTS的额外优势，例如控制合成语音的说话速率或音高，对长篇音频的鲁棒性，以及在多说话者场景下的可扩展性。由于这些优势，我们认为Glow-TTS是现有TTS模型的一个很好的替代选择。

------

### 8. FastSpeech

!> https://arxiv.org/abs/1905.09263

<!-- https://blog.csdn.net/weixin_42721167/article/details/118226439 -->




### 9. FastSpeech v2

!> https://arxiv.org/abs/2006.04558

<!-- https://blog.csdn.net/weixin_42721167/article/details/118934862 -->





### 10. FastPitch

!> https://arxiv.org/pdf/2006.06873.pdf

<!-- https://blog.csdn.net/weixin_42721167/article/details/119783774 -->
<!-- https://zhuanlan.zhihu.com/p/420863679 -->


<!-- # Flow -->

### 11. OverFlow

!> https://arxiv.org/abs/2211.06892


### 12. SC-GlowTTS

!> https://arxiv.org/abs/2104.05557


### 13. RAD-TTS

!> https://nv-adlr.github.io/RADTTS

!> https://openreview.net/pdf?id=0NQwnnwAORi


<!-- # Diffusion-->
### 14. Diff-TTS

!> https://arxiv.org/abs/2104.01409


### 15. Grad-TTS

!> https://arxiv.org/abs/2105.06337


<!-- other -->

### 16. Align-TTS

!> https://arxiv.org/abs/2003.01950



### 17. Capacitron

!> https://arxiv.org/abs/1906.03402




### 18. Delightful TTS

!> https://arxiv.org/abs/2110.12612




### 19. Mixer_TTS/Mixer-TTS-x

!> https://arxiv.org/abs/2110.03584


