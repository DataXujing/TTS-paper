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


### 5. SpeedySpeech

!> https://arxiv.org/abs/2008.03802


<!-- #Transformer -->
### 6. TransformerTTS

!> https://arxiv.org/abs/1809.08895


### 7. Glow-TTS

!> https://arxiv.org/abs/2005.11129



### 8. FastSpeech

!> https://arxiv.org/abs/1905.09263


### 9. FastSpeech v2

!> https://arxiv.org/abs/2006.04558


### 10. FastPitch

!> https://arxiv.org/pdf/2006.06873.pdf


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


