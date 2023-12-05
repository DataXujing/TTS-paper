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



------

#### 3.Deep Voice 3: Scaling Text-to-Speech with Convolutional Sqeuence Learning

<!-- https://blog.csdn.net/qq_37175369/article/details/81476473?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-81476473-blog-113062548.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-81476473-blog-113062548.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&utm_relevant_index=9 -->



------

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


