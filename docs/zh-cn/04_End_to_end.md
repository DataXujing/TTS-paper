## End-to-End Models

完全端到端的模型，输入文本特征直接输出音频波形如下图所示：

<div align=center>
    <img src="zh-cn/img/ch4/01/p1.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch4/01/p2.png" /> 
</div>

### 1. VITS / VITS2

<!-- https://blog.csdn.net/weixin_44649780/article/details/132406232 -->

#### 1. VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

!> arxiv (2021): https://arxiv.org/abs/2106.06103

!> github: https://github.com/jaywalnut310/vits

!> demo: https://jaywalnut310.github.io/vits-demo/index.html

##### Abstract

最近已经提出了几种支持单阶段训练(single-stage training)和并行采样(parallel sampling)的端到端文本转语音 （TTS） 模型，但它们的生成质量不如两阶段 TTS 系统。我们提出了一种并行的端到端 TTS方法，该方法比当前的两阶段模型产生更自然的音频。
我们的方法采用变分推理（variational inference)，并通过归一化流(normalizing flows)和对抗性训练过程(adversarial training process)进行增强，从而提高了生成建模的表达能力。我们还提出了一个随机持续时间预测器(duration predictor)，以合成来自输入文本的不同节奏(韵律rhythms)的语音。通过对潜在变量（ latent variables）和随机持续时间预测器（stochastic duration predictor）的不确定性建模，我们的方法表达了自然的一对多关系，其中一个文本输入可以以多种方式以不同的音高和节奏进行生成。对 LJ Speech（单个说话人数据集）的主观人工评估（平均意见分数，或 MOS）表明，我们的方法优于最好的公开可用的 TTS 系统，并实现了与GT相当的MOS得分。

##### 1.Introduction

文本到语音转换 （TTS） 系统通过多个组件从给定文本合成原始语音波形。随着深度神经网络的快速发展，TTS 系统管道已简化为除文本预处理（如文本规范化和音素化）之外的两阶段生成建模。第一阶段是从预处理的文本中生成中间语音表示，例如梅尔频谱图（Shen et al.， 2018）或语言特征（Oord et al.， 2016）。第二阶段是生成以中间表示为条件的原始波形（Oord et al.， 2016;Kalchbrenner等 人，2018 年）。每个两阶段管道的模型都是独立开发的。

（相关工作的总结）基于神经网络的自回归 TTS 系统已经显示出合成真实语音的能力（Shen et al.， 2018;Li et al.， 2019），但它们的顺序生成过程使得难以充分利用现代并行处理器。为了克服这一限制并提高合成速度，已经提出了几种非自回归方法。在文本到频谱图生成步骤中，从预先训练的自回归教师网络中提取注意力图（任 et al.， 2019;Peng et al.， 2020）试图降低学习文本和频谱图之间对齐的难度。最近，基于似然的方法通过估计或学习最大化目标梅尔频谱图可能性的比对，进一步消除了对外部对准器的依赖（Zeng等人 ，2020 年;Miao et al.， 2020;Kim et al.， 2020）的同时，生成对抗网络 （GAN） （Goodfellow et al.， 2014） 已在第二阶段模型中进行了探索。基于 GAN 的前馈网络具有多个判别器，每个判别器区分不同尺度或周期的样本，可实现高质量的原始波形合成（Kumar et al.， 2019;Bińkowski et al.， 2019;Kong等 人，2020 年）。

尽管并行 TTS 系统取得了进展，但两阶段管道仍然存在问题，因为它们需要顺序训练或微调（Shen et al.， 2018;Weiss et al.， 2020） 进行高质量生成语音，其中第二阶段的模型是使用第一阶段的模型生成的样本进行训练的。此外，它们对预定义中间特征的依赖性排除了学习隐藏表示来进一步提高性能的可能性。最近，FastSpeech 2s和 EATS （Donahue et al.， 2021）等几项工作提出了有效的端到端训练方法，例如对短音频剪辑而不是整个波形进行训练，利用梅尔频谱图解码器来帮助文本表示学习，并设计专门的频谱图损失来放宽目标和生成语音之间的长度不匹配的问题。然而，尽管利用学习到的表示可能会提高性能，但它们的生成的音频的综合质量落后于两阶段系统。

在这项工作中，我们提出了一种并行的端到端 TTS 方法，该方法比当前的两阶段模型产生更自然的音频。使用变分自动编码器 （VAE） （Kingma & Welling， 2014），我们通过潜在变量连接 TTS 系统的两个模块，以实现高效的端到端学习。为了提高我们方法的表现力，以便合成高质量的语音波形，我们将归一化流应用于波形域的条件先验分布和对抗训练。除了生成细粒度的音频外，TTS 系统还必须表达一对多关系，其中文本输入可以以多种方式以不同的变化（例如，音高和持续时间）说出。为了解决一对多问题，我们还提出了一个随机持续时间预测器，以合成来自输入文本的不同节奏（韵律）的语音。通过对潜在变量和随机持续时间预测器的不确定性建模，我们的方法捕获了无法用文本表示的语音变化。

与最好的公开可用的 TTS 系统相比，我们的方法获得了更自然的语音和更高的采样效率，即 Glow-TTS（Kim et al.， 2020）和 HiFi-GAN（Kong et al.， 2020）。我们公开了演示页面和源代码。


##### 2.Method

<div align=center>
    <img src="zh-cn/img/ch4/01/p3.png" /> 
</div>

在本节中，我们将解释我们提出的方法及其架构。所提出的方法主要在前三个小节中描述：条件 VAE 公式;从变分推理得出的对齐估计;提高合成质量的对抗性训练。本节末尾介绍了整体架构。图 1（a） 和 1（b） 分别显示了我们方法的训练和推理程序。从现在开始，我们将我们的方法称为端到端文本转语音 （VITS） 的对抗性学习的变分推理。

###### 2.1 Varuational Inference

**2.1.1 Overview**

VITS可以表示为条件VAE, 其目的是最大化难以处理的数据编辑多数似然 $\log p_{\theta}(x|c)$ 的变分下限,也成为证据下限( the evidence lower bound (ELBO))：

<div align=center>
    <img src="zh-cn/img/ch4/01/p4.png" /> 
</div>

其中 $p_{\theta}(z|c)$ 表示给定条件$c$的潜在变量$z$的先验分布，$p_{\theta}(x|z)$是数据点$x$的似然函数，$q_{\phi}(z|x)$是近似的后验分布。训练损失是负的ELBO,可以看做是重建损失$-\log p_{\theta}(x|z)$和KL散度$\log q_{\phi}(z|x)-\log p_{\theta}(z|c)$的和，其中$z\sim q_{\phi}(z|x)$。

**2.1.2 Reconstruction loss(重建损失)** 

作为重建损失的目标数据点，我们使用梅尔频谱图而不是原始波形，用 $x_{mel}$ 表示。我们通过解码器将潜在变量 $z$
 上采样到波形域 $\hat{y}$,并将$\hat{y}$转换为梅尔频谱域$\hat{x}_ {mel}$ 。然后， 使用$L_1$损失计算预测和目标的mel-spectrogram作为重建损失:

$$L_{recon}=||x_{mel}-\hat{x}_ {mel}||_{1}$$ 

这可以看作是假设数据分布为拉普拉斯分布并忽略常数项的最大似然估计。我们定义了梅尔频谱图域中的重建损失，通过使用近似于人类听觉系统响应的梅尔量表来提高感知质量。请注意，从原始波形进行梅尔频谱图估计不需要可训练的参数，因为它只使用 STFT 和线性投影到梅尔尺度上。此外，估计仅在训练期间使用，而不是推理。在实践中，我们不去上采样整个潜在变量$z$,而是使用部分序列作为解码器的输入，窗口化生成器用于高效的端到端的训练 (Ren et al., 2021; Donahue et al., 2021)。


**2.1.3 KL散度**

先验编码器（prior encoder) $c$ 的输入条件由从文本中提取的音素（phonemes)$c_{text}$和音素与潜在变量之间的对齐$A$组成。
对齐是一个硬单调的注意力矩阵，其$|c_{text}| \times |z|$维度表示每个输入音素扩展为与目标语音进行时间对齐的时间。由于对齐没有GT标签，因此我们必须在每次训练迭代中估计对齐，我们将在2.2.1节中讨论。在我们的问题设置中，我们的目标是为后验编码器（posterior encoder)通过平更高分辨率的信息。因此，我们使用目标语音的线性尺度频谱图$x_{lin}$作为输入，而不是梅尔频谱图。请注意，修改后的输入并不违反变分推理的特性。KL散度的定义为：

<div align=center>
    <img src="zh-cn/img/ch4/01/p5.png" /> 
</div>

因式分解的正态分布用于参数化我们的先验和后验编码器。我们发现，提高先验分布的表达性对于生成真实样本很重要。因此，我们应用了一个归一化流 $f_{\theta}$
 （Rezende & Mohamed， 2015），它允许在因子化的正态先验分布之上，按照变量变化的规则，将简单分布逆向转换为更复杂的分布：

<div align=center>
    <img src="zh-cn/img/ch4/01/p6.png" /> 
</div>


###### 2.2 Alignment Estimation

**2.2.1 Monotonic alignment search（单调对齐搜索）**

为了估计输入文本和目标语音之间的对齐 $A$方式，我们采用了单调对齐搜索 （MAS） （Kim et al.， 2020），这是一种搜索对齐方式的方法，该方法使数据通过normalizing flow $f$ 参数化的可能性最大化：

<div align=center>
    <img src="zh-cn/img/ch4/01/p7.png" /> 
</div>

其中候选对齐被限制为单调和非跳跃，因为人类按顺序阅读文本而不会跳过任何单词。 为了找到最佳对齐方式，Kim et al. （2020） 使用了动态规划。在我们的设置中直接应用 MAS 很困难，因为我们的目标是 ELBO，而不是确切的对数似然。因此，我们重新定义 MAS 以找到使 ELBO 最大化，这归结为找到使潜在变量 $z$ 的对数似然最大化：

<div align=center>
    <img src="zh-cn/img/ch4/01/p8.png" /> 
</div>

由于等式5与等式6相似，我们可以在不修改的情况下使用原始 MAS 实现。附录 A 包括 MAS 的伪代码。

**2.2.2 Duration prediction from text**

我们可以通过对 $\sum_{j} A_{i,j}$的每一行中的所有列求和来计算每个输入token $d_i$的持续时间。正如之前的工作（Kim et al.， 2020）所提出的，持续时间可用于训练确定性的持续时间预测器，但它无法表达一个人每次都以不同的语速说话的方式。为了生成类似人类的语音节奏，我们设计了一个随机持续时间预测器，以便其样本遵循给定音素的持续时间分布。随机持续时间预测器是一种基于流的生成模型，通常通过最大似然估计进行训练。然而，直接应用最大似然估计是困难的，因为每个输入音素的持续时间是 

1） 一个离散整数，需要对其进行反量化才能使用连续归一化流，

2） 一个标量，它可以防止由于可逆性而导致的高维变换。

我们应用变分反量化（Ho et al.， 2019）和变分数据增强（Chen et al.， 2020）来解决这些问题。具体来说，我们引入了两个随机变量 $u$和 $ν$，它们与持续时间序列 $d$的时间分辨率和维度 相同，分别用于变分反量化和变分数据增强。我们将$u$的值限制为$[0,1)$，以便$d-u$的差值变成正实数序列，并且我们沿channel concat $ν$和 $d$以生成更高维度的潜在表示。我们通过近似的后验分布 
$q_{\phi}(u,v|d,c_{text})$对两个变量进行采样。结果目标是音素持续时间的对数似然的变分下限：

<div align=center>
    <img src="zh-cn/img/ch4/01/p9.png" /> 
</div>

然后，训练损失 $L_{dur}$是负变分下限。我们将停止梯度运算符（van den Oord et al.， 2017）应用于输入条件，以防止输入梯度的反向传播，以便持续时间预测器的训练不会影响其他模块的训练。

采样过程相对简单;音素持续时间是通过随机持续时间预测器的逆变换从随机噪声中采样的，然后将其转换为整数。

###### 2.3 Adversarial Training

为了在我们的学习系统中采用对抗性训练，我们添加了一个判别器 D
 ，用于区分解码器 G生成的输出和真值波形y
 。在这项工作中，我们使用了两种成功应用于语音合成的损失;用于对抗性训练的最小二乘损失函数（Mao et al., 2017) 和用于训练生成器的附加特征匹配损失（Larsen et al.， 2016）：

<div align=center>
    <img src="zh-cn/img/ch4/01/p10.png" /> 
</div>

其中$T$表示判别器的总层数，$D^l$输出判别器第$l$层的特征，$N_l$是第$l$层特征的特征数。值得注意的是，特征匹配损失可以看做是在判别器的隐藏层中测量的重建损失，建议作为VAE元素重建损失的替代方案(Larsen et al., 2016).

###### 2.4 Final Loss

将 VAE 和 GAN 训练相结合，训练我们的条件 VAE 的总损失可以表示如下：

<div align=center>
    <img src="zh-cn/img/ch4/01/p11.png" /> 
</div>

###### 2.5 Model Architecture

所提出的模型的整体架构由后验编码器、先验编码器、解码器、判别器和随机持续时间预测器组成。后验编码器和判别器仅用于训练，不用于推理。附录 B 中提供了体系结构详细信息。

**2.5.1 Posterior Encoder**

对于Posterior Encoder，我们使用 WaveGlow（Prenger et al.， 2019）和 Glow-TTS（Kim et al.， 2020）中使用的非因果 WaveNet 残差块。WaveNet 残差块由具有gated activation unit 和skip connection的膨胀卷积层组成。拼接线性投影层生成正态后验分布的均值和方差。对于多说话人的情况，我们在残差块中使用全局条件 （Oord et al.， 2016） 来添加说话人嵌入(speaker embedding)。

**2.5.2 Prior Encoder**

先验编码器由处理输入音素$c_{text}$的文本编码器 和提高先验分布灵活性的规范化流 $f_{theta}$组成。文本编码器是一种 transformer 编码器（Vaswani et al.， 2017），它使用相对位置表示（Shaw et al.， 2018）而不是绝对位置编码。$c_{text}$ 可以通过文本编码器和文本编码器上方的线性投影层获得隐藏的表示 $h_{text}$，该层产生用于构建先验分布的平均值和方差。归一化流是一堆仿射耦合层（Dinh et al.， 2017），由一堆 WaveNet 残差块组成。为简单起见，我们将规范化流设计为雅可比行列式为 1 的体积保留转换。对于 multi-speaker 设置，我们将 speaker embedding 添加到通过全局调节的归一化流中的残差块中。

**2.5.3 Decoder**

解码器本质上是 HiFi-GAN V1 生成器（Kong et al.， 2020）。它由一堆转置卷积组成，每个卷积后面都有一个多感受野融合模块 （MRF）。MRF 的输出是具有不同感受野大小的残差块的输出之和。对于多speaker设置，我们添加一个线性层来转换speaker嵌入，并将其添加到 input latent 变量 $z$中。

**2.5.4 Discriminator（判别器）**

我们遵循 HiFi-GAN 中提出的多周期判别器的判别器架构 （Kong et al.， 2020）。多周期判别器是马尔可夫基于窗口的子判别器（Kumar et al.， 2019）的混合体，每个子判别器都对输入波形的不同周期模式进行操作。

**2.5.5 Stochastic Duration Predictor**

随机持续时间预测器根据条件输入 $h_{text}$估计音素持续时间的分布。为了对随机持续时间预测器进行有效的参数化，我们将残差块与膨胀和深度可分离的卷积层堆叠在一起。我们还将神经样条流（Durkan et al.， 2019）应用于耦合层，它通过使用单调有理二次样条以可逆非线性变换的形式出现。与常用的仿射耦合层相比，神经样条流通过相似数量的参数提高了转换表现力。对于多speaker设置，我们添加一个线性层来转换说话人嵌入并将其添加到 input $h_{text}$中。

##### 3.Experiments

###### 3.1 Datasets

我们在两个不同的数据集上进行了实验。我们使用 LJ Speech数据集 （Ito， 2017） 与其他公开可用的模型进行比较和 VCTK 数据集 （Veaux et al.， 2017）用来验证我们的模型是否可以学习和表达不同的语音特征。LJ Speech 数据集由单个说话人的 13100 个简短音频剪辑组成，总时长约为 24 小时。音频格式是 16 位 PCM，采样率为 22 kHz，我们无需任何操作即可使用它。我们将数据集随机分为训练集（12500 个样本）、验证集（100 个样本）和测试集（500 个样本）。VCTK 数据集由大约 44,000 个简短的音频剪辑组成，这些剪辑由 109 名英语母语人士以各种口音说出。音频剪辑的总长度约为 44 小时。音频格式为 16 位 PCM，采样率为 44 kHz。我们将采样率降低到 22 kHz。我们将数据集随机分为训练集（43470 个样本）、验证集（100 个样本）和测试集（500 个样本）。


###### 3.2 Preprocessing

我们使用线性频谱图，该频谱图可以通过短时傅里叶变换 （STFT） 从原始波形中获得，作为后验编码器（posterior encoder）的输入。转换的 FFT 大小（FFT size）、窗口大小（window size ）和跃点大小（hop size of the transform）分别设置为 1024、1024 和 256。我们使用 80 个波段的 梅尔 尺度频谱图进行重建损失，这是通过将 梅尔滤波器组应用于线性频谱图获得的。

我们使用国际音标 （ International Phonetic Alphabet：IPA） 序列作为 prior encoder的输入。我们使用开源软件将文本序列转换为 IPA 音素序列 （Bernard， 2021），并且在 Glow-TTS 实施后，转换后的序列穿插着一个空白标记。

###### 3.3 Training

网络使用AdamW优化器（Loshchilov & Hutter，2019）进行训练，其中有 $\beta_1=0.8$, $\beta_2=0.99$
权重衰减 $\lambda=0.01$。学习率衰减为每个 epoch 因子为 $0.999^{\frac{1}{8}}$，初始学习率为 $2\times 10^{-4}$
。继以前的工作（Ren et al.， 2021;Donahue et al.， 2021）中，我们采用了窗口化生成器训练，这是一种只生成部分原始波形的方法，以减少训练期间的训练时间和内存使用。我们随机提取窗口大小为 32 的潜在表示片段馈送到解码器，而不是馈送整个潜在表示，并从GT原始波形中提取相应的音频片段作为训练目标。我们在 4 个 NVIDIA V100 GPU 上使用混合精度训练。每个 GPU 的批量大小设置为 64，模型训练多达 800k 步。


###### 3.4 Experimental Setup for Comparison

将我们的模型与最好的公开可用模型进行了比较。自回归模型 Tacotron 2 和基于流的非自回归模型 Glow-TTS 作为第一阶段模型，使用 HiFi-GAN 作为第二阶段模型。我们使用了他们的公共实现和预训练权重。由于两阶段 TTS 系统理论上可以通过顺序训练实现更高的合成质量，因此我们将微调的 HiFi-GAN 高达 100k 步长作为第一阶段模型的预测输出的声码器用来合成语音波形。我们凭经验发现，在teacher-forcing 模式下，使用 Tacotron 2 生成的梅尔频谱图对 HiFi-GAN 进行微调，导致 Tacotron 2 和 Glow-TTS 的质量比使用 Glow-TTS 生成的梅尔频谱图进行微调更好，因此我们将更好的微调 HiFi-GAN 作为 Tacotron 2 和 Glow-TTS的声码器。

由于每个模型在采样过程中都有一定程度的随机性，因此我们在整个实验过程中修复了控制每个模型随机性的超参数。Tactron 2 的 pre-net 中 dropout 的概率设置为 0.5。对于 Glow-TTS，先验分布的标准差设置为 0.333。对于 VITS，随机持续时间预测器的输入噪声的标准差设置为 0.8，我们将比例因子 0.667 乘以先验分布的标准差。


##### 4.Results

###### 4.1 Speech Synthesis Quality

我们进行了众包的 MOS 测试来评估质量。评分者聆听随机选择的音频样本，并按照从 1 到 5 的 5 分制对它们的自然度进行评分。允许评分者对每个音频样本评估一次，并且我们对所有音频剪辑进行了标准化，以避免振幅差异对分数的影响。这项工作中的所有质量评估都是以这种方式进行的。

评估结果如表 1 所示。VITS 的性能优于其他 TTS 系统，并实现了与GT相似的 MOS。VITS （DDP） 采用与 Glow-TTS 相同的确定性持续时间预测器架构，而不是随机持续时间预测器，在 MOS 评估中得分在 TTS 系统中排名第二。这些结果表明：

1） 随机持续时间预测器比确定性持续时间预测器生成更真实的音素持续时间

2） 我们的端到端训练方法是生成比其他 TTS 模型更好的样本的有效方法，即使保持相似的持续时间预测器架构。

<div align=center>
    <img src="zh-cn/img/ch4/01/p12.png" /> 
</div>

我们进行了一项消融研究以证明我们方法的有效性，包括先验编码器中的归一化流和线性标度频谱图后验输入。消融研究中的所有模型都经过了高达 300k 步的训练。结果如表 2 所示。去除先验编码器中的归一化流会导致比基线降低 1.52 MOS，这表明先验分布的灵活性会显著影响综合质量。用梅尔频谱图替换后验输入的线性尺度频谱图会导致质量下降 （-0.19 MOS），表明高分辨率信息对 VITS 在提高合成质量方面是有效的。

<div align=center>
    <img src="zh-cn/img/ch4/01/p13.png" /> 
</div>

###### 4.2 Generalization to Multi-Speaker Text-to-Speech

为了验证我们的模型可以学习和表达不同的语音特征，我们将我们的模型与 Tacotron 2、Glow-TTS 和 HiFi-GAN 进行了比较，它们显示了扩展到多说话人语音合成的能力（Jia et al.， 2018;Kim et al.， 2020;Kong等 人，2020 年）。我们在 VCTK 数据集上训练了模型。我们按照 2.5 节 中的描述为模型添加了说话人嵌入。对于 Tacotron 2，我们广播了说话人嵌入并将其与编码器输出连接，对于 Glow-TTS，我们按照之前的工作应用了全局调节。评估方法与 Section 4.1 中描述的方法相同。如表 3 所示，我们的模型实现了比其他模型更高的 MOS。这表明我们的模型以端到端的方式学习和表达各种语音特征

<div align=center>
    <img src="zh-cn/img/ch4/01/p14.png" /> 
</div>

###### 4.3 Speech Variation

我们验证了随机持续时间预测器产生多少种不同的语音长度，以及合成样本有多少种不同的语音特征。与 Valle et al. （2021） 类似，这里的所有样本都是从一句话“有多少变化？图 2（a） 显示了每个模型生成的 100 个话语的长度直方图。虽然由于确定性持续时间预测器，Glow-TTS 仅生成固定长度的话语，但我们模型中的样本遵循与 Tacotron 2 相似的长度分布。图 2（b） 显示了在多说话人设置中，我们的模型中的 5 个说话人身份中的每一个生成的 100 个话语的长度，这意味着该模型学习了与说话人相关的音素持续时间。在图3中，用YIN算法提取的10个话语的F0轮廓（De Cheveigné & Kawahara，2002）显示我们的模型生成具有不同音调和节奏的语音，而图3（d）中每个不同说话人身份生成的五个样本表明，我们的模型对每个说话人身份表达的语音长度和音高非常不同。请注意，Glow-TTS 可以通过增加先验分布的标准差来增加音高的多样性，但相反，它可能会降低合成质量。

<div align=center>
    <img src="zh-cn/img/ch4/01/p15.png" /> 
</div>

###### 4.4 Synthesis Speed

我们将模型的合成速度与并行的两阶段 TTS 系统 Glow-TTS 和 HiFi-GAN 进行了比较。我们测量了整个过程中的同步运行时间，以从 LJ Speech 数据集的测试集中随机选择 100 个句子的音素序列生成原始波形。我们使用了单个 NVIDIA V100 GPU，批量大小为 1。结果如表 4 所示。由于我们的模型不需要模块来生成预定义的中间表示，因此它的采样效率和速度大大提高。


<div align=center>
    <img src="zh-cn/img/ch4/01/p16.png" /> 
</div>


##### 5.Related Work

###### 5.1 End-to-End Text-to-Speech

目前，两阶段的神经 TTS 模型可以合成类似人类的语音（Oord et al.， 2016;Ping et al.， 2018;Shen et al.， 2018）。但是，它们通常需要使用第一阶段模型输出进行训练或微调声码器，这会导致训练和部署效率低下。他们也无法获得端到端方法的潜在好处，该方法可以使用学习的隐藏表示而不是预定义的中间特征。

最近，有人提出了单阶段的端到端 TTS 模型，以解决直接从文本生成原始波形的更具挑战性的任务，这些波形包含比梅尔频谱图更丰富的信息（例如，高频响应和相位）。FastSpeech 2s（Ren et al.， 2021）是 FastSpeech 2 的扩展，它通过采用对抗训练和辅助梅尔频谱图解码器来帮助学习文本表示，从而实现端到端并行生成。但是，为了解决一对多问题，FastSpeech 2s 必须从用作训练输入条件的语音中提取音素持续时间、音高和能量。EATS （Donahue et al.， 2021） 也采用了对抗性训练和可微分的对齐方案。为了解决生成语音和目标语音之间的长度不匹配问题，EATS 采用了动态规划计算的软动态时间扭曲损失。Wave Tacotron （Weiss et al.， 2020） 将归一化流与 Tacotron 2 相结合，以实现端到端结构，但仍然是自回归的。上述所有端到端 TTS 模型的音频质量都低于两阶段模型。

与上述端到端模型不同，通过使用条件 VAE，我们的模型 

1） 学习直接从文本合成原始波形，而无需额外的输入条件

2） 使用动态规划方法 MAS 来搜索最佳对齐而不是计算损耗

3） 并行生成样本

4） 优于最好的公开两阶段模型。


###### 5.2 Variational Autoencoders

VAEs（Kingma & Welling，2014）是使用最广泛的基于可能性的深度生成模型之一。我们对 TTS 系统采用有条件的 VAE。条件 VAE 是一种条件生成模型，其中观察到的条件调节用于生成输出的潜在变量的先验分布。在语音合成中，Hsu et al. （2019） 和 Zhang et al. （2019） 将 Tacotron 2 和 VAEs 结合起来学习语音风格和韵律。BVAE-TTS（Lee等 人，2021 年）基于双向 VAE 并行生成梅尔频谱图（Kingma等人 ，2016 年）。与之前将 VAE 应用于第一阶段模型的工作不同，我们将 VAE 应用于并行的端到端 TTS 系统。

Rezende & Mohamed （2015），Chen et al. （2017） 和Ziegler & Rush （2019）通过增强前验和后验分布的表现力和标准化流来提高VAE性能。为了提高先验分布的表示能力，我们在条件先验网络中添加了归一化流，从而生成了更真实的样本。

与我们的工作类似，Ma et al. （2019） 提出了一种条件 VAE，在非自回归神经机器翻译的条件先验网络中对流进行归一化，即 FlowSeq。然而，我们的模型可以显式地将潜在序列与源序列进行比对这一事实与 FlowSeq 不同，后者需要通过注意力机制来学习隐式比对。我们的模型通过 MAS 将潜在序列与时间对齐的源序列进行匹配，从而消除了将潜在序列转换为标准正态随机变量的负担，从而简化了归一化流的架构。

###### 5.3 Duration Prediction in Non-Autoregressive Text-to-Speech

自回归 TTS 模型（Taigman等 人，2018 年;Shen et al.， 2018;Valle et al.， 2021）通过其自回归结构和几种技巧（包括在推理和启动过程中保持辍学概率）生成具有不同节奏的多样化语音（Graves，2013）。 并行 TTS 模型（Ren et al.， 2019;Peng et al.， 2020;Kim et al.， 2020;Ren et al.， 2021;Lee et al.， 2021）一直依赖于确定性持续时间预测。这是因为并行模型必须预测目标音素持续时间或一条前馈路径中目标语音的总长度，这使得很难捕捉语音节奏的相关联合分布。在这项工作中，我们提出了一个基于流的随机持续时间预测器，它学习估计音素持续时间的联合分布，从而并行产生不同的语音节奏。


##### 6.Conclusion

在这项工作中，我们提出了一个并行的 TTS 系统 VITS，它可以以端到端的方式学习和生成。我们进一步引入了随机持续时间预测器来表达不同的语音节奏。生成的系统直接从文本合成自然发音的语音波形，而无需经过预定义的中间语音表示。我们的实验结果表明，我们的方法优于两阶段 TTS 系统，并达到接近人类的质量。我们希望所提出的方法将用于许多使用两阶段 TTS 系统的语音合成任务中，以实现性能改进并享受简化的训练程序。我们还想指出，尽管我们的方法在 TTS 系统中集成了两个独立的生成管道，但仍然存在文本预处理问题。研究语言表示的自我监督学习可能是删除文本预处理步骤的一个可能方向。我们将发布我们的源代码和预训练模型，以促进未来许多方向的研究。


##### A. Monotonic Alignment Search

我们在图 4 中展示了 MAS 的伪代码。尽管我们搜索的是使 ELBO 最大化，而不是数据的精确对数似然，但我们可以使用 Glow-TTS 的 MAS 实现，如第 2.2.1 节所述。

<div align=center>
    <img src="zh-cn/img/ch4/01/p17.png" /> 
</div>

##### B. Model Configurations

在本节中，我们主要描述了 VITS 的新添加部分，因为我们对模型的几个部分遵循了 Glow-TTS 和 HiFi-GAN 的配置：我们使用与 Glow-TTS 相同的Transformer编码器和 WaveNet 残差块;我们的解码器和多周期判别器分别与 HiFi-GAN 的生成器和多周期判别器相同，只是我们对解码器使用不同的输入维度，并附加了一个子判别器。

###### B.1. Prior Encoder and Posterior Encoder

先验编码器中的归一化流是四个仿射耦合层的堆栈，每个耦合层由四个 WaveNet 残差块组成。由于我们将仿射耦合层限制为体积保留变换，因此耦合层不会产生缩放参数。

后验编码器由 16 个 WaveNet 残差块组成，采用线性尺度对数幅度频谱图，并生成具有 192 个通道的潜在变量。

###### B.2. Decoder and Discriminator

我们的解码器的输入是从先验或后验编码器生成的潜在变量，因此解码器的输入通道大小为 192。对于解码器的最后一个卷积层，我们删除了一个 bias 参数，因为它会导致在混合精度训练期间出现不稳定的梯度尺度。

对于判别器，HiFi-GAN 使用包含五个周期（period)的$[2,3,5,7,11]$
 的子判别器的多周期判别器和包含三个子判别器的多尺度判别器。为了提高训练效率，我们只保留对原始波形进行操作的多尺度判别器的第一个子判别器，并丢弃对平均池化波形进行操作的两个子判别器。结果判别器可以看作是带有周期 
$[1,2,3,5,7,11]$的多周期判别器。


###### B.3. Stochastic Duration Predictor

图 5（a） 和 5（b） 分别显示了随机持续时间预测器的训练和推理过程。随机持续时间预测器的主要构建块是膨胀和深度可分离卷积 （DDSConv） 残差块，如图 5（c） 所示。DDSConv 块中的每个卷积层后跟一个层归一化层和 GELU 激活函数。我们选择使用扩张和深度可分离的卷积层来提高参数效率，同时保持较大的感受野大小。

<div align=center>
    <img src="zh-cn/img/ch4/01/p18.png" /> 
</div>

持续时间预测器中的后验编码器和归一化流模块是基于流的神经网络，具有相似的架构。区别在于，后验编码器将高斯噪声序列转换为两个随机变量$ν$,$u$表示近似后验分布 $q_{\phi}(u,v|d,c_{text})$，而归一化流模块将 $d-u$和$v$转换为高斯噪声序列，以表示增强和反量化数据的对数似然： $\log p_{theta}(d-u,v|v_{text})$
如 Section 2.2.2 中所述。

所有输入条件都通过条件编码器处理，每个条件编码器由两个 1x1 卷积层和一个 DDSConv 残差块组成。后验编码器（posterior encoder）和归一化流模块具有四个神经样条流耦合层。每个耦合层首先通过 DDSConv 模块处理输入和输入条件，并生成 29 个通道参数，用于构造 10 个有理二次函数。我们将所有耦合层和条件编码器的隐藏维度设置为 192。图 6（a） 和 6（b） 显示了随机持续时间预测器中使用的条件编码器和耦合层的架构。

<div align=center>
    <img src="zh-cn/img/ch4/01/p19.png" /> 
</div>

###### C. Side-by-Side Evaluation

我们通过对 50 个项目的 500 个评分，在 VITS 和GT之间进行了 7 分比较平均意见评分 （CMOS） 评估。我们的模型在 LJ Speech 和 VCTK 数据集上分别实现了 -0.106 和 -0.270 CMOS，如表 5 所示。这表明，尽管我们的模型优于最好的公开可用的 TTS 系统 Glow-TTS 和 HiFi-GAN，并且在 MOS 评估中取得了与GT相当的分数，但与我们的模型相比，评分者仍然偏爱GT。

<div align=center>
    <img src="zh-cn/img/ch4/01/p20.png" /> 
</div>

###### D. Voice Conversion

在多说话人设置中，我们不会在文本编码器中提供说话人身份，这使得从文本编码器估计的潜在变量学习是与说话人无关的表示。使用与说话人无关的表示形式，我们可以将一个说话人的录音转换为另一个说话人的声音。对于给定的说话人身份 
$s$和说话人的话语，我们可以从相应的话语音频中获得线性频谱图 $x_{lin}$ 。我们可以通过后验编码器和前验编码器中的归一化流将$x_{lin}$转换为 独立于说话人的表示$e$：

<div align=center>
    <img src="zh-cn/img/ch4/01/p21.png" /> 
</div>

然后，我们可以通过归一化流的逆变换$f^{-1}_ {\theta}$和解码器 $G$，从表示 $e$
中合成目标说话人$\hat{s}$ 的声音$\hat{y}$：

$$\hat{y}=G(f^{-1}_ {\theta}(e|\hat{s})|\hat{s})$$

学习独立于说话人的表示并将其用于语音转换可以看作是 Glow-TTS 中提出的语音转换方法的扩展。我们的语音转换方法提供原始波形，而不是 Glow-TTS 中的梅尔频谱图。语音转换结果如图 7 所示。它显示了具有不同音高级别的音高轨道的类似趋势。

<div align=center>
    <img src="zh-cn/img/ch4/01/p22.png" /> 
</div>

------

#### VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design

!> arxiv(2023): https://arxiv.org/abs/2307.16430

!> github: https://github.com/daniilrobnikov/vits2

!> demo: https://vits-2.github.io/demo/


##### Abstract

单阶段文本转语音模型(Single-stage text-to-speech models)最近得到了积极的研究，其结果优于两阶段系统。虽然之前的单阶段模型已经取得了很大的进步，但从其间歇性不自然性、计算效率、对音素转换的强依赖性(intermittent unnaturalness, computational efficiency, and strong dependence on phoneme conversion)等方面还有改进的空间。在这项工作中，我们介绍了 VITS2，这是一种单阶段文本转语音模型，它通过改进之前工作的几个方面来有效地合成更自然的语音。我们提出了改进的模型架构和训练机制，所提出的方法在提高自然度、多说话人模型中语音特征的相似性以及训练和推理的效率方面是有效的。此外我们的方法可以显着减少以前工作中对音素转换的强烈依赖，是完全端到端的单阶段方法。

##### 1.Introduction

基于深度神经网络的文本转语音技术的最新发展取得了重大进展。基于深度神经网络的文本转语音是一种从输入文本生成相应原始波形的方法;它有几个有趣的功能，通常使文本转语音任务具有挑战性。快速回顾一下这些功能后发现，文本转语音任务涉及将文本（一个不连续的特征）转换为连续的波形。输入和输出的时间步长差为数百倍，它们之间的对齐必须非常精确才能合成高质量的语音音频。此外，输入文本中不存在的韵律和说话人特征应该自然地表达出来，这是一个一对多的问题，其中文本输入可以用多种方式说出来。使合成高质量语音具有挑战性的另一个因素是，人类在收听音频时专注于单个组件;因此，即使构成整个音频的数十万个信号中有一小部分是不自然的，人类也可以轻松感知它们。效率是使任务变得困难的另一个因素。合成的音频具有很高的时间分辨率，通常每秒包含超过 20,000 个数据，需要高效的采样方法。 [语音合成的任务的挑战]

由于文本转语音任务功能，该解决方案也可以很复杂。以前的工作通过将从输入文本生成波形的过程分为两个级联阶段来解决这些问题。一种流行的方法包括在第一阶段 [1， 2， 3， 4， 5， 6， 7] 中从输入文本中生成中间语音表示，例如梅尔频谱图或语言特征，然后在第二阶段 [8， 9， 10， 11， 12， 13， 14， 15] 中生成基于这些中间表示的原始波形.两阶段管道系统具有简化每个模型和促进训练的优势;但是，它们也具有以下限制。1） 误差从第一级传播到第二级。2） 它不是利用模型内部学习到的表示，而是通过人类定义的特征（如梅尔频谱图或语言特征）进行中介。3） 生成中间特征所需的计算。最近，为了解决这些限制，人们积极研究了直接从输入文本生成波形的单级模型 [16:End-to-end adversarial text-to-speech(2021)， 7：FastSpeech 2s(2021)， 17:VITS， 18:JETS(2022)]。单阶段模型不仅优于两阶段管道系统，而且还显示出生成与人类几乎无法区分的高质量语音的能力。[通常的解决方案]

虽然以前的工作 VITS在单阶段方法上取得了巨大成功，但VITS存在以下问题：间歇性不自然、持续时间预测器效率低、缓解对齐和持续时间建模限制的复杂输入格式（使用空白标记）、多说话人模型中说话人相似度不足、训练缓慢以及对音素转换的强烈依赖。在这项工作中，我们提供了解决这些问题的方法。我们提出了一个通过对抗性学习训练的随机持续时间预测器，通过利用 transformer block和说话人条件的文本编码器来更好地模拟多个说话人的特征，从而改进归一化流(normalizing flow)。我们证实所提出的方法提高了质量和效率。此外，我们通过使用标准化文本作为模型的输入的实验表明，这些方法减少了对音素转换的依赖。因此，这些方法更接近于完全端到端的单阶段方法。 [VITS的缺点和VITS2是如何解决的]


##### 2.Method

在本节中，我们描述了四个小节的改进：持续时间预测、带有归一化流的增强变分自动编码器、对齐搜索和说话人条件文本编码器。我们提出了一种方法，使用对抗性学习来训练持续时间预测器，以在训练和合成中都高效地合成自然语音。我们的模型基本上使用之前工作 [4：Glow-TTS， 17:VITS] 中提出的单调对齐搜索 （MAS） 来学习对齐，我们进一步建议进行修改以提高质量。此外，我们提出了一种通过将 transformer 块引入归一化流来提高自然度的方法，从而可以在转换分布时捕获长期依赖关系。此外，我们修改了speaker条件，以提高多speaker模型中的speaker相似性。


<div align=center>
    <img src="zh-cn/img/ch4/02/p1.png" /> 
</div>

###### 2.1 Stochastic Duration Predictor with Time Step-wise Conditional Discriminator

VITS表明，与确定性方法相比，基于流的随机持续时间预测器在提高合成语音的自然性方面更有效。它显示出很好的结果;但是，基于流的方法需要相对更多的计算和一些复杂的技术。我们提出了一个具有对抗性学习的随机持续时间预测器，与VITS的工作相比，它以更高的训练和合成效率合成更自然的语音。提出的持续时间预测器和判别器的概述如图 1（a） 所示。我们应用对抗性学习来训练持续时间预测器，使用条件判别器训练持续时间预测器，该判别器被馈送与生成器相同的输入，以适当地区分预测的持续时间。我们使用文本 的隐藏表示$h_{text}$和高斯噪声 $z_d$作为生成器 $G$的输入，输出为$\hat{d}$和使用 MAS 在对数刻度中的输出持续时间$d$，用作判别器 $D$的输入 。一般生成对抗网络的判别器的输入是固定长度的，而每个输入标记的持续时间的预测以及input的token的长度都是变化的。为了正确区分可变长度的输入，我们提出了一个时间逐步判别器，它区分所有token的每个预测持续时间。我们使用两种类型的损失;用于对抗性学习的最小二乘损失函数 [19] 和均方误差损失函数：

<div align=center>
    <img src="zh-cn/img/ch4/02/p2.png" /> 
</div>

我们提出的持续时间预测器和训练机制允许短步学习持续时间，并且持续时间预测器作为最后一个训练步骤单独训练，这减少了训练的整体计算时间。

###### 2.2 Monotonic Alignment Search with Gaussian Noise

遵循前面的工作 [4， 17]，我们将 MAS 引入我们的模型以学习对齐。该算法产生文本和音频之间的对齐，在所有可能的单调对齐中具有最高的概率，并且该模型经过训练以最大化其概率。该方法有效;  `However, after searching and optimizing a particular alignment, it is limited in exploration to search for other alignments that are more appropriate`。为了缓解这种情况，我们在计算的概率中添加了一个小的高斯噪声。这为模型提供了搜索其他对齐的额外机会。我们只在训练开始时添加此噪声，因为 MAS 使模型能够快速学习对齐。参考之前的工作 [4]，其中详细描述了该算法。 为前向操作中所有可能的位置计算的最大对数似然表示为$Q$。我们在计算 
$Q$值中添加较小的高斯噪声$ϵ$

<div align=center>
    <img src="zh-cn/img/ch4/02/p3.png" /> 
</div>

$z$是从normalizing flow中变换得到的潜在变量，$ϵ$是来自于标准正态分布的噪声，标准差是$P$, 噪声的尺度从0.001每一步递减$2\times 10^{-6}$。

###### 2.3 Normalizing Flows using Transformer Block

VITS展示了变分自编码器通过归一化流增强的合成高质量语音音频的能力。归一化的流（normalizing flows)包括卷积块，卷积块是捕获相邻数据模式并使模型能够合成高质量语音的有效结构。在转换分布时，捕获长期依赖关系的能力可能至关重要，因为语音的每个部分都与不相邻的其他部分相关。尽管卷积块可以有效地捕获相邻模式，但由于其感受野的限制，它在捕获长期依赖关系方面存在劣势。因此，我们在归一化的流中添加一个带有残差连接的小型 transformer 模块，以能够捕获长期依赖关系，如图 1（b） 所示。

Figure 2 shows an actual attention score map and the receptive field of the convolution block. We can confirm that the transformer block collects information at various positions when transforming the distribution, which is impossible with the receptive field.

<div align=center>
    <img src="zh-cn/img/ch4/02/p4.png" /> 
</div>

###### 2.4 Speaker-Conditioned Text Encoder

由于多说话人模型是根据说话人的状态用一个模型合成多个特征的语音，因此表达每个说话人的个性语音特征不仅是自然的，也是一个重要的质量因素。以前的工作表明，单阶段的模型可以对多说话人进行高质量的建模。考虑到某些特征，例如说话人的特定发音和语调，会显着影响每个说话人的语音特征的表达，但这些特征不包含在输入文本中，我们设计了一个以说话人信息为条件的文本编码器，通过在编码输入文本时学习特征来更好地模拟每个说话人的各种语音特征。我们在文本编码器的第三个 transformer 块上输入speaker vector，如图 1（c） 所示。


##### 3.Experiments

我们在两个不同的数据集上进行了实验。我们使用 LJ Speech数据集 [20] 来确认自然度的提高，并使用 VCTK 数据集 [21] 来验证我们的模型是否可以更好地再现说话人特征。LJ Speech 数据集由单个说话人的 13100 个简短音频剪辑组成，总时长约为 24 小时。音频格式是 16 位 PCM，采样率为 22.05 kHz，我们无需任何操作即可使用它。我们将数据集随机分为训练集（12500 个样本）、验证集（100 个样本）和测试集（500 个样本）。VCTK 数据集由大约 44,000 个简短的音频剪辑组成，这些剪辑由 109 名英语母语人士以各种口音说出。音频剪辑的总长度约为 44 小时。音频格式为 16 位 PCM，采样率为 44.1 kHz。我们将采样率降低到 22.05 kHz。我们将数据集随机分为训练集（43470 个样本）、验证集（100 个样本）和测试集（500 个样本）。

我们使用 80 个波段梅尔尺度频谱图来计算重建损失。与之前的工作 [17] 相比，我们使用相同的频谱图作为后验编码器的输入。快速傅里叶变换、窗口和跃点大小分别设置为 1024、1024 和 256(he fast Fourier transform, window, and hop sizes were set to 1024, 1024, and 256, respectively)。

我们使用音素序列和归一化文本作为模型的输入进行了实验。我们使用开源软件 [22:Phonemizer <https://github.com/bootphon/phonemizer>] 将文本序列转换为国际音标序列，并将序列提供给文本编码器。与之前的工作 [17] 相比，我们没有使用空白标记。对于规范化文本的实验，我们使用开源软件 [23] 通过简单的规则对输入文本进行规范化，并将其提供给文本编码器。

网络使用AdamW优化器（Loshchilov & Hutter，2019）进行训练，其中有 $\beta_1=0.8$, $\beta_2=0.99$
权重衰减 $\lambda=0.01$。学习率衰减为每个 epoch 因子为 $0.999^{\frac{1}{8}}$，初始学习率为 $2\times 10^{-4}$
。继以前的工作（Ren et al.， 2021;Donahue et al.， 2021）中，我们采用了窗口化生成器训练，这是一种只生成部分原始波形的方法，以减少训练期间的训练时间和内存使用。我们随机提取窗口大小为 32 的潜在表示片段馈送到解码器，而不是馈送整个潜在表示，并从GT原始波形中提取相应的音频片段作为训练目标。我们在 4 个 NVIDIA V100 GPU 上使用混合精度训练。每个 GPU 的批量大小设置为 256，生成波形的网络和持续时间预测器分别被训练高达 800k 和 30k。

##### 4.Results

###### 4.1 Evaluation of Naturalness

为了确认所提出的模型可以合成自然的语音，进行了众包平均意见评分 （MOS） 测试。评分者在听取了从测试集中随机选择的音频样本后，以 1 到 5 的 5 分制对他们的自然度进行了评分。考虑到之前的工作 [ 17] 已经证明了与人类录音相似的质量，我们还进行了比较平均意见分数 （CMOS） 测试，该测试适用于通过直接比较来评估高质量样本。在聆听了从测试集中随机选择的音频样本后，评分者以 3 到 -3 的 7 分制对他们的相对偏好进行了自然度评分。
评分者被允许对每个音频样本进行评估一次。所有音频样本都经过归一化，以避免振幅差异的影响。我们使用了之前工作 [17] 的官方实现和预训练权重作为比较模型。评估结果见表 1 和表 2（a）。我们的方法与以前的工作 [17] 之间的 MOS 差异为 0.09，CMOS 和置信区间分别为 0.201 和 
±0.105。结果表明，我们的方法显著提高了合成语音的质量。此外，我们用方法[18]评估了CMOS，该方法使用不同的结构和训练机制显示出良好的性能。为了进行评估，我们使用官方实现和预训练权重生成了样本。评估的 CMOS 和置信区间分别为 0.176 和 
±0.125，表明我们的方法明显优于该方法。

<div align=center>
    <img src="zh-cn/img/ch4/02/p5.png" /> 
</div>

###### 4.2 Ablation Studies

消融研究以验证所提出的方法的有效性。为了验证用对抗性学习训练的随机持续时间预测器的有效性，它被替换为具有相同结构并使用 L2 损失训练的确定性持续时间预测器。确定性持续时间预测器被训练到与之前工作相同的步骤 [17]。为了验证对齐搜索中使用的噪声调度的有效性，在没有噪声的情况下对模型进行了训练。我们在规范化流程中没有 transformer block 的情况下训练了模型，以验证其有效性。评估结果如表 1 所示。确定性持续时间预测器、无噪声的对齐搜索和无变压器模块的归一化流的消融研究的 MOS 差异分别为 0.14、0.15 和 0.06。由于我们不使用空白标记和线性频谱图，计算效率会得到提高，并且删除一些提出的方法与以前的工作相比性能较低 [17]。结果表明，所提方法对提高质量具有较好效果。


###### 4.3 Evaluation of Speaker Similarity

为了确认多说话人模型中说话人相似度的提高，通过众包进行了与之前工作 [25] 类似的相似性 MOS 测试。在测试中，从测试集中随机采样的人类录制音频作为参考，评分者以 1 到 5 的五分制对参考音频和相应的合成音频之间的相似性进行评分。与第 4.1 节一样，允许评分者对每个音频样本评估一次，并且音频样本被标准化。评估结果见表 2（b）。VITS2 的评分比之前的工作高 0.2 MOS [17]，这表明我们的方法在建模多个说话人时可以提高说话人相似度的有效性。

##### 4.4 Reduced dependency on the phoneme conversion

以前的工作 [17， 26] 在单阶段方法中显示出良好的性能，但仍然强烈依赖音素转换。因为标准化文本不会告知其实际发音，所以学习准确的发音具有挑战性。目前，它是实现完全端到端的单阶段语音合成的关键障碍。我们提出，我们的方法通过可理解性测试显着改善了这个问题。在使用 Google 的自动语音识别 API 在测试集中转录 500 个合成音频后，我们以真实文本为参考计算了字符错误率 （CER）。我们将以下四个模型的结果与基本事实进行了比较：使用音素序列的模型、使用标准化文本的模型、使用音素序列的先前工作以及使用标准化文本的先前工作。表 3 显示了比较，它证实了所提出的模型不仅优于以前的工作，而且我们使用标准化文本的模型的性能与使用音素序列的模型的性能相当。它展示了数据驱动、完全端到端方法的可能性。

<div align=center>
    <img src="zh-cn/img/ch4/02/p6.png" /> 
</div>

###### 4.5 Comparison of Synthesis and Training Speed

我们将模型的合成和训练速度与之前的工作进行了比较 [17]。我们测量了整个过程中的同步运行时间，从 LJ Speech 数据集中随机选择的 500 个句子的输入序列中生成原始波形。我们使用了单个 NVIDIA V100 GPU，批量大小为 1。我们还测量并平均了在 4 个 NVIDIA V100 GPU 上 5 个 epoch 中每个步骤的训练计算所用时间。表 4 显示了结果。由于持续时间预测器效率更高，可以单独训练，并且输入序列比以前的工作更短，因此其训练和合成速度得到了提高;改进分别为 20.5% 和 22.7%。


##### 5.Conclusion

我们提出了 VITS2，这是一种单阶段文本转语音模型，可以有效地合成更自然的语音。我们通过将对抗性学习引入持续时间预测器，提高了训练和推理的效率和自然性。将 transformer 块添加到规范化流中，以在转换分布时捕获长期依赖关系。通过将高斯噪声合并到比对搜索中，提高了合成质量。对音素转换的依赖性显著减少，这在实现完全端到端的单阶段语音合成方面构成了挑战。测试结果还显示，整体清晰度得到了提高。我们通过实验、质量评估和计算速度测量证明了我们提出的方法的有效性。语音合成领域仍然存在各种必须解决的问题，我们希望我们的工作可以成为未来研究的基础。

------

### 2. Char2Wav:END-TO-END SPEECH SYNTHESIS

<!-- 2017 -->
<!-- https://blog.ailemon.net/2020/08/17/paper-share-char2wav-tts/ -->

<!-- https://blog.csdn.net/qq_40168949/article/details/96437321 -->

<!-- https://blog.csdn.net/weixin_34417635/article/details/92499763 -->

!> 2017 ICLR 作者中有 Yoshua Bengio，包括SampleRNN（2017）,WaveNet(2016)[两个都是声码器]，这些方法从现在看可能不是最优的方式，但是其paper也是值得我们去学习的。

!> paper: https://openreview.net/forum?id=B1VWyySKx

!> github: https://github.com/sotelo/parrot

!> demo: http://josesotelo.com/speechsynthesis

!> arxiv(SampleRNN 2017): https://arxiv.org/abs/1612.07837

!> arxiv(WaveNet 2016): https://arxiv.org/abs/1609.03499


#### Abstract

Char2Wav是端到端的语音合成模型，包括两个模块： reader和neural vocoder。 reader是一个带有attention的encoder-decoder模型，encoder是一个双向的RNN,输入为text或phonemes,decoder是带有attention的RNN用来产生声码器声学特征。neural vocoder是指 SampleRNN 的一种条件式的扩展，其可以根据中间表征（intermediate representations）生成原始的声波样本（Neural
vocoder refers to a conditional extension of SampleRNN which generates raw
waveform samples from intermediate representations）。与用于语音合成的传统模型不同，Char2Wav 可以学习直接根据文本生成音频。

#### 1.Introduction

语音合成的主要任务包括将文本映射为音频信号。语音合成有两个主要目标：**可理解性**和**自然度**。可理解性是指合成音频的清晰度，特别是听话人能够在多大程度上提取出原信息。自然度则描述了无法被可理解性直接获取的信息，比如听的整体容易程度、全局的风格一致性、地域或语言层面的微妙差异等等。

传统的语音合成方法是将这个任务分成两个阶段来完成的。第一个阶段被称为前端（frontend）是将文本转换为语言特征，这些特征通常包括音素、音节、词、短语和句子层面的特征（Zen, 2006; Zen et al., 2013; van den Oord et al., 2016）。第二个阶段被称为后端（backend），以前端所生成的语言特征为输入来生成对应的声音。WaveNet（van den Oord et al., 2016）就是一种可实现高质量的「神经后端（neural backend）」的方法。要更加详细地了解传统的语音合成模型，我们推荐参阅 Taylor ( 2009 ) 。

定义好的语言特征通常需要耗费大量时间，而且不同的语言也各有不同。在本论文中，我们将前端和后端整合到了一起，可以通过端到端的方式学习整个过程。这个流程消除了对专业语言学知识的需求，这就移除了在为新语言创建合成器时所面临的一个主要瓶颈。我们使用了一个强大的模型来从数据中学习这种信息。


#### 2.Related Work

基于注意力的模型之前已经在机器翻译（Cho et al., 2014; Bahdanau et al., 2015）、语音识别（Chorowski et al., 2015; Chan et al., 2016）和计算机视觉（Xu et al. 2015）等领域得到了应用。我们的工作受到了 Alex Graves ( Graves, 2013; 2015 ) 的工作很大的影响。在一个客座讲座中，Graves 展示了一个使用了一种注意机制的语音合成模型，这是他之前在手写生成方面的研究成果的延伸。不幸的是，这个延伸工作没有被发表出来，所以我们不能将我们的方法和他的成果进行直接的比较。但是，他的结果给了我们关键的启发，我们也希望我们的成果能有助于端到端语音合成的进一步发展。

#### 3.Model Description

##### 3.1 reader

我们采用了 Chorowski et al.( 2015 ) 的标记法。一个基于注意力的循环序列生成器（ARSG/attention-based recurrent sequence generator）是指一种基于一个输入序列 $X$ 生成一个序列 $Y= ( y_1, . . . , y_T ) $的循环神经网络。$X$被一个编码器预处理输出一个序列 $h = ( h_1, . . . , h_L )$ 。在本研究中，输出 $Y$是一个声学特征的序列，而 $X$ 则是文本或要被生成的音素序列。此外，该编码器是一个双向循环网络。

<div align=center>
    <img src="zh-cn/img/ch4/03/p1.png" /> 
</div>

##### 3.2 Neural Vocoder

语音合成的质量会受到vocoder的限制，为了确保高质量的输出，用SampleRNN- a learned parametric neural module.
SmapleRNN用于建模extremely长时依赖性，其中的垂直结构用于捕捉序列不同时刻的动态。捕捉长的audio step（词级别）和短的audio step之间的长相关很重要。
使用conditional version model学习vocoder 特征序列和对应audio sample之间的映射，每一个时刻的输出取决于它的vocoder特征和过去时刻的输出。


!> 这里补充一些SampleRCNN：https://arxiv.org/abs/1612.07837 的介绍

SampleRNN是“无条件的端到端的神经音频生成模型 An Unconditional End-to-End Neural Audio Generation Model”，由Soroush Mehri，Kundan Kumar，Ishaan Gulrajani，Rithesh Kumar，Shubham Jain，Jose Manuel Rodriguez Sotelo，Aaron Courville和Yoshua Bengio发明。正如描述所说，这个模型就像WaveNet一样，是一个用于逐个样本（sample-by-sample）生成音频的端到端模型。

与基于卷积层的WaveNet不同，SampleRNN基于循环层。据作者说，GRU单元效果最好，但可以使用任何类型的RNN单元。这些层layers被分组为“tiers”。这些tiers构成了一个层次结构：在每个tier中，单个时间步的输出，通过学习的上采样upsampling，从较低的tier调整几个时间步。因此，不同的tiers以不同的时钟速率运行，这意味着它们可以学习不同的抽象级别。例如，在最低tier中，一个时间步长对应于一个样本，而在最高层中，一个时间步长可以对应于四分之一秒，其可以包括一个或甚至几个单音notes。这样我们可以从表示作为一系列单音notes到一系列原始样本samples。此外，每个tier中的每个时间步长都由在同一tier中的先前时间步长中生成的样本来调节。最低tier不是recurrent的，而是一个自回归autoregressive的多层感知器MLP（multi-layer perceptron），由几个最后的样本和更高tier的输出决定。

让我们看一个例子（原始论文中的图例）：

<div align=center>
    <img src="zh-cn/img/ch4/03/p2.png" /> 
</div>

这里我们有3个tier。最低的一个是MLP，它将最后一个样本作为输入，从中间tier作为上采样输出。中间tier从最低tier开始，调整4个时间步长，并将最后生成的样本和最高tier的上采样输出作为输入。最高tier 调整了距离中间tier 4个时间步长，并将最后生成的16个样本作为输入，这是该tier的前一个时间步长生成的样本总数。

正如我们所看到的，这个模型很容易理解，因为它包含由上采样层分离的众所周知的recurrent layers，它们是普通的线性变换，以及一个多层感知器MLP。它的计算成本也很便宜，因为只有最低tier 在样本级别运行，而在较高级别会运行较慢，因此它们对总体计算时间的消耗较小。相比之下，WaveNet需要为每个样本计算每个layer的输出。


#### 4.Training Details

先分别训练reader和vocoder，reader的输出目标是WORLD特征，vocoder的输入是WORLD特征。最后，end-to-end的fine-tune整个模型。

#### 5.Results

我们目前不提供对结果的全面定量分析(估计是效果不好)，仅提供了一些demo的样例，下图提供了我们的模型的合成的语音以及与文本的对齐的可视化。

<div align=center>
    <img src="zh-cn/img/ch4/03/p3.png" /> 
</div>

------


### 3. ClariNet:ParallelWave Generation in End-to-End Text-to-Speech

<!-- 百度 2019 -->

!> arxiv(ICLR 2019): https://arxiv.org/abs/1807.07281

!> demo: https://clarinet-demo.github.io/

!> github(非官方): https://github.com/ksw0306/ClariNet

!> arxiv(Parallel WaveNet 2018): https://arxiv.org/abs/1711.10433

!> 这篇paper看的真是一知半解！我们的目的也是仅了解这些paper中的方法，在真是的产品中大概率效果不好！


#### Abstract

在这项工作中，我们提出了一种通过WaveNet生成parallel wave的新解决方案。与parallel WaveNet (van den Oord et al., 2018)相比，we distill a Gaussian inverse autoregressive flow from the autoregressive WaveNet by minimizing a regularized KL divergence between their highly-peaked output distributions. 我们的方法以封闭形式计算 KL 散度，这简化了训练算法并提供了非常有效的蒸馏。此外，我们还引入了第一个用于语音合成的text到wave的神经网络架构,它是完全卷积的，可以从头开始进行快速的端到端训练。它的性能明显优于之前将文本到频谱图模型连接到单独训练的 WaveNet 的管道（Ping et al.， 2018）。我们还成功蒸馏了一个parallel waveform合成器，它以这个端到端的模型的hidden represent 作为条件（We also successfully distill a parallel waveform synthesizer conditioned on the hidden representation in this end-to-end model. ）。

#### 1.Inroduction

语音合成，也称为文本转语音 （TTS），传统上是通过复杂的多阶段手动工程实现的 （Taylor， 2009）。TTS 深度学习方法最近的成功导致了高保真语音合成（van den Oord et al.， 2016a），更简单的“端到端”pipeline（Sotelo et al.， 2017;Wang et al.， 2017;Ping et al.， 2018）和再现数千种不同声音的单个 TTS 模型（Ping et al.， 2018）。

WaveNet （van den Oord et al.， 2016a） 是一种用于波形合成的自回归生成模型。它以原始音频的非常高的时间分辨率（例如，每秒 24,000 个样本）运行。它的卷积结构通过teacher-forcing完整的音频样本序列来实现训练中的并行处理。然而，WaveNet 的自回归特性使其推理速度非常慢，因为每个样本都必须从输出分布中提取，然后才能在下一个时间步作为输入传入。为了实时生成高保真语音，必须开发高度工程化的推理内核（例如，Arık et al.， 2017a）。

最近，van den Oord et al. （2018） 提出了一个教师-学生框架，从自回归教师 WaveNet 中蒸馏一个并行前馈网络。非自回归学生模型可以比实时快 20 倍的速度生成高保真语音。为了在蒸馏过程中通过随机样本进行反向传播，并行 WaveNet 采用 the mixture of logistics (MoL) distribution（Salimans等 人，2017 年）作为教师 WaveNet 的输出分布。and a logistic distribution based inverse autoregressive flow (IAF) (Kingma et al., 2016) as the student model. 它最小化了一组损失，包括 student 和 teacher 网络的输出分布之间的 KL 散度。然而，必须应用蒙特卡洛方法来近似 Logistic 分布和 MoL 分布之间棘手的 KL 散度，这可能会在高度峰值分布的梯度中引入较大的方差，并导致实践中的训练不稳定。[parrel wavenet 的缺点]

在这项工作中，我们提出了一种基于高斯 IAF 的新型parrel wave生成方法。具体来说，我们做出了以下贡献：

1. 我们证明，单个方差界高斯(a single variance-bounded Gaussian )足以在 WaveNet 中对原始波形进行建模，而不会降低音频质量。与parrel WaveNet 中的量化代理损失(quantized surrogate loss) （Salimans et al.， 2017） 相比，我们的高斯自回归 WaveNet(Gaussian autoregressive WaveNet) 只是用最大似然估计 （MLE） 进行训练。
2. 我们通过最小化它们的峰值输出分布之间的正则化 KL 散度，从自回归 WaveNet 中提炼出一个高斯 IAF。我们的方法提供了 KL 散度的封闭式估计，这在很大程度上简化了蒸馏算法并稳定了训练过程。
3. 在以前的研究中，“端到端”语音合成实际上是指具有单独波形合成器（即声码器）的文本到频谱图模型（Sotelo et al.， 2017;Wang et al.， 2017）。 我们推出了第一个用于 TTS 的text到wave的神经网络架构，它是完全卷积的，支持从头开始的快速端到端训练。在我们的架构中，WaveNet 模块以隐藏状态而不是梅尔频谱图为条件（Ping et al.， 2018;Shen et al.， 2018），这对于从头开始训练的成功至关重要。我们的文本到波形模型在自然度方面明显优于单独训练的pipeline（Ping et al.， 2018）。
4. 我们还成功地蒸馏了一个并行神经声码器，该声码器以端到端架构中学习的隐藏表示为条件。与具有自回归声码器的模型相比，具有并行声码器的文本到波形模型获得了有竞争力的结果。

我们将本文的其余部分组织如下。第 2 节讨论了相关工作。我们在第 3 节中提出了parrel wave的生成方法，并在第 4 节中介绍了text到wave的架构。我们在第 5 节报告实验结果，在第 6 节总结论文。


#### 2.Related Work

神经语音合成已经获得了最先进的结果，最近引起了很多关注。提出了几种神经 TTS 系统，包括 Deep Voice 1 （Arık et al.， 2017a）、Deep Voice 2 （Arık et al.， 2017b）、Deep Voice 3 （Ping et al.， 2018）、Tacotron （Wang et al.， 2017）、Tacotron 2 （Shen et al.， 2018）、Char2Wav （Sotelo et al.， 2017）和 VoiceLoop （Taigman et al.， 2017） 2018）. Deep Voice 1 & 2保留了传统的TTS管道，该管道具有单独的字素到音素、音素持续时间、基频和波形合成模型。相比之下，Deep Voice 3、Tacotron 和 Char2Wav 采用基于注意力的序列到序列模型（Bahdanau et al.， 2015），从而产生更紧凑的架构。在文献中，这些模型通常被称为 “端到端” 语音合成。然而，它们实际上依赖于传统的声码器（Morise et al.， 2016）、Griffin-Lim 算法（Griffin 和 Lim，1984）或单独训练的神经声码器（Ping et al.， 2018;Shen et al.， 2018）将预测的频谱图转换为原始音频。在这项工作中，我们提出了第一个基于 Deep Voice 3 的 TTS  text到wave的神经架构 （Ping et al.， 2018）。

基于神经网络的声码器，如 WaveNet （van den Oord et al.， 2016a） 和 SampleRNN （Mehri et al.， 2017），在语音合成的最新进展中发挥着非常重要的作用。在 TTS 系统中，WaveNet 可以以语言特征、基频 （ 
$F_0$）、音素持续时间（van den Oord et al.， 2016a;Arık et al.， 2017a） 或来自文本到频谱图模型预测的梅尔频谱图（Ping et al.， 2018）作为输入。我们通过在 梅尔频谱图和端到端模型中的隐藏表示上对其进行调节来测试我们的parallel waveform合成方法。

标准化流(Normalizing flows)（Rezende 和 Mohamed，2015 年;Dinh et al.， 2014）是一系列随机生成模型，其中通过应用一系列可逆变换，将简单的初始分布转换为更复杂的初始分布。归一化流提供了任意复杂的后验分布，使其非常适合变分自动编码器中的推理网络（Kingma 和 Welling，2014）。逆自回归流 （IAF） （Kingma et al.， 2016） 是一种特殊类型的归一化流，其中每个可逆变换都基于自回归神经网络。因此，IAF 可以重用最成功的自回归架构，例如 PixelCNN 和 WaveNet（van den Oord et al.， 2016b， a）。学习具有最大可能性的 IAF 可能非常缓慢。在这项工作中，我们通过最小化 KL 散度的数值稳定变体，从预训练的自回归生成模型中提炼出高斯 IAF。

知识蒸馏最初是为了将大模型压缩成小模型而提出的（Bucilua et al.， 2006）。在深度学习中（Hinton et al.， 2015），通过最小化学生输出之间的损失（例如，L2 或交叉熵），从教师网络中提炼出一个较小的学生网络。在parallel WaveNet 中，通过最小化反向 KL 散度(reverse KL divergence (Murphy, 2014).)，从自回归 WaveNet 中提炼出非自回归学生网 （Murphy， 2014）。类似的技术也应用于机器翻译的非自回归模型（Gu et al.， 2018;Kaiser等 人，2018 年;Lee等 人，2018 年;Roy et al.， 2018）。


#### 3. Parallel Wave Generation

在本节中，我们将高斯自回归 WaveNet 作为教师网，将高斯逆自回归流作为学生网。然后，我们开发我们的知识蒸馏算法。

##### 3.1 Gaussian Autoregressive WaveNet

WaveNet使用条件概率的链式法则将高维的waveform $x=\\{x_1,...,x_T\\}$建模为条件分布的乘积：
$$p(x|c;\theta)=\prod^{T}_ {t=1}p(x_t|x_{< t},c;\theta)$$

这里的$c$是一个条件（例如，第4节中的mel-spectrogram或hidden state),$\theta$是模型参数，自回归的Wavenet将$x_{ < t }$作为输入，输出$x_t$的概率分布。

Parallel WaveNet (van den Oord et al., 2018) 主张在PixelCNN++中使用mixture of logitic (MoL) distribution 用于自回归教师网络。因为与分类分布（例如，16 位音频的 65,536 个 softmax 单位）相比，它需要的输出单元要少得多。实际上，student-net 的输出分布需要在样本𝒙上可区分，并允许在蒸馏中从教师到学生的反向传播。因此，还需要为教师 WaveNet 选择连续分布。直接最大化 MoL 的对数似然容易出现数值问题，并且必须采用 PixelCNN++ 中引入的量化代理损失（quantized surrogate loss ）。

在这项工作中，我们证明了 WaveNet 的单个高斯输出分布足以对原始波形进行建模。这可能会引起建模能力的担忧，因为我们使用单个高斯分布而不是高斯分布的混合（Chung et al.， 2015）。我们将在实验中证明它们的可比性能。具体来说，给定先前样本$x_t$的条件 
分布为，

<div align=center>
    <img src="zh-cn/img/ch4/04/p1.png" /> 
</div>

其中$\mu(x_{ < t};\theta)$和$\sigma(x_{ < t};\theta)$分别是自回归WaveNet预测的平均值和标准差。在实践中，该网络在对数尺度$\log \sigma (x_{ < t })$上进行预测和操作，以实现数值稳定性。给定观察到的数据，我们对$\theta$进行最大似然估计。请注意，该模型可能会给出没有实值噪声(即$\mu(x{ < t } \approx x_t)$)的16位离散$x_t$非常准确的预测，那么当可以自由最小化$\sigma(x_{ < t})$时，对数似然计算可能会变的数值不稳定。因此，训练过程中对$\log \sigma(x_{ < t})$的预测进行裁剪。

在附录A中我们讨论了裁剪到对数尺度的常数的重要性。我们还尝试了反量化技巧，像16位样本添加俊宇噪声$\mu \in [0,\frac{2}{65526}]$,类似于图像建模(e.g., Uria et al., 2013)。事实上，这些技巧是等效的，因为他们都上界了对量化数据进行建模的连续似然。我更喜欢clipping技巧，因为他显式控制模型行为并简化了概率密度蒸馏。


##### 3.2 Gaussian Inverse Autoregressive Flow (IAF)

标准化流（Normalizing flows）（Rezende 和 Mohamed，2015 年;Dinh et al.， 2017）通过应用可逆变换 $x=f(z)$ 将简单初始密度$q)z)$（例如，各向同性高斯）映射到复数初始密度。
$f$是一个双射，可以通过变量的更改公式得到$x$的分布：

<div align=center>
    <img src="zh-cn/img/ch4/04/p2.png" /> 
</div>


其中$det(\frac{\partial f(z)}{\partial z})$是雅可比行列式，通常计算成本很高。逆自回归流（IAF）（Kingma et al.， 2016） 是一种特殊的归一化流，具有简单的雅可比行列式。在IAF中，$z$和$x$维度相同，并且变换基于一个自回归的网络，以$z$为输入：$x_t=f(z_{ < t };\phi)$,其中$\phi$是模型参数。注意，第$t$个变量$x_t$仅取决于先前和当前的潜在变量$z_{ < t }$,因此雅可比矩阵是一个三角矩阵，行列式是对角巷的乘积，

<div align=center>
    <img src="zh-cn/img/ch4/04/p3.png" /> 
</div>

这很容易计算。Parallel WaveNet （van den Oord et al.， 2018） 使用基于single logistic distribution的 IAF 来匹配教师网络的nixture of logistics distribution(MoL)。

我们使用高斯IAF（Kingma et al.， 2016）并将其转为$x_t=f(z_{\leq t};\phi)$定义为：

<div align=center>
    <img src="zh-cn/img/ch4/04/p4.png" /> 
</div>


其中，位移函数$\mu(z_{ < t };\phi)$和缩放函数$\sigma(z_{< t};\phi)$由第 3.1 节中的自回归 WaveNet 建模。IAF的变换在给定$z$的提艾签下是并行计算$x$的。从而有效利用 GPU 等资源。重要的是，如果我们假设$z_t \sim N(z_t| \mu_0,\sigma_0)$,很容易的观察到$x_t$也是遵循高斯分布的,

<div align=center>
    <img src="zh-cn/img/ch4/04/p5.png" /> 
</div>

其中，$\mu_q=\mu_0 \times \sigma(z_{ < t};\phi) + \mu(z_{< t};\phi)$和$\sigma_q=\sigma_0 \times \sigma(z_{< t};\phi)$，Note that 
𝒙 are highly correlated through the marginalization of latents 𝒛, and the IAF jointly models 𝒙 at all timesteps.

为了评估 观测数据 𝒙 的可能性 ，我们可以使用恒等式 （3） 和 （4），并插入方程 （5） 中定义的变换，这将得到，

<div align=center>
    <img src="zh-cn/img/ch4/04/p6.png" /> 
</div>

但是，需要方程 （5） 中的逆变换$f^{-1}$，



<div align=center>
    <img src="zh-cn/img/ch4/04/p7.png" /> 
</div>

从观察到的 𝒙计算相应的𝒛 ，这是自回归且缓慢的。因此，直接通过最大似然法学习 IAF 可能非常缓慢。

通常，规范化流需要一系列转换，直到分布 $q(x;\phi)$ 达到所需的复杂程度。首先，我们从各向同性高斯分布 $N(0,I)$中抽取一个白噪声样本 $z^{(0)}$。然后，我们重复应用方程 （5） 中定义的变换 

<div align=center>
    <img src="zh-cn/img/ch4/04/p8.png" /> 
</div>

我们在算法 1 中总结了此过程。请注意，这些参数不会在不同的流之间共享。

<div align=center>
    <img src="zh-cn/img/ch4/04/p9.png" /> 
</div>

##### 3.3 Knowledge Distillation

###### 3.3.1 Regularized KL Divergence

van den Oord et al. （2018） 提出了概率密度蒸馏法，以规避 IAF 最大似然学习的困难。 在蒸馏中，目标是最小化学生 IAF 和预训练教师 WaveNet 之间的序列水平反向 KL 散度。这种序列水平的 KL 散度可以通过从IAF中采样𝒛和 𝒙=f(𝒛)进行近似。但它可能会表现出高方差。可以通过边缘化每个时间步的提前一步预测来减少该估计的方差（van den Oord et al.， 2018）。但是，并行 WaveNet 必须在每个时间步长运行单独的蒙特卡洛采样，因为logistics和mixture of logistics之间的每个时间步长 KL 差异仍然很棘手。Indeed, parallel WaveNet first draws a white noise sample 𝒛, then it draws multiple different samples $x_t$ from $q(x_t|z_{< t})$ to estimate the intractable integral. Our method only need to draw one sample 𝒛, then it computes the KL divergence in closed-form thanks to the Gaussian setup.
 
给定一个白噪声样本 𝒛，算法 1 输出样本 $x=f(z)$，以及具有均值 $\mu_q$和标准差$\sigma_q$ 的输出高斯分布 $\sigma_q q(x_t|z_{< t};\phi)$.
 我们将样本 𝒙 输入到自回归 WaveNet 中，并获得其具有均值 $\mu_p$和标准差$\sigma_q$的输出分布 $\sigma_q q(x_t|z_{< t};\theta)$
。可以证明学生的输出分布和教师的输出分布之间的每时间步 KL 散度 具有封闭式表达式

<div align=center>
    <img src="zh-cn/img/ch4/04/p10.png" /> 
</div>

这也形成了对学生分布$q(x)$和教师$p(x)$之间序列水平 KL 差异的无偏估计。


In this submission, we lower bound $\log \sigma_p$ and $\log \sigma_q$ at -7 before calculating the KL divergence. 

<div align=center>
    <img src="zh-cn/img/ch4/04/p11.png" /> 
</div>

##### 3.3.2 STFT Loss

在知识蒸馏中，通常的做法是使用真实数据集来合并额外的损失（例如，Kim 和 Rush，2016）。实证上，我们发现仅用 KL 散度损失训练学生 IAF 会导致whisper voices。 van den Oord et al. （2018） 主张用平均功率loss(average power loss )来解决这个问题，这实际上在他们的实验中与训练音频剪辑的长度较短（即 
0.32s）相结合。随着裁剪长度的增加，平均功率损失的效果会降低。相反，我们计算学生 IAF 的输出样本 𝒙与相应的真实音频 $𝒙_n$之间的帧级损失:

<div align=center>
    <img src="zh-cn/img/ch4/04/p13.png" /> 
</div>

其中 $|STFT(x)|$是短期傅里叶变换 （STFT） 的幅度 B=1025是我们将 FFT 大小设置为 2048时的频率区间数。我们使用 12.5ms 移帧、 50ms 窗口长度和 Hanning 窗口。我们最终的损失函数是平均 KL 散度和帧级损失的线性组合，我们只需在所有实验中将它们的系数设置为 1。


#### 4.Text-to-Wave Architecture

在本节中，我们将介绍我们的全卷积文本到波形架构（参见图 Fig2 （a）） 用于端到端 TTS。我们的架构基于 Deep Voice 3 （DV3），这是一种基于卷积注意力的 TTS 系统 （Ping et al.， 2018）。DV3 能够将文本特征（例如字符、音素和重音）转换为频谱特征（例如，对数梅尔频谱图和对数线性频谱图）。这些频谱特征可以用作单独训练的波形合成模型（如 WaveNet）的输入。相比之下，我们从注意力机制中学到的隐藏表示，通过一些中间处理直接提供给 WaveNet，端到端地从头开始训练整个模型。

<div align=center>
    <img src="zh-cn/img/ch4/04/p14.png" /> 
</div>

请注意，根据隐藏表示来调节 WaveNet 对于从头开始训练的成功至关重要。事实上，我们试图根据 DV3 预测的梅尔频谱图来限制 WaveNet，因此 WaveNet 损失的梯度可以通过 DV3 反向传播，以改进文本到频谱图模型。当整个模型从头开始训练时，我们发现它的性能比单独的训练管道略差。主要原因是 DV3 预测的梅尔频谱图在早期训练时可能不准确，并可能破坏 WaveNet 的训练。为了获得满意的结果，需要预训练 DV3 和 WaveNet，然后微调整个系统（例如，Zhao et al.， 2018）。

架构由四个组件组成：

+ Encoder：与 DV3 中的卷积编码器相同，它将文本特征编码为内部隐藏表示形式。
+ Decoder：DV3 中的因果卷积解码器，它以自回归的方式将编码器表示解码为 log-mel 频谱图。
+ Bridge-net：一个卷积中间处理块，用于处理来自解码器的隐藏表示并预测对数线性频谱图。与解码器不同，它是非因果性的，因此可以利用未来的上下文信息。此外，它还将 hidden 表示从 frame level 上采样到 sample-level。
+ 声码器（Vocoder）：用于合成波形的高斯自回归 WaveNet，它以桥接网的上采样隐藏表示为条件。这个组件可以被从 autoregressive vocoder 提炼出来的学生 IAF 替换。

整体目标函数是解码器、桥接网络和声码器损失的线性组合;我们只是在实验中将所有系数设置为 1。我们引入了 bridge-net 来利用未来的时间信息，因为它可以应用非因果卷积。我们架构中的所有模块都是卷积的，可以实现快速训练。 并缓解了基于 RNN 的模型中的常见困难（例如，梯度消失和爆炸问题（Pascanu et al.， 2013））。在整个模型中，我们使用来自 DV3 的卷积块（参见Fig 2（c）） 作为基本构建块。它由一个带有门控线性单元 （GLU） 的一维卷积和一个残差连接组成。在所有实验中，我们将 dropout 概率设置为 0.05。我们将在以下小节中提供更多详细信息。


##### 4.1 Encoder-Decoder

我们使用与 DV3 相同的编码器-解码器架构（Ping等 人，2018 年）。编码器首先将字符或音素转换为可训练的嵌入，然后是一系列卷积块来提取长距离文本信息。解码器自回归预测具有 L1 损失的 log-mel 频谱图（训练时教师强迫）。它从 1x1 卷积层开始，对输入 log-mel 频谱图进行预处理，然后应用一系列因果卷积和注意力。在字符嵌入和 log-mel 频谱图之间学习基于多跳注意力的对齐。


##### 4.2 Bridge-net

解码器的隐藏状态被馈送到 bridge-net 进行时间处理和上采样。然后将输出的隐藏表示馈送到声码器进行波形合成。 Bridge-net 由一堆卷积块和两层转置的 2-D 卷积组成，与 softsign 交错，以将每个时间步的隐藏表示从每秒 80 个上采样到每秒 24,000 个。时间上的上采样步幅分别为 15和 20对于两个层。相应地，我们将 2-D 卷积滤波器大小设置为 (30,3)和 (40,3)，其中滤波器大小（在时间上）从步幅中增加一倍，以避免棋盘伪影(checkerboard artifacts)（Odena et al.， 2016）。


#### 5.Experiment

<div align=center>
    <img src="zh-cn/img/ch4/04/p15.png" /> 
</div>

数据来自于百度研究院的内部数据的测试，结果的可信性值得怀疑。


#### 6.Conclusion

在这项工作中，我们首先证明了单个高斯输出分布足以在 WaveNet 中对原始波形进行建模，而不会降低音频质量。然后，我们提出了一种基于高斯逆自回归流 （IAF） 的平行波生成方法，其中 IAF 是通过最小化高度峰值分布的正则化 KL 散度从自回归 WaveNet 中提炼出来的。与并行 WaveNet 相比，我们的蒸馏算法以封闭形式估计 KL 散度，并在很大程度上稳定了训练过程。此外，我们提出了第一个用于 TTS 的text到wave的神经网络架构，它可以以端到端的方式从头开始训练。我们的文本到波形架构优于单独训练的管道，并为完全端到端的 TTS 开辟了研究机会。我们还通过提炼一个以端到端模型中的隐藏表示为条件的并行神经声码器来展示吸引人的结果。


------


### 4. FastSpeech 2s

参考:[FastSpeech V2](zh-cn/03_Text_to_spectrogram?id=_9-fastspeech-2-fast-and-high-quality-end-to-end-text-to-speech)



### 5. EATS

<!-- 2021：  END-TO-END ADVERSARIAL TEXT-TO-SPEECH-->

!> arxiv(2021): https://arxiv.org/abs/2006.03575


### 6. Wave-Tacotron

<!-- google 2021 -->




### 7. JETS

<!-- 2020 -->

!> arxiv(2020): https://arxiv.org/pdf/2203.16852.pdf


