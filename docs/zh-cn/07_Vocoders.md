
## Vocoders(声码器)

TTS的工作主要是把文本信息转成音频信息，其大致流程分为前端处理和后端处理两个部分。前端的工作主要是语言领域的处理，主要包括分句、文本正则、分词、韵律预测、拼音预测（g2p)，多音字等等。后端的主要工作是把前端预测的语言特征转成音频的时域波形，大体包括声学模型和声码器，其中声学模型是把语言特征转成音频的声学特征，声码器的主要功能是把声学特征转成可播放的语音波形。声码器的好坏直接决定了音频的音质高低，尤其是近几年来基于神经网络声码器的出现，使语音合成的质量提高一个档次。目前，声码器大致可以分为基于相位重构的声码器和基于神经网络的声码器。基于相位重构的声码器主要因为TTS使用的声学特征（mel特征等等）已经损失相位特征，因此使用算法来推算相位特征，并重构语音波形。基于神经网络的声码器则是直接把声学特征和语音波形做mapping，因此合成的音质更高。目前，比较流行的神经网络声码器主要包括wavenet、wavernn、melgan、waveglow、fastspeech和lpcnet等等。

本部分我们介绍声码器的一些paper,主要是基于神经网络的神声码器的介绍。
<!-- https://zhuanlan.zhihu.com/p/321798376 -->

------

### 1. MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis

!> https://arxiv.org/abs/1910.06711

!> https://github.com/descriptinc/melgan-neurips

<!-- https://blog.csdn.net/qq_28662689/article/details/105971998 -->

<!-- https://lxp-never.blog.csdn.net/article/details/125145532?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-125145532-blog-104771912.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-125145532-blog-104771912.235%5Ev39%5Epc_relevant_3m_sort_dl_base4&utm_relevant_index=2 -->

<!-- https://blog.csdn.net/qq_28662689/article/details/105971998 -->

<!-- https://zhuanlan.zhihu.com/p/157321340 -->

#### 摘要

以前的工作（Donahue等人，2018a；Engel等人，2019a）已经发现用GAN生成相干的原始音频波形是一个挑战。在本文中，我们证明了通过引入一系列结构变化和简单的训练技术，可以可靠地训练GANs以产生高质量的相干波形。主观评价指标（Mean-Opinion Score，简称MOS）表明了该方法对高质量mel谱-spectrogram inversion(反推)的有效性。为了建立这些技术的通用性，我们展示了我们的模型在语音合成、音乐领域翻译和无条件音乐合成方面的定性结果。我们通过消融实验来评估模型的各个组成部分，并提出一套方法来设计conditional sequence synthesis tasks的通用鉴别器和生成器。我们的模型是非自回归的，完全卷积的，参数明显少于竞争模型，并且可以推广到看unsee speaker进行梅尔谱图反演(mel-spectrogram inversion)。在没有任何针对硬件的优化技巧下我们的Pytorch实现在GTX1080Ti GPU上的运行速度比实时快100倍以上，在CPU上比实时运行快2倍以上。

#### 1.引言

建模原始音频是一个特别具有挑战性的问题，因为数据时间分辨率很高(通常至少16000个样本每秒)，并且在不同的时间尺度上存在短期和长期的依赖关系。因此，与其直接建模原始时间音频，大多数方法通常对原始时间信号更低分辨率音频建模来简化问题。通常选择这样的表示形式比原始音频更容易建模，同时保留足够的信息以允许准确地倒转回音频。在言语方面，对齐的语言特征(Van Den Oord等人，2016)和mel-spectograms (Shen等人，2018;Gibiansky等人，2017)是两种常用的中间表示。因此，音频建模通常被分解为两个阶段。

1. 将文本转换成一种中间特征表示，然后对这种特征进行建模。
2. 将中间表示法转换回音频。

在本研究中，我们关注的是后一阶段，并选择mel-spectogram作为中间表征。目前的mel-spectogram反演方法可以分为三类：
1. 纯信号处理技术
2. 自回归神经网络
3. 非自回归神经网络

我们将在接下来描述这三种主要的研究方向。

**纯信号处理方法**

不同的信号处理方法已被探索，以找到一些方便的低分辨率音频表示，既可以容易地建模和有效地转换回时间音频。例如，**Griffin-Lim**（Griffin＆Lim，1984）算法允许有效地将STFT序列解码回时域信号，代价是引入较强的机器人伪像（robotic artifacts），如Wang等人所述（2017）。目前已经研究了更复杂的表示和信号处理技术。例如，**WORLD**声码器（MORISE et al，2016）引入了一种中间类表示形式，专门针对基于类似于mel频谱图的特征的语音建模而设计。  WORLD声码器与专用信号处理算法配对，以将中间表示映射回原始音频。 它已成功用于进行文本到语音的合成，例如在Char2Wav中，其中WORLD声码器功能通过基于注意力的递归神经网络进行建模（Sotelo等，2017; Shen等，2018; Ping 等人，2017）。 **这些纯信号处理方法的主要问题在于，从中间特征到音频的映射通常会引入明显的伪像**。

**基于自回归神经网络的模型**

WaveNet (Van Den Oord等人，2016)是一种全卷积自回归序列模型，可以根据与原始音频时间一致的语言特征生成高度真实的语音样本。它也能够产生高质量的无条件语音和音乐样本。SampleRNN (Mehri等人，2016)是一种实现无条件波形生成的替代架构，它使用多尺度递归神经网络在不同时间分辨率上显式地为原始音频建模。WaveRNN (Kalchbrenner et al.， 2018)是一种基于简单的单层递归神经网络的更快的自回归模型。WaveRNN引入了各种技术，如稀疏化和子尺度生成，以进一步提高合成速度。这些方法已经在文本到语音合成(Sotelo et al., 2017; Shen et al., 2018; Ping et al., 2017)和其他音频生成任务(Engel et al., 2017)中取得了最先进的成果。不幸的是，由于音频样本必须按顺序生成，因此使用这些模型进行推理的速度天生就很慢且效率低下。因此，自回归模型通常不适合实时应用。（我们将在下面的其他章节详细的介绍自回归的这些Vocoder)

**非自回归模型**

近来，人们致力于开发非自回归模型以反转低分辨率音频表示。这些模型比自回归模型快几个数量级，因为它们具有高度可并行性，并且可以充分利用现代深度学习硬件（例如GPU和TPU）。现在已经出现了两种不同的方法来训练这种模型。

1. Parallel Wavenet（Oord等人，2017）和Clarinet（Ping等人，2018）将经过训练的自回归解码器提炼成基于流的卷积学生模型。使用基于Kulback-Leibler散度: $KL[P_{student}||P_{teacher}]$以及其他感知损失项的概率蒸馏目标对学生网络进行了训练。
2. WaveGlow（Prenger等人，2019）是基于流的生成模型，基于Glow（Kingma＆Dhariwal，2018）。 WaveGlow是一种非常高容量的生成流，它由12个耦合和12个可逆`1x1`卷积组成，每个耦合层由8层扩张卷积堆叠组成。作者指出，需要为期8周的GPU训练，才能获得单个扬声器模型的高质量结果。尽管在GPU上推理速度很快，但是模型的庞大尺寸使其对于内存预算有限的应用程序不切实际。

**GANs for audio**

到目前为止，尚未针对音频建模探索的一种方法是生成对抗网络（GAN）（Goodfellow et al，2014）。 GAN在无条件图像生成（Gulrajani等，2017; Karras等，2017，2018），图像到图像翻译方面（Isola等，2017; Zhu等，2017; Wang等2018b）和视频到视频合成（Chan等人，2018; Wang等人，2018a）取得了稳步进展。尽管它们在计算机视觉方面取得了巨大的成功，但在使用GAN进行音频建模方面，我们还没有看到多少进展。Engel等人(2019b)使用GAN通过模拟STFT幅度和相位角来生成音乐音色，而不是直接模拟原始波形。 Neekhara等（2019）提出使用GANs来学习从梅尔频谱图到简单幅度频谱图的映射，并将其与相位估计相结合，恢复原始音频波形。Yamamoto等人(2019)使用GAN提取自回归模型，生成原始语音音频，但是他们的结果表明仅对抗损失不足以产生高质量的波形；它需要基于KL散度的蒸馏目标作为一个关键组成部分。迄今为止，使它们在此领域中良好运行一直是一项挑战（Donahue等，2018a）。

主要贡献：
+ 我们提出了MelGAN，一个非自回归前馈卷积架构，用于在GAN设置中生成音频波形。 据我们所知，这是第一项成功训练GAN的原始音频生成工作，而没有额外的蒸馏或感知损失功能，同时仍能产生高质量的文本到语音合成模型的第一项工作。
+ 通过在通用音乐翻译、文本到语音的生成和无条件音乐合成方面的实验，我们证明了自回归模型可以很容易地用快速并行的MelGAN解码器代替，尽管质量会有轻微的下降。
+ 我们还表明，MelGAN大大快于其他mel-spectrogram反演方案。特别是，它比目前最快的模型快10倍(Prenger et al，2019)，而且音频质量没有明显下降。

#### 2.MelGAN模型

在这一节中，我们描述我们的mel-spectrogram反演的生成器和鉴别器架构。我们描述了模型的核心组件，并讨论了执行无条件音频合成的修改。我们在参数数量和在CPU和GPU上的推理速度方面，将提出的模型与竞争方法进行了比较。 下图显示了整体架构。

<div align=center>
    <img src="zh-cn/img/ch7/01/p1.png" /> 
</div>

##### 2.1 生成器

**架构**： 我们的生成器是一个完全卷积的前馈网络，输入的信号为梅尔谱图$s$，输出的原始波形为$x$。由于梅尔谱图在`256x lower`的时间分辨率上（用于所有实验），因此我们使用一维转置卷积层对输入序列进行上采样。 每个转置的卷积层后面是一堆带有扩张卷积(空洞卷积)的残差块。 与传统的GAN不同，我们的生成器不使用全局噪声向量作为输入。 我们在实验中注意到，当额外的噪声input到生成器时，生成的波形几乎没有感知差异。 这是违反直觉的结果，因为$s \to x$的求逆涉及到一个一对多的映射，因为$s$是$x$的有损压缩。 但是，这一发现与Mathieu(2015）和和Isola（2017）等人的观点不一致，他们认为如果条件信息非常强，则噪声输入并不重要。

**感应感受野（Induced receptive field）**：在基于卷积神经网络的图像生成器中，由于感受野的高度重叠，空间上靠近的像素点之间存在一种感应偏差。 我们设计的生成器架构，以产生一个感应偏置，**即音频时间步长之间存在长距离相关性**。我们在每个上采样层之后添加了带有空洞卷积的残差块，这样在时间上，后续每一层的输出激活都有明显的输入重叠。 堆叠的空洞卷积层的感受野随层数的增加而指数增加。 与Van Den Oord等类似（2016年），将这些纳入我们的生成器使我们能够有效地增加每个输出时间步长的感受野。 这有效地揭示了相距较远的时间步长的感受野中存在较大的重叠，从而产生更好的长距离相关性。

**Checkerboard artifacts**（棋盘效应）： 正如Odena等人（2016年）所指出的，如果未仔细选择转置卷积层的卷积核大小和stride，则反卷积生成器很容易生成“棋盘格”(马赛克)模式。 Donahue等人（2018b）对原始波形生成进行了研究，发现这种重复的模式会导致可听到的高频嘶嘶声。 为了解决这个问题，我们仔细选择反卷积层的卷积核大小和步幅（stride），作为Donahue等人(2018b)中引入的PhaseShuffle层的一个更简单的替代方案。 跟随Odena等人(2016)，我们使用卷积核大小为stride的倍数。 如果未正确选择空洞卷积卷积核核大小，则这种重复模式的另一个来源可能是空洞卷积的堆叠。 我们确保dilation随卷积核大小的增长而增长，这样堆叠的感受野看起来就像一个完全平衡的(均匀地看到输入)和以卷积核大小作为分支因子的对称树。（这部分翻译真垃圾）

**标准化技术（Normalization technique）**：我们注意到，生成器结构总选择归一化技术对于样本生成质量至关重要。用于图像生成的常用conditional GAN架构（Isola等人，2017；Wang等人，2018b）在生成器的所有层中使用实例归一化(instance Normalization, Ulyanov et al.2016)。但是，在音频生成的情况下，我们发现实例规范化会冲走重要的音高信息，使音频听起来具有金属感。根据Zhang等人和Park等人(2019)的建议，在生成器上应用频谱归一化(spectral normalization)（Miyato等人，2018）时，我们也获得了较差的结果。我们认为，对鉴别器的强烈Lipshitz约束会影响用于训练生成器的特征匹配目标（在3.2节中进行了说明）。在所有可用的归一化技术中，权重归一化（Weight normalization）（Salimans和Kingma，2016）效果最好，因为它不会限制鉴别器的容量或对激活进行归一化。它只是通过将权重矢量的比例从方向上解耦来简单地重新参数化权重矩阵，从而获得更好的训练动力学。因此，我们在生成器的所有层中使用 **权重归一化**(weight normalization)。

##### 2.2 判别器

**多尺度结构（Multi-Scale Architecture）**: 与Wang et al. (2018b)相同，我们采用了具有3个判别器block（D1、D2、D3）的尺度架构，这些判别器具有相同的网络结构，但在不同的音频尺度scale上运行。 D1操作在原始音频的尺度上，而D2,D3分别操作在原始音频下采样2倍和4倍的尺度上。 下采样是使用卷积核大小为4的strided average pooling。音频具有不同层次的结构，因此可以激发不同尺度的多个判别器。 这种结构具有感应偏差，每个判别器都可以学习音频不同频率范围的特征。 例如，对下采样音频进行操作的判别器无法访问高频分量，因此倾向于仅基于低频分量学习判别特征。

**基于窗的目标（Window-based objective）**: 每个单独的判别器都是基于马尔可夫窗口的判别器（类似于图像修复，Isola等人（2017）），由一系列kernel size的跨步卷积层组成（consisting of a sequence of strided convolutional
layers with large kernel size）。我们利用分组卷积（grouped convolutions）来允许使用更大的卷积核，同时保持较小的参数数量。虽然标准GAN判别器学习在整个音频序列的分布之间进行分类，而基于窗的判别器学习在小音频块的分布之间进行分类。由于判别器损耗是在每个窗口都非常大（等于判别器的感受野）的重叠窗口上计算的，因此，MelGAN模型学习在各个块之间保持一致性。我们选择了基于窗的判别器，因为它们已经被证明可以捕获基本的高频结构，需要较少的参数，运行速度更快，并且可以应用于可变长度的音频序列。与生成器类似，我们在判别器的所有层中使用权重归一化。

##### 2.3 训练目标

为了训练GAN，我们的GAN目标函数使用hinge损失版本 (Lim & Ye, 2017; Miyato et al., 2018)。我们还试验了最小二乘(LSGAN)公式(Mao et al.，2017)，并注意到hinge版本有轻微改进。

$$公式1：\min_{D_{k}} \mathbb{E}_ {x}\left[\min \left(0,1-D_{k}(x)\right)\right]+\mathbb{E}_ {s, z}\left[\min \left(0,1+D_{k}(G(s, z))\right)\right], \forall k=1,2,3$$
$$公式2：\min_ {G} \mathbb{E}_ {s, z}\left[\sum_{k=1,2,3}-D_{k}(G(s, z))\right]$$

其中$x$表示原始波形，$s$表示条件信息(例如。mel-spectrogram)和$z$表示高斯噪声向量。

**特征匹配**：除了判别器的信号外，我们使用特征匹配损失函数（Larsen等，2015）来训练生成器。 该目标最小化真实音频和合成音频的判别器特征图之间的$L_1$距离。 直观的说，这可以看作是学习的相似性度量，一个判别器学习了一个特征空间，从而从真实数据中判别出假数据。 值得注意的是，我们没有使用任何损失的原始音频空间。 这与其他有条件的GAN（Isola等人，2017）相反，其中$L_1$损失被用来匹配有条件生成的图像及其相应的ground-truths，以增强全局一致性。 实际上，在我们的案例中，在音频空间中增加$L_1$损耗会引入可听噪声，从而损害音频质量。
$$公式3：\mathcal{L}_ {\mathrm{FM}}\left(G, D_{k}\right)=\mathbb{E}_ {x, s \sim p_{\text {data }}}\left[\sum_{i=1}^{T} \frac{1}{N_{i}}\left\|D_{k}^{(i)}(x)-D_{k}^{(i)}(G(s))\right\|_{1}\right]$$

为了简化符号，$D_k^{(i)}$表示第$k$个判别器块的第$i$层特征图输出，$N_i$表示每一层的单元数，特征匹配类似于感知损失(Dosovitskiy & Brox, 2016; Gatys et al., 2016; Johnson et al., 2016)。在我们的工作中，我们在所有判别器块的每个中间层使用特征匹配。
$$公式4：\min_ {G}\left(\mathbb{E}_ {s, z}\left[\sum_{k=1,2,3}-D_{k}(G(s, z))\right]+\lambda \sum_{k=1}^{3} \mathcal{L}_ {\mathrm{FM}}\left(G, D_{k}\right)\right)$$

##### 2.4 参数数量和推理速度

在我们的体系结构中，归纳偏差使得整个模型在参数数量上明显小于竞争模型。由于是非自回归且完全卷积的模型，因此我们的模型推理速度非常快，能够在GTX1080 Ti GPU上以2500kHz的频率全精度运行（比最快的竞争模型快10倍以上），在CPU上达到50kHz（更多） 比最快的竞争机型快25倍）。 我们认为我们的模型也非常适合硬件特定的推理优化（例如Tesla V100的半精度（Jia等人，2018; Dosovitskiy＆Brox，2016）和量化（如Arik等人（2017）所做的那样）），这将进一步提高推理速度，下图给出了详细的比较。

<div align=center>
    <img src="zh-cn/img/ch7/01/p2.png" /> 
</div>

#### 3.结果

为了鼓励重现性，我们在论文所附的代码中附加了代码：<https://github.com/descriptinc/melgan-neurips>。

##### 3.1 Ground truth mel-spectrogram反演

**消融实验**: 首先，为了理解我们提出的模型的各个组成部分的重要性，我们对重建的音频进行了定性和定量分析，以完成声谱图反演任务。我们删除某些关键的结构，并使用测试集评估音频质量。下图显示了通过人类听力测试评估的音频质量的平均意见得分。每个模型在LJ语音数据集上进行了40万次迭代训练（Ito，2017）。我们的分析得出以下结论：生成器中没有堆叠的空洞卷积或删除权重归一化会导致高频伪像。使用单个判别器（而不是多尺度判别器）会产生金属音频，尤其是在说话人呼吸时。此外，在我们内部的6个干净的说话人数据集上，我们注意到这个版本的模型跳过了某些浊音部分，完全丢失了一些单词。使用频谱归一化或去除基于窗口的判别器损失会使我们难以学习到清晰的高频模式，从而导致样本产生明显的噪声。在真实波形和生成的原始波形之间添加额外的$L_1$惩罚，会使样本听起来像金属质感，并带有额外的高频伪像。

<div align=center>
    <img src="zh-cn/img/ch7/01/p3.png" /> 
</div>

**基准竞争模型**: 接下来，比较MelGAN在将ground truth mel-spectrograms转化为raw音频与现有方法(如WaveNet vocoder, WaveGlow, Griffin-Lim和ground truth audio)的性能，我们运行了一个独立的MOS测试，其中MelGAN 训练模型直到收敛（大约2.5M迭代）。 与消融研究类似，这些比较是在LJ语音Datset训练的模型上进行的。 比较结果如下图所示。

<div align=center>
    <img src="zh-cn/img/ch7/01/p4.png" /> 
</div>

实验结果表明，MelGAN在质量上可与目前最先进的高容量（high capacity）基于波形的模型(如WaveNet和WaveGlow)相媲美。我们相信，通过进一步探索将GANs用于音频合成的这一方向，在未来可以迅速弥补这一性能差距。（说白了不如WaveNet和WaveGlow)

**Generalization to unseen speakers**: 有趣的是，我们注意到，当我们在包含多个说话者的数据集上训练MelGAN时（内部6个说话者数据集由3个男性和3个女性说话者组成，每个说话者大约需要10个小时），结果模型能够推广到全新的（看不见的）说话者 在训练集外。 该实验验证了MelGAN是能够学习说话人不变的mel频谱图到原始波形的映射。

为了提供一个易于比较的指标来系统地评估这种泛化（针对当前和未来的工作），我们在公开的VCTK数据集上运行了MOS听力测试，用于实地梅尔谱图反演（Veaux等人，2017） 。 该测试的结果如下图所示：

<div align=center>
    <img src="zh-cn/img/ch7/01/p5.png" /> 
</div>

##### 3.2 端到端语音合成

我们在提出的MelGAN与竞争模型之间进行了定量和定性的比较，这些模型基于梅尔频谱图 inversion 用于端到端语音合成。 我们将MelGAN模型插入端到端语音合成管道如下图所示，并使用竞争模型评估文本到语音样本的质量。

<div align=center>
    <img src="zh-cn/img/ch7/01/p6.png" /> 
</div>

具体来说，我们比较了使用MelGAN进行频谱图反转与使用Text2mel((开源char2wav模型的改进版本))的WaveGlow时的采样质量（Sotelo等人，2017）。 Text2mel生成mel-谱图，而不是声码器帧，使用音素作为输入表示，并可以与WaveGlow或MelGAN耦合来反转生成的mel-谱图。 我们使用此模型是因为它的采样器训练速度更快，并且不会像Tacotron2那样执行任何mel频率削波。 此外，我们还采用了最先进的Tacotron2模型(Shen et al.， 2018)和WaveGlow进行基线比较。我们使用NVIDIA在Pytorch Hub存储库中提供的Tacotron2和WaveGlow的开源实现来生成示例。在使用WaveGlow时，我们使用官方存储库中提供的强度为0:01的去噪器来删除high frequency artifacts。MOS测试结果如下图所示。

<div align=center>
    <img src="zh-cn/img/ch7/01/p10.png" /> 
</div>

对于所有的实验，MelGAN都是在单个NVIDIA RTX 2080Ti GPU上以批处理大小16进行训练的。我们用Adam作为优化器，对于生成器和判别器的学习率为1e-4，$\bet_1=0.5$且$\beta_2=0.9$。合成样例Demo:<https://melgan-neurips.github.io>,<https://www.descript.com/overdub>。

结果表明，作为TTS管道的声码器组件，MelGAN可以与一些迄今为止性能最好的模型相媲美。为了更好地进行比较，我们还使用Text2mel + WaveNet声码器创建了一个TTS模型。我们使用Yamamoto(2019)提供的预训练过的WaveNet声码器模型，对Text2mel模型进行相应的数据预处理。然而，该模型获得的MOS评分仅为3.40+0.04。在我们所有的端到端TTS实验中，我们只在GT谱图上训练神经声码器，然后直接在生成的谱图上使用它。我们怀疑Text2Mel + WaveNet实验的糟糕结果可能是由于没有在生成的谱图上对WaveNet声码器进行校正(如在Tacotron2中所做的那样)。因此，我们决定不在表格中报告这些分数。

##### 3.3 非自回归解码器的音乐翻译(\*)

为了证明MelGAN是健壮的，并且可以插入到目前使用自回归模型进行波形合成的任何设置中，我们用MelGAN生成器替换了通用音乐翻译网络(Mor等人，2019)中的wavenet-type自回归译码器。

在本实验中，我们使用作者提供的预训练的通用音乐编码器，将16kHz的原始音频转换为64通道的潜在码序列，在时间维上降低采样因子800。这意味着该域独立潜在表示的信息压缩率为12.5。仅使用来自目标音乐域的数据，我们的MelGAN解码器被训练来从我们前面描述的GAN设置中的潜在代码序列重建原始波形。我们调整模型的超参数，得到10,10,2,2,2的上采样因子，以达到输入分辨率。对于MusicNet上的每个选定域(Thickstun et al.，2018)，在可用数据上的RTX2080 Ti GPU上训练一个解码器4天。

添加了MelGAN解码器的音乐翻译网络能够以良好的质量将任意音乐域的音乐翻译到它所训练的目标域。我们将我们模型中的定性样本与原始模型进行比较:<https://melgan-neurips.github.io>。在RTX2080 Ti GPU上，增强版只需160毫秒就能翻译1秒的输入音乐音频，比在相同硬件上的原始版快2500倍。

##### 3.4 VQ-VAE非自回归解码器(\*)

为了进一步确定我们方法的通用性，我们将矢量量化的VAEs (van den Oord et al.， 2017)中的解码器替换为我们提出的反向学习解码器。VQ-VAE是一种变分自编码器，它产生一个下采样离散潜编码的输入。VQ-VAE使用一个高容量自回归波网解码器来学习数据条件$p(x|z_q)$。

下图显示了用于音乐生成任务的VQ-VAE的改编版本。在我们的变体中，我们使用两个编码器。该本地编码器将该音频序列编码成一个64向下采样的时间序列$z_e$。然后使用码本将该序列中的每个向量映射到512个量化向量中的1个。这与(van den Oord等人，2017)中提出的结构相同。第二个编码器输出一个全局连续值潜行向量$y$。

<div align=center>
    <img src="zh-cn/img/ch7/01/p7.png" /> 
</div>

我们展示了无条件钢琴音乐生成后续的定性样本(Dieleman等人，2018)，其中我们在原始音频尺度上学习单层VQ-VAE，并使用一个普通的自回归模型(4层LSTM, 1024单元)来学习离散序列上的先验。我们无条件地使用训练好的递归先验模型对$z_q$进行采样，对y进行单位高斯分布的采样。定性地说，在相同的离散延迟序列的条件下，从全局潜在先验分布中采样会导致低电平的波形变化，如相移，但从感觉上输出听起来非常相似。通过局部编码器($z_q$)获取的离散潜在信息被高度压缩，全局潜在信息能更好地捕捉到数据条件$p(x|z_q,y)$中的随机性，因此对提高重构质量至关重要。我们使用大小为256的潜向量，并使用与mel-谱图反演实验相同的超参数进行训练。我们使用4x、4x、2x和2x比率的上采样层来实现64x上采样。

#### 4.结论及未来工作

我们介绍了一种专为条件音频合成而设计的GAN结构，并对其进行了定性和定量的验证，证明了所提方法的有效性和通用性。我们的模型有以下优点:它非常轻量，可以在单台桌面GPU上快速训练，并且在推理时非常快。我们希望我们的生成器可以是一个即插即用的替代方案，在任何较高水平的音频相关任务中计算量大的替代方案。

虽然该模型能很好地适应训练和生成变长序列的任务，但它受到时间对齐条件信息要求的限制。实际上，它被设计用于输出序列长度是输入序列长度的一个因数的情况下，而在实践中并不总是这样。同样，基于成对的ground truth数据进行特征匹配也存在局限性，因为在某些情况下不可行。对于无条件综合，所提出的模型需要将一系列条件变量的学习延迟到其他更适合的方法，如VQ-VAE。学习用于音频的高质量无条件GAN是未来工作的一个非常有趣的方向，我们相信这将受益于结合在本工作中介绍的特定架构的选择。

#### 附录

**附录A: 模型架构**

Mel光谱图反演任务的生成器和判别器架构：

<div align=center>
    <img src="zh-cn/img/ch7/01/p8.png" /> 
</div>

残差堆叠的架构：

<div align=center>
    <img src="zh-cn/img/ch7/01/p9.png" /> 
</div>

**附录B: 超参数和训练细节**

我们在所有实验中使用的批量大小为16。 Adam的学习速率为0.0001，$\beta_1 = 0.5,\beta_2 = 0.9$被用作生成器和判别器的优化器。我们使用10作为特征匹配损失项的系数。 我们使用pytorch来实现我们的模型，该模型的源代码随此提交一起提供。对于VQGAN实验，我们使用大小为256的全局潜矢量，其中KL项限制在1.0以下，以避免后部崩溃。 我们在Nvidia GTX1080Ti或GTX 2080Ti上训练了我们的模型。在补充材料中， 我们发现我们的模型在训练的很早就开始产生可理解的样本。

**附录C: 评价方法- MOS**

我们进行了平均意见评分（MOS）测试，以比较我们的模型与竞争体系结构的性能。 我们通过收集由不同模型生成的样本以及一些原始样本来构建测试。 在训练过程中没有看到所有生成的样本。 MOS得分是根据200个人的总体计算得出的：要求他们每个人通过对1到5个样品进行评分来盲目评估从该样品池中随机抽取的15个样品的子集。对样品进行展示并一次对其进行评级。 测试是使用Amazon Mechanical Turk进行的众包，我们要求测试人员戴上耳机并讲英语。 在收集所有评估之后，通过平均分数$m_i$来估计模型$i$的MOS分数$\mu_i$。 此外，我们计算得分的$95％$置信区间。$\hat{\sigma }_ i$是所收集分数的标准偏差。

$$\hat{\mu}_ {i}=\frac{1}{N_{i}} \sum_{k=1}^{N_{i}} m_{i, k}$$
$$C I_{i}=\left[\hat{\mu}_ {i}-1.96 \frac{\hat{\sigma}_ {i}}{\sqrt{N_{i}}}, \hat{\mu}_ {i}+1.96 \frac{\hat{\sigma}_ {i}}{\sqrt{N_ {i}}}\right]$$


#### 参考文献
Arik, S. Ö., Chrzanowski, M., Coates, A., Diamos, G., Gibiansky, A., Kang, Y., Li, X., Miller, J.,Ng, A., Raiman, J., et al. Deep voice: Real-time neural text-to-speech. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 195–204. JMLR. org, 2017.

Chan, C., Ginosar, S., Zhou, T., and Efros, A. A. Everybody dance now. arXiv preprint arXiv:1808.07371, 2018.

Dieleman, S., van den Oord, A., and Simonyan, K. The challenge of realistic music generation:modelling raw audio at scale. In Advances in Neural Information Processing Systems, pp. 7989–7999, 2018.

Donahue, C., McAuley, J., and Puckette, M. Adversarial audio synthesis. arXiv preprint arXiv:1802.04208, 2018a.

Donahue, C., McAuley, J., and Puckette, M. Adversarial audio synthesis. arXiv preprint arXiv:1802.04208, 2018b.

Dosovitskiy, A. and Brox, T. Generating images with perceptual similarity metrics based on deep networks. In Advances in neural information processing systems, pp. 658–666, 2016.

Engel, J., Resnick, C., Roberts, A., Dieleman, S., Norouzi, M., Eck, D., and Simonyan, K. Neural audio synthesis of musical notes with wavenet autoencoders. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1068–1077. JMLR. org, 2017.

Engel, J., Agrawal, K. K., Chen, S., Gulrajani, I., Donahue, C., and Roberts, A. Gansynth: Adversarial neural audio synthesis. arXiv preprint arXiv:1902.08710, 2019a.

Engel, J., Agrawal, K. K., Chen, S., Gulrajani, I., Donahue, C., and Roberts, A. Gansynth: Adversarial neural audio synthesis. arXiv preprint arXiv:1902.08710, 2019b.

Gatys, L. A., Ecker, A. S., and Bethge, M. Image style transfer using convolutional neural networks.In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2414–2423,2016.

Gibiansky, A., Arik, S., Diamos, G., Miller, J., Peng, K., Ping, W., Raiman, J., and Zhou, Y. Deep voice 2: Multi-speaker neural text-to-speech. In Advances in neural information processing systems, pp. 2962–2970, 2017.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In Advances in neural information processing systems, pp.2672–2680, 2014.

Griffin, D. and Lim, J. Signal estimation from modified short-time fourier transform. IEEE Transactions on Acoustics, Speech, and Signal Processing, 32(2):236–243, 1984.

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. C. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5767–5777, 2017.

Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.1125–1134, 2017.

Ito, K. The lj speech dataset. https://keithito.com/LJ-Speech-Dataset/, 2017.

Jia, Z., Maggioni, M., Staiger, B., and Scarpazza, D. P. Dissecting the nvidia volta gpu architecture via microbenchmarking. arXiv preprint arXiv:1804.06826, 2018.

Johnson, J., Alahi, A., and Fei-Fei, L. Perceptual losses for real-time style transfer and superresolution.In European conference on computer vision, pp. 694–711. Springer, 2016.

Kalchbrenner, N., Elsen, E., Simonyan, K., Noury, S., Casagrande, N., Lockhart, E., Stimberg, F.,Oord, A. v. d., Dieleman, S., and Kavukcuoglu, K. Efficient neural audio synthesis. arXiv preprint arXiv:1802.08435, 2018.

Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of gans for improved quality,stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. arXiv preprint arXiv:1812.04948, 2018.

Kingma, D. P. and Dhariwal, P. Glow: Generative flow with invertible 1x1 convolutions. In Advances in Neural Information Processing Systems, pp. 10215–10224, 2018.

Larsen, A. B. L., Sønderby, S. K., Larochelle, H., and Winther, O. Autoencoding beyond pixels using a learned similarity metric. arXiv preprint arXiv:1512.09300, 2015.

Lim, J. H. and Ye, J. C. Geometric gan. arXiv preprint arXiv:1705.02894, 2017.

Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z., and Paul Smolley, S. Least squares generative adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision,pp. 2794–2802, 2017.

Mathieu, M., Couprie, C., and LeCun, Y. Deep multi-scale video prediction beyond mean square error. arXiv preprint arXiv:1511.05440, 2015.

Mehri, S., Kumar, K., Gulrajani, I., Kumar, R., Jain, S., Sotelo, J., Courville, A., and Bengio,Y. Samplernn: An unconditional end-to-end neural audio generation model. arXiv preprint arXiv:1612.07837, 2016.

Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

Mor, N.,Wolf, L., Polyak, A., and Taigman, Y. Autoencoder-based music translation. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=
HJGkisCcKm.

MORISE, M., YOKOMORI, F., and OZAWA, K. World: A vocoder-based high-quality speech synthesis system for real-time applications. IEICE Transactions on Information and Systems,E99.D(7):1877–1884, 2016. doi: 10.1587/transinf.2015EDP7457.

Neekhara, P., Donahue, C., Puckette, M., Dubnov, S., and McAuley, J. Expediting tts synthesis with adversarial vocoding. arXiv preprint arXiv:1904.07944, 2019.

Odena, A., Dumoulin, V., and Olah, C. Deconvolution and checkerboard artifacts. Distill, 2016. doi:10.23915/distill.00003. URL http://distill.pub/2016/deconv-checkerboard.Oord,
A. v. d., Li, Y., Babuschkin, I., Simonyan, K., Vinyals, O., Kavukcuoglu, K., Driessche, G.v. d., Lockhart, E., Cobo, L. C., Stimberg, F., et al. Parallel wavenet: Fast high-fidelity speech synthesis. arXiv preprint arXiv:1711.10433, 2017.

Park, T., Liu, M.-Y., Wang, T.-C., and Zhu, J.-Y. Semantic image synthesis with spatially-adaptive normalization. arXiv preprint arXiv:1903.07291, 2019.

Ping, W., Peng, K., Gibiansky, A., Arik, S. O., Kannan, A., Narang, S., Raiman, J., and Miller,J. Deep voice 3: Scaling text-to-speech with convolutional sequence learning. arXiv preprint arXiv:1710.07654, 2017.

Ping, W., Peng, K., and Chen, J. Clarinet: Parallel wave generation in end-to-end text-to-speech.arXiv preprint arXiv:1807.07281, 2018.

Prenger, R., Valle, R., and Catanzaro, B. Waveglow: A flow-based generative network for speech synthesis. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3617–3621. IEEE, 2019.

Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pp.901–909, 2016.

Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., Chen, Z., Zhang, Y., Wang,Y., Skerrv-Ryan, R., et al. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP), pp. 4779–4783. IEEE, 2018.

Sotelo, J., Mehri, S., Kumar, K., Santos, J. F., Kastner, K., Courville, A., and Bengio, Y. Char2wav: End-to-end speech synthesis. 2017.

Thickstun, J., Harchaoui, Z., Foster, D. P., and Kakade, S. M. Invariances and data augmentation for supervised music transcription. In International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

Ulyanov, D., Vedaldi, A., and Lempitsky, V. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.

Van Den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N.,Senior, A. W., and Kavukcuoglu, K. Wavenet: A generative model for raw audio. SSW, 125, 2016.

van den Oord, A., Vinyals, O., et al. Neural discrete representation learning. In Advances in Neural Information Processing Systems, pp. 6306–6315, 2017.

Veaux, C., Yamagishi, J., MacDonald, K., et al. Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit. University of Edinburgh. The Centre for Speech Technology Research(CSTR), 2017.

Wang, T.-C., Liu, M.-Y., Zhu, J.-Y., Liu, G., Tao, A., Kautz, J., and Catanzaro, B. Video-to-video synthesis. In Advances in Neural Information Processing Systems (NIPS), 2018a.

Wang, T.-C., Liu, M.-Y., Zhu, J.-Y., Tao, A., Kautz, J., and Catanzaro, B. High-resolution image synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8798–8807, 2018b.

Wang, Y., Skerry-Ryan, R., Stanton, D., Wu, Y., Weiss, R. J., Jaitly, N., Yang, Z., Xiao, Y., Chen, Z.,Bengio, S., et al. Tacotron: Towards end-to-end speech synthesis. arXiv preprint arXiv:1703.10135,2017.

Yamamoto, R. r9y9/wavenet_vocoder, Oct 2019. URL https://github.com/r9y9/wavenet_vocoder.

Yamamoto, R., Song, E., and Kim, J.-M. Probability density distillation with generative adversarial networks for high-quality parallel waveform generation. arXiv preprint arXiv:1904.04472, 2019.

Zhang, H., Goodfellow, I., Metaxas, D., and Odena, A. Self-attention generative adversarial networks. arXiv preprint arXiv:1805.08318, 2018.

Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pp. 2223–2232, 2017.

------

### 2. Multi-Band MelGAN:Faster Waveform Generation for High-Qualty Text-to-Speech

!> https://arxiv.org/abs/2005.05106

<!-- https://zhuanlan.zhihu.com/p/319799626 -->
<!-- https://zhuanlan.zhihu.com/p/350333116 -->

MelGAN的结构已经在前文详细叙述了 ，该paper主要着点于针对MelGAN的一种加速优化方法，甚至在精细调参的情况下，语音合成效果还能小优于 MelGAN。
而达到这种有效改进的核心是 **切分频段**，这里使用的方法是PQMF，通过分频段减少上采样计算量，并且针对特定频域合成，优化合成速率和质量。

首先重点讲解PQMF，关于这部分资料的确较少，本文会从$z$变换开始介绍，逐步讲解PQMF的信号处理逻辑，当然，最好还是去阅读这方面原文，推荐是 《Introduction to digital audio coding and standards》 这本书的第四章。 然后基于 PQMF Multi-band MelGAN 的改进 ，比如 loss 的调整，也会在本文中叙述。

#### 1.性能比较

Multi-Band MelGAN 论文中，可以从下表中看出，相比于MelGAN，MB-MelGAN参数量降低了一倍多，合成速率明显加快。

<div align=center>
    <img src="zh-cn/img/ch7/02/p1.png" /> 
</div>

同时合成语音的效果，不降反升，但我觉得这其中有一定层度是取决于 **STFT loss** （也就是他们改进的loss）的作用导致的

<div align=center>
    <img src="zh-cn/img/ch7/02/p2.png" /> 
</div>

#### 2.PQMF

首先在讲解分频方法 PQMF之前，我们要对为什么分频有个理解，甚至更基础的频域是用来干嘛的这一点，也需要有个实际的认识。直接说结论，频域是用来压缩语音的，并且频谱这种载体有利于编码，所以在信号领域应用广泛。最基础的频域就是傅里叶变换后得到的频域，实际是一定长度的复数向量，其中包含了功率和相位的信息。但为方便计算，把复数取模长转为实数，这个实数其实也就是能量，也就是我们常说的频谱（一般还有log计算，正则化数量级）。这其实也就是Vocoder存在的最大意义，如何把能量谱还原回去。

频谱中的每一维能量实际是一一对应一个频率的，所以可以人为的将不同频段在频域上简单有效的分离。这种方法叫做 **子频带编码(sub-band coding)**，将频带表征为多个子频带而非全频带的频域特征。

多个子频带这种表征在许多领域都有相应应用，比如对于音频信号分频段进行量化压缩，在语音合成中的作用主要是通过多个子频带分别进行转换，达到加速推断的目的，在保持高质量的合成语音的同时减少计算量

这种子频带编码的时频映射是通过包含多个滤波器的滤波器组实现的，每个滤波器取不同的下采样倍率，通过滤波器组后对信号进行量化编码。在解码合成的阶段将信号上采样，再通过还原滤波器组，还原信号。这个过程可以抽象表述如下图

<div align=center>
    <img src="zh-cn/img/ch7/02/p3.jpg" /> 
</div>

在Neural Vocoder 当中运用这种子频带编码，语音信号经过分析滤波器组(analysis filters)然后训练模型，在推断过程中再应用合成滤波器组(synthesis filters)，由于子频带之间条件独立(conditionally independently)，而每个频带下采样了相应倍数，这样模型结构实际是进行了对应压缩，从而减少计算量。

##### 2.1 z变换

因为下文中的部分推导需要用到z变换的相关性质，这里简要叙述下z变换。下文有部分公式，别排斥，我觉得这已经是最方便理解的方法了，而且公式有利于直接改成代码。

首先考虑一个采样率为$F_s=1/T_s$的信号$x$,其傅里叶变换可表示为：
$$X(f)=\frac{1}{F_s}\sum_{n=-\infty}^{\infty}x(nT_s)e^{-j2\pi\frac{nf}{F_s}}$$
其中$f$为对应的品PV，取值范围为$-F_s/2$到$F_s/2$,$T_s$为最大周期有$F_s=1/T_s$,$j$表示虚数单位。定义一个频率$f$到复数空间的映射$z(f)=e^{j2\pi f/T_s}$

将z带入傅里叶变换，有$X(f)=\frac{1}{F_s}\sum_{n=-\infty}^{\infty}x(nT_s)z(f)^{-n}$,因为$F_s,T_s$都是和信号采样率相关的常数，令$x[n]-x(nT_s),z=z(f)$,同样$1/F_s$也是常数，因此可以得到傅里叶变换的z变换表达式：
$$X(z)=\sum_{n=-\infty}^{\infty}x[n]z^{-n}$$

在下面的叙述中会用到几条关于z变换的性质，分别是 
1. 两离散序列的和的z变换 等于 两序列z变换的和；
2. 两离散序列的卷积的z变换 等于 两序列z变换的积； 
3. 序列的延迟可以通过z变换表示,对于$y[n]=x[n-D]$则有
$$Y(z)=\sum_{n=-\infty}^{\infty}y[n]z^{-n}=\sum_{n=-\infty}^{\infty}x[n-D]z^{-n}=\sum_{n=-\infty}^{\infty}x[m]z^{-m+D}=X(z)z^{D}$$
4. 序列的下采样K倍的变换可以用z变换表示，对于 $y[n]=x[nK]$，有
$$Y(z)=\frac{1}{K}\sum_{r=0}^{K}X(z^{1/K}e^{-j2\pi r/K})$$
5. 同样序列的上采样K倍的变换可以用z变换表示，对于$y[n]=x[m],n=mK$,有$Y(z)=X(z^K)$

##### 2.2 双通道重构滤波器组

基于以上知识可以粗略的开始PQMF的介绍了，但在进行多频带分解之前，我们来考虑一个简单情况 双通道重构滤波器组（two-channel perfect reconstruction filter bank），将信号分解为两个频带。理想情况下，两个滤波器完美分割频带信息，也就是说这里的下采样和上采样倍率都为2

<div align=center>
    <img src="zh-cn/img/ch7/02/p4.jpg" /> 
</div>

那么不考虑中间特征的量化去噪等操作，可以将合成过程写作z换形式：
$$X^{'}(z)=Y_0(z^2)G_0(z)+Y_1(z^2)G_1(z)$$

分解的过程也z变换公式化为：

$$Y_i(z)=\frac{1}{2}(H_i(z^{1/2})X(z^{1/2})+H_i(-z^{1/2})X(-z^{1/2}) )$$

将分解带入合成公式，有：

$$X^{'}(z)=\frac{1}{2}(H_0(z)G_0(z)+H_1(z)G_1(z))X(z) + \frac{1}{2}(H_0(-z)G_0(z)+H_1(-z)G_1(z))X(-z)$$

由于我们想要完美重构会原始语音，也就是要求$x^{'}(z)=X(z)$,那么带入能够得到：
$$\frac{1}{2}(H_0(z)G_0(z)+H_1(z)G_1(z))=1$$
$$\frac{1}{2}(H_0(-z)G_0(z)+H_1(-z)G_1(z))=0$$

这样解得$G$和$H$之间的关系，有$G_0(z)=-H_1(z)$以及$G_1(z)=H_0(z)$。 再考虑一种更为简单的滤波器组，正交镜像滤波器组(Quadrature Mirror Filters, QMF)，这种滤波器组有一个性质就是$H_1(z)=-H_0(-z)$,采用这种类型滤波器组，双通道重构滤波器将更加容易被构造，有 
$G_0(z)=H_0(z)$以及$G_0(z)=H_0(-z)$,这样就可以只需要设计一个滤波器就能得到对应的双通道重构滤波器组了。

##### 2.3 多子频带滤波器

将双通道的情况推广到多通道的情况，基于“正交镜像滤波器组可以完美重构信号”这一结论。如果有正交镜像滤波器组的高维推广，就可以构造多通道的重构滤波器组，从而实现语音信号的多子频带的分解与合成。

正交镜像滤波器组的高维推广的近似解也就是pseudo-QMF，由于相关证明过多这里直接给出其表达式，对于K通道的PQMF滤波器组$k=0,...,K-1$,有如下形式：
$$h_k[n]=h[n]cos(\pi (\frac{k+\frac{1}{2}}{K})(n-\frac{N-1}{2})+\Phi_k)$$
$$g_k[n]=h_k[N-1-n]$$
式子中的$N$表示$h[n]$的长度，相位$\Phi_k$满足$\Phi_k-\Phi_{k-1}=\frac{\pi}{2}(2r+1)$,其中$r$为确定整数，在Multi-Band MelGAN中取0。

确定好PQMF滤波器组形式后现在需要确定其中 $h[n]$滤波器的形式，这里采用的是凯撒窗(Kaiser window)原型滤波器(prototype filter) 。原型滤波器是一种电子滤波器设计，用作模板以针对特定应用生成修改的滤波器设计。它们是无量纲化设计的一个示例，通过该设计可以缩放或转换所需的滤波器。可以通过凯撒窗(Kaiser window)将原型滤波器的构造限制在单一参数变量情况，从而简化原型滤波器的设计过程。设计好的滤波器形式如下:
$$f_i(n)=\frac{sin(\omega_c(n-0.5N))}{\pi(n-0.5N)}$$

这里的$\omega_c$为截止频率是一个人为设定的常数，将上述滤波器通过凯撒窗(Kaiser window)将得到原型滤波器表达式$h(n)=f_i(n)\omega(n)$。 凯撒窗凸显主要频率段能量，同时削弱其他频段能量，表达式如下：
$$\omega(n)=\frac{I_o(\beta)\sqrt{1-((n-0.5N)/0.5N)^2}}{I_o(\beta)}$$

$I_o()$为零阶修正贝塞尔函数(zeroth-order modified Bessel function)，用于控制主要频率段宽窄，其中$\beta$为设定的常数。
$$I_o(x)=1+\sum_{k=1}^{\infty}(\frac{(0.5x)^k}{k!})^2$$

观察最后原型滤波器的公式，其实只有三个需要人为设定的数值，也就是说通过设定截止频率$\omega_c$,阶数(taps)$N$,以及kaiser窗的参数 
$\beta$,即可确定原型滤波器的具体表达，从而构造PQMF滤波器组。

通过构造好的PQMF，可以对语音分离频段然后进行子频带语音的编码与解码，最后再重构回语音。同样也可以将子频段语音作为参照，对应使用Vocoder合成子频带语音，然后再用PQMF重构滤波器组合成最终语音，实现语音合成的加速。

#### 3.Multi-Band MelGAN

Multi-Band MelGAN主要的改进分为两方面，
1. 一方面由于MelGAN生成器模型中参数量占比最大的是上采样层，因而可以通过合成子频段再将多个子频段语音通过PQMF重构滤波器组合成最终语音来加速语音合成。 
2. 另一方面就是加入了STFT loss 优化训练。

##### 3.1 STFT loss

搜狗发现使用MelGAN的生成器loss，Multi-Band MelGAN收敛非常慢，因而使用了一种更为有效的基于快速傅里叶变换(STFT)的loss。STFT loss由两部分组成，一部分考虑频谱收敛性(spectral convergence)，另一部分考虑对数谱能量(log STFT magnitude)之间的关系，分别对应公式

$$L_{sc}(x,\tilde{x})=\frac{\parallel \mid STFT(x)\mid-\mid STFT(\tilde{x})\mid \parallel_F}{\parallel \mid STFT(x)\mid \parallel_F}$$

$$L_{mag}(x,\tilde{x})=\frac{1}{N}\parallel log \mid STFT(x)\mid -log\mid STFT(\tilde{x})\mid \parallel_1 $$

其中$\parallel A \parallel_F$表示$F$范数，$|STFT()|$表示语音快速傅里叶变换后的能量(magnitudes)，$N$表示傅里叶变换后的能量的长度，即fft_length/2。

使用不同STFT参数配置(比如fft长度，窗长，帧长)得到不同分辨率(multi-resolution)的STFT loss，使用M个不同分辨率的STFT loss整合得到

$$L_{mr\_stft}(G)=\mathbb{E}_ {x,\tilde{x}}[\frac{1}{M}\sum_{m=1}^{M}(L_{sc}^m(x,\tilde{x})+L_{mag}^{m}(x,\tilde{x}))]$$

将多分辨率STFT loss 替代MelGAN的feature matching项，作为生成器loss：

$$min_G \mathbb{E}_ {s,z}[\lambda \sum_{k=1}^K(D_k(G(s,z))-1)^2] + \mathbb{E}_ {s}[L_{mr\_stft}(G)]  $$

其中$s$表示输入特征Mel Spectrogram，$z$表示高斯噪音。

同时为了解决训练收敛较慢的问题以及判别器性能优于生成器可能导致无法更新梯度，Multi-Band MelGAN的训练采用先预训练一定步数的生成器，当逐渐可以合成语音了再加入判别器一同训练。

##### 3.2 整体结构

模型结构方面其实MB-MelGAN相比于MelGAN并未加以改变，通过组合上述 PQMF 和 STFT loss 可以得到如下的结构图

<div align=center>
    <img src="zh-cn/img/ch7/02/p5.png" /> 
</div>

具体的每个模块的构造于MelGAN 无异,可以参考 MelGAN.

#### 参考文献

Multi-band melgan: Faster waveform generation for high-quality text-to-speech.

Melgan: Generative adversarial networks for conditional waveform synthesis.

Introduction to digital audio coding and standards

Highresolution image synthesis and semantic manipulation with conditional gans.

A kaiser window approach for the design of prototype filters of cosine modulated filterbanks.

------
### 3. Parallel WaveGAN:A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-Resolution Spectrogram

!> https://arxiv.org/pdf/1910.11480.pdf

!> https://github.com/kan-bayashi/ParallelWaveGAN

<!-- https://blog.csdn.net/weixin_42721167/article/details/119451215 -->

#### 摘要

我们提出了Parallel WaveGAN，一种使用生成式对抗网络的无蒸馏、快速和占用空间小的波形生成方法。该方法通过联合优化多分辨率谱图和对抗Loss函数来训练非自回归WaveNet，能够有效地捕捉真实语音波形的时频分布。由于我们的方法不需要在传统的师生框架中使用密度蒸馏，整个模型易于训练。此外，我们的模型在结构紧凑的情况下也能生成高保真语音。其中，提出的并行WaveGAN只有1.44M个参数，在单个GPU环境下生成24kHz语音波形的速度比实时速度快28.68倍。感知听力测试结果表明，本文提出的方法在基于Transformer的文本到语音框架中获得了4.16的平均意见得分(MOS)，与基于蒸馏的最好的Parallel WaveNet系统做对比。
       
关键词 - 神经声码器，TTS，GAN，Parallel WaveNet，Transformer

#### 1.介绍

文本到语音(TTS)框架中的深度生成模型显著提高了合成语音信号的质量(《Statistical parametric speech synthesis using deep neural networks》，《Effective spectral and excitation modeling techniques for LSTM-RNN-based speech synthesis systems》，《Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions》)。值得注意的是，自回归生成模型如WaveNet已经显示出比传统参数声码器更优越的性能(《WaveNet: A generative model for raw audio》，《A comparison of recent waveform generation and acoustic modeling methods for neural-network-based speech synthesis》，《Speaker-dependent WaveNet vocoder》，《An investigation of multi-speaker training for WaveNet vocoder》，《Excitnet vocoder: A neural excitation model for parametric speech synthesis systems》)。然而，由于自回归的特性，其推理速度较慢，因此在实时场景中的应用受到限制。

解决这种局限性的一种方法是利用基于教师-学生框架的快速波形生成方法(《Parallel WaveNet: Fast high-fidelity speech synthesis》，《ClariNet: Parallel wave generation in end-to-end text-to-speech》，《Probability density distillation with generative adversarial networks for high-quality parallel waveform generation》)。在这个框架中，定义为概率密度蒸馏的桥梁将自回归教师WaveNet的知识转移到基于逆自回归流(IAF)的学生模型(《Improved variational inference with inverse autoregressive flow》)。虽然IAF学生能够以合理的感知质量实现实时生成语音，但在训练过程中仍然存在问题：不仅需要一个训练良好的教师模型，还需要一种试错方法来优化复杂的密度蒸馏过程。

为了克服上述问题，我们提出了一种基于生成式对抗网络(generative adversarial network, GAN)(《Generative adversarial nets》)的并行波形生成方法——Parallel WaveGAN。与传统的基于蒸馏的方法不同，Parallel WaveGAN不需要两个阶段，而是连续的教师-学生训练过程。
该方法仅通过优化多分辨率短时傅里叶变换(STFT)和对抗损失函数的组合来训练非自回归WaveNet模型，使该模型能够有效地捕获真实语音波形的时频分布。因此，整个训练过程比传统方法简单得多，并且模型参数较少，可以产生自然发声的语音波形。我们的贡献总结如下：
1. 提出了一种多分辨率短时傅立叶变换损失和波形域对抗损失的联合训练方法。该方法既适用于传统的基于蒸馏的Parallel WaveNet(如ClariNet)，也适用于提出的无蒸馏Parallel WaveGAN。
2. 由于所提出的Parallel WaveGAN可以在没有任何教师-学生框架的情况下进行简单的训练，因此我们的方法大大减少了训练和推理时间。特别是训练过程变得快4.82倍(从13.5天到2.8天，使用两个NVIDIA Telsa V100 GPU)和推理过程变得快1.96倍(从14.62到28.68 倍实时速度生成24kHz语音波形，使用单个NVIDIA Telsa V100 GPU)，与传统的ClariNet模型相比。
3. 我们将提出的Parallel WaveGAN与基于Transformer的TTS声学模型相结合(《Attention is all you need》，《Neural speech synthesis with Transformer network》，《FastSpeech: Fast, robust and controllable text to speech》)。感知听力测试结果表明，提出的Parallel WaveGAN模型达到了4.16 MOS，与基于蒸馏的ClariNet模型相比具有一定的竞争力。

#### 2.相关工作

在Parallel WaveNet框架中使用GAN的想法并不新鲜。在我们之前的工作中，IAF学生模型被纳入到生成器中，并通过最小化对抗损失以及Kullback-Leibler散度(KLD)和辅助损失(《Probability density distillation with generative adversarial networks for high-quality parallel waveform generation》)进行联合优化。由于GAN学习了真实语音信号的分布，该方法显著提高了合成信号的感知质量。但基于密度精馏的训练阶段复杂，限制了其应用。

我们的目标是尽量减少训练传统教师-学生框架的两阶段管道的努力。换句话说，我们提出了一种新的方法训练Parallel WaveNet不需要任何蒸馏过程。Juvela等人(《GELP: GAN-excited linear prediction for speech synthesis from melspectrogram》)也提出了类似的方法(例如GAN激发线性预测，GELP)，利用对抗式训练方法产生声门刺激。然而，由于GELP需要线性预测(LP)参数来将声门激励转换为语音波形，因此，当LP参数包含TTS声学模型不可避免的误差时，可能会出现质量下降。为了避免这个问题，我们的方法是直接估计语音波形。由于很难捕捉语音信号的动态特性，包括声带运动和声道共振(分别由GELP中的声门兴奋和LP参数表示)，**我们提出了对抗损失和多分辨率STFT损失的联合优化方法**，以捕获真实语音信号的时频分布。因此，即使参数较少，整个模型也易于训练，同时有效地减少了推断时间，提高了合成语音的感知质量。

#### 3.方法

##### 3.1 基于GAN的并行波形生成

GAN是生成模型，由两个独立的神经网络组成:生成器(G)和判别器(D)(《Generative adversarial nets》)。在我们的方法中，一个基于WaveNet型以辅助特征(如梅尔谱图)为条件作为生成器，它将输入噪声并行地转换为输出波形。生成器与原始WaveNet的不同之处在于：(1)我们使用非因果卷积而不是因果卷积；(2)输入为高斯分布的随机噪声；(3)模型在训练和推理阶段都是非自回归的。

生成器学习真实波形的分布，通过试图欺骗判别器来识别生成器样本为真实的。这个过程是通过最小化对抗损失(Ladv)来完成的，如下所示：

$$L_{adv}(G,D)=\mathbb{E}_ {z\sim N(0,1)}[(1-D(G(z)))^2]$$

其中$z$为输入白噪声。注意，为了简洁，$G$的辅助特性被省略了。

另一方面，利用以下优化准则训练判别器，在将Ground Truth分类为真实的同时，将生成的样本正确分类为假样本：

$$L_{D}(G,D)=\mathbb{E}_ {x\sim P_{data}}[(1-D(x))^2]+\mathbb{E}_ {z\sim N(0,1)}[(1-D(G(z)))^2]$$

式中，$x$和$P_{data}$分别表示目标波形及其分布。

##### 3.2 多分辨率STFT辅助损耗

为了提高对抗训练过程的稳定性和效率，我们提出了一种多分辨率短时傅里叶变换辅助损失算法。下图显示了我们将多分辨率短时傅立叶变换损失与3.1节中描述的对抗训练方法相结合的框架。

<div align=center>
    <img src="zh-cn/img/ch7/03/p1.png" /> 
</div>

与前面的工作(《Probability density distillation with generative adversarial networks for high-quality parallel waveform generation》)相似，我们定义单个**STFT损失**如下：

$$L_s(G)=\mathbb{E}_ {z\sim p(z),x \sim P_{data}}[L_{sc}(x,\hat{x})+L_{mag}(x,\hat{x})]$$

其中$\hat{x}$表示生成的样本$G(x)$,$L_{sc}$和$L_{mag}$分别表示光谱收敛性和对数STFT幅值损失，定义如下(《Fast spectrogram inversion using multi-head convolutional neural networks》)：

$$L_{sc}=\frac{|| |STFT(x)|-|STFT(\hat{x})| ||_ F}{|| |STFT(x)| ||_ F}$$
$$L_{mag}=\frac{1}{N}||log|STFT(x)|-log|STFT(\hat{x})| ||_ 1$$

其中$||\cdot||_ F$和$||\cdot||_ 1$分别为Frobenius范数和$L_1$范数;$|STFT(\cdot)|$和$N$分别表示STFT幅值和幅值中元素的个数。

我们的多分辨率STFT损失是不同分析参数(即FFT大小、窗口大小和帧移)下STFT损失的总和。设M为STFT损耗数，则多分辨率STFT辅助损耗$L_{aux}$表示为：
$$L_{aux}(G)=\frac{1}{M}\sum^{M}_ {m=1}L_{s}^{(m)}(G)$$

在基于STFT的信号时频表示中，存在时间分辨率和频率分辨率之间的权衡；例如，增加窗口大小可以获得更高的频率分辨率，而降低时间分辨率(《The wavelet transform, time-frequency localization and signal analysis》)。通过结合不同分析参数的多种短时傅立叶变换损耗，极大地帮助生成器了解语音(《Neural source-filterbased waveform model for statistical parametric speech synthesis》)的时频特性。此外，它还防止生成器过拟合到固定的STFT表示，这可能导致在波形域的次优性能。

我们对生成器的最终损失函数定义为多分辨率STFT损失和对抗损失的线性组合，如下所示：
$$L_G(G,D)=L_{aux}(G)+\lambda_{adv}L_{adv}(G,D)$$
其中$\lambda_{adv}$表示平衡两个损失项的超参数。通过对波形域对抗损失和多分辨率STFT损失的联合优化，可以有效地了解真实语音波形的分布。

#### 4.实验

##### 4.1 实验设置

**数据集**： 在实验中，我们使用了一个由一位女性日语专业人士记录的语音和平实平衡的语料库。语音信号以24kHz采样，每个采样用16比特量化。共使用11449个话语(23.09小时)进行训练，使用250个话语(0.35小时)进行验证，使用250个话语(0.34小时)进行评价。提取限频(70 ~ 8000Hz)的80波段log-mel谱图作为波形生成模型(即局部条件(《WaveNet: A generative model for raw audio》))的输入辅助特征。帧长设置为50ms，移位长度设置为12.5ms。训练前将梅尔谱图特征归一化，使其均值和单位方差均为零

**模型细节**：
提出的Parallel WaveGAN由30层空洞残差卷积块组成，以指数方式增加三个扩张周期。剩余通道和跳跃通道数设为64，卷积滤波器大小设为3。该判别器由10层非因果空洞的一维卷积组成，具有泄漏的ReLU激活函数(α = 0.2)。步幅设置为1，从1到8的一维卷积应用线性增加的扩张，除了第一层和最后一层。通道的数量和滤波器的大小与生成器相同。我们对生成器和判别器(《Weight normalization: A simple reparameterization to accelerate training of deep neural networks》)的所有卷积层都进行了权值归一化。

在训练阶段，由三种不同的STFT损失之和计算多分辨率STFT损失，如下图所示。判别器损失由判别器的每个时间步长标量预测的平均值计算。根据初步实验结果，将$L_G(G,D)$中的$\lambda_{adv}$设定为4.0。用RAdam优化器($\epsilon = 1e^{-6}$对模型进行400K步的训练，以稳定训练(《On the variance of the adaptive learning rate and beyond》)。注意，前100K步的判别器是固定的，之后联合训练两个模型。batch size大小设置为8，每个音频剪辑的长度设置为24K时间样本(1.0秒)。对生成器和判别器分别设置初始学习率为0.0001和0.00005。每200K步学习率降低一半。

<div align=center>
    <img src="zh-cn/img/ch7/03/p2.png" /> 
</div>

作为基线系统，我们同时使用了自回归Gaussian WaveNet和Parallel WaveNet(即ClariNet)(《ClariNet: Parallel wave generation in end-to-end text-to-speech》，《Probability density distillation with generative adversarial networks for high-quality parallel waveform generation》)。该WaveNet由24层空洞残差卷积块组成，具有4个膨胀周期。剩余通道和跳跃通道的数量设置为128个，滤波器大小设置为3。采用RAdam优化器对模型进行1.5M步的训练。学习率设置为0.001，每200K步学习率降低一半。batch size大小设置为8，每个音频剪辑的长度设置为12K时间样本(0.5秒)。

 为了训练基线ClariNet，我们使用上述的自回归WaveNet作为教师模型。ClariNet以高斯IAFs为基础，由六个流程组成。每个流的参数由10层空洞残差卷积块以指数增长的膨胀周期进行参数化。剩余通道和跳跃通道的数量设置为64，滤波器大小设置为3。

平衡KLD和STFT辅助损失的权重系数分别设为0.5和1.0。使用与Parallel WaveGAN相同的优化器设置，对模型进行了400K步的训练。我们还研究了对抗损失ClariNet作为GAN和密度蒸馏的混合方法。模型结构与基线ClariNet相同，但采用KLD、STFT和对抗损失的混合训练，其中平衡它们的权重系数分别设为0.05、1.0和4.0。采用固定的判别器对模型进行200K步的训练，对其余200K步的生成器和判别器进行联合优化。
       
在整个波形生成模型中，对输入辅助特征进行最近邻上采样，然后进行二维卷积，使辅助特征的时间分辨率与语音波形的采样率相匹配(《Probability density distillation with generative adversarial networks for high-quality parallel waveform generation》，《Deconvolution and checkerboard artifacts》)。注意，辅助特征不用于判别器。所有模型都使用了两个NVIDIA Tesla V100 GPU进行训练。实验在NAVER智能机器学习(NSML)平台(《NSML: Meet the mlaas platform with a real-world case study》)上进行。

##### 4.2 评价

为评价知觉质量，采用平均意见评分(MOS)检验。18位以日语为母语的人被要求对合成语音样本做出高质量的判断，使用以下5种可能的回答：1 = Bad; 2 = Poor; 3 = Fair; 4 = Good; and 5 = Excellent。从评价集合中随机抽取20个话语，然后使用不同的模型进行合成。

<div align=center>
    <img src="zh-cn/img/ch7/03/p3.png" /> 
</div>

上图给出了不同生成模型的推理速度和MOS测试结果。结果表明：
1. 有短时傅立叶变换损失的系统比没有短时傅立叶变换损失的系统(即自回归WaveNet)表现更好。注意，大多数听者对自回归WaveNet系统产生的高频噪声不满意。这可以用以下事实来解释：在WaveNet中，只有频段有限(70 - 8000Hz)的梅尔光谱图用于局部调节，而其他系统能够通过STFT损失直接学习全频带频率信息。
2. 所提出的基于短时傅立叶变换损失的多分辨率模型比传统的单一短时傅立叶变换损失模型具有更高的感知质量(分别比较系统3和系统6与系统2和系统5)。这证实了多分辨率STFT损耗有效地捕获了语音信号的时频特性，使其能够获得更好的性能。
3. 提出的对抗性损失在ClariNet上没有很好地工作。然而，当它与TTS框架相结合时，可以发现它的优点，这将在下一节中讨论。
4. 最后，提出的Parallel WaveGAN达到4.06 MOS。虽然与ClariNet相比，Parallel WaveGAN的感知质量相对较差，但它产生语音信号的速度是ClariNet的1.96倍。

此外，该方法的优点在于训练过程简单。我们测量了获得最优模型的总训练时间，如下图所示。由于Parallel WaveGAN不需要任何复杂的密度蒸馏，优化只需要2.8天的训练时间，比自回归WaveNet和ClariNet分别快2.64和4.82倍

<div align=center>
    <img src="zh-cn/img/ch7/03/p4.png" /> 
</div>

##### 4.3 从文本到语音

为了验证所提方法作为TTS框架声码器的有效性，我们将Parallel WaveGAN与基于Transformer的参数估计相结合(《Attention is all you need》，《Neural speech synthesis with Transformer network》，《FastSpeech: Fast, robust and controllable text to speech》)。

为了训练Transformer，我们使用音素序列作为输入，从录音语音中提取梅尔谱图作为输出。该模型由一个6层编码器和一个6层解码器组成，每个编码器和解码器都是基于多头注意力(有8个头)。配置遵循之前的工作(《FastSpeech: Fast, robust and controllable text to speech》)，但模型被修改为接受重音作为音高重音语言(如日语)(《Investigation of enhanced Tacotron text-to-speech synthesis systems with self-attention for pitch accent language》)的外部输入。该模型使用RAdam优化器进行1000个周期的训练，并使用预热学习率调度(《Attention is all you need》)。初始学习率设置为1.0，使用动态批大小(平均64)策略稳定训练。

在合成步骤中，输入的音素和重音序列通过Transformer TTS模型转换为相应的梅尔谱图。通过输入得到的声学参数，声码器模型生成时域语音信号。

为了评估生成的语音样本的质量，我们进行了MOS测试。测试设置与4.2节中描述的相同，但我们在测试中使用了自回归WaveNet和经过多分辨率STFT损失训练的并行生成模型(分别在表2中描述的系统1、3、4和6)。MOS试验结果如表4所示，其中可以总结如下：
1. 有对抗损失的ClariNet比没有对抗损失的ClariNet表现更好，尽管在分析/合成情况下，他们的知觉质量几乎相同(系统3和系统4如表2所示)。这意味着使用对抗损失有利于提高模型对由声学模型引起的预测措辞误差的稳健性。
2. 对抗性训练的优点也有利于提出的Parallel WaveGAN系统。

因此，采用Transformer TTS模型的Parallel WaveNet达到了4.16 MOS，达到了最佳的基于蒸馏的Parallel WaveNet(ClariNet-GAN)。

<div align=center>
    <img src="zh-cn/img/ch7/03/p5.png" /> 
</div>

#### 5.结论

提出了一种基于GAN的无蒸馏、快速、小足迹的并行波形生成方法。通过联合优化波形域对抗损失和多分辨率STFT损失，我们的模型能够学习如何生成真实的波形，而不需要任何复杂的概率密度蒸馏。实验结果表明，该方法在基于Transformer的TTS框架内达到了4.16 MOS，在仅1.44M模型参数的情况下，生成24kHz语音波形的速度是实时的28.68倍。未来的研究包括改进多分辨率短时傅里叶变换的辅助损失，以更好地捕捉语音特征(如引入相位相关损失)，并验证其在各种语料库(包括表达语料库)中的性能。

<!-- ### 4. GAN-TTS discriminators -->
------

### 4. WaveNet: A Generative Model for Raw Audio

!> https://arxiv.org/abs/1609.03499

!> 源码详解参考： <https://zhuanlan.zhihu.com/p/24568596>


<!-- https://zhuanlan.zhihu.com/p/414519043 -->
<!-- https://blog.csdn.net/weixin_42721167/article/details/112593690 -->
<!-- https://zhuanlan.zhihu.com/p/338245185 -->

<!-- https://www.bilibili.com/video/BV1ZK4y1Z73J/?vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->
<!-- https://www.bilibili.com/video/BV13v411w7p7/?vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

<!-- https://zhuanlan.zhihu.com/p/662017509 -->

**评论**：文章对模型的设计非常好，很有参考价值。模型中采用的空洞卷积的方法来极大的增加感受野，对序列数据建模很有用。

#### 摘要

本文提出了一种对于生成原始语音模型的深度神经网络。这个模型是完全的概率自回归模型。有以下几个亮点：

1. 通过这个模型，可以生成更加真实的语音。
2. 可以以相同的保真度捕捉多人的语音并且可以在多人之间切换。
3. 在音乐上的合成，也会生成高保真的语音片段。
4. 作为一个判别模型，在音素识别上有可观的前景。

#### 1.简介

受到最近利用神经自回归生成模型模拟复杂的分布例如（图像和文字）的启发，这篇文章挖掘了原始语音生成的一些技术：使用神经网络架构，把**像素或者单词的联合概率作为条件概率分布的乘积的建模方法**。本文要试图解决的问题是： 这些方法是否可以在宽带原始音频波形的生成中的应用。这些音频波形信号具有非常高的短时分辨率。

基于 PixelCNN, 这篇文章的主要贡献是：
+ 展示了WaveNet可以生成原始语音信号，其自然度由人类裁判进行主观评分，这在语音合成（TTS）领域还未被报道过。
+ 为了处理原始音频生成中所需的**大跨度时间依赖**，我们基于空洞因果卷积（dilated causal convolutions）开发了新的架构，它具有非常大的感受野(receptive filed)。
+ 展示了如果基于说话人身份进行训练，单个模型可以生成不同风格的语音。
+ 同样的架构在小规模语音识别数据集的测试中获得了很好的结果，同时用于音乐等其他形态的音频生成中也有很好的前景。

#### 2.WaveNet

类似于PixelCNN，我们有联合概率密度分布$x=x_1,...,x_T$
 , 其作为条件概率密度的乘积： $$ p(x)=\prod_{t=1}^Tp(x_t|x_1,...,x_{t-1}) $$ 其实我们可以观察到，每一个音频样本都取决于前面的样本，也就说这里是一个时间序列模型。类似于PixelCNN，有以下特点：
+ 条件概率分布是通过多层卷积实现的，网络中没有pooling层。
+ 模型使用softamx的层来输出，使用最大似然函数（MLE）进行优化。
+ 因为对数易于处理，我们通过MLE在验证集上进行超参数优化的时候可以很容易测定模型的overfitting/underfitting。

##### 2.1 空洞因果卷积(Dilated causal convolutions)

<div align=center>
    <img src="zh-cn/img/ch7/04/p1.png" /> 
</div>

使用因果卷积(causal convolutions)来保证 $p(x_{t+1}|x_1,...,x_t)$
 不包含 $x_{t+1},...,x_T$
 的信息。对于图像处理任务中，类似的是采用masked convolution。对于1-D数据，秩序将输出偏移几步就行。动态图参考： [WaveNet: A Generative Model for Raw Audio | DeepMind](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)

**Causal convolutions和RNN的区别：**

1. 在训练阶段，由于标定真实数据$x$的所有时间步骤都是已知的，因此所有时间步骤的条件概率预测可以**并行进行**。在推断阶段，预测结果是串行的：每一个预测出的样本都被传回网络用于预测下一个样本
2. 因为没有循环连接，通常训练起来比RNN更快。

但是存在一个问题就是，因果卷积存在的一个问题就是需要很多的卷积层或者很大的filter来实现增加感受野的功能。比如说上面那张图片的感受野只有`5 (= # layers + filter length - 1)` ，关于感受野的计算， 参考[感受野(Receptive Field)的理解与计算](https://zhuanlan.zhihu.com/p/113487374), 所以我们在这里使用了空洞（扩张）卷积来成倍增加感受野，并且没有很大的计算消耗。

<div align=center>
    <img src="zh-cn/img/ch7/04/p2.png" /> 
</div>

在上图中，感受野分别增加了`1，2，4，8`倍。空洞卷积（dilated convolution）可以使模型在层数不大的情况下有非常大的感受野。在这篇论文中，扩大系数都翻倍至上限，然后重复循环， 例如：`1, 2, 4, . . . , 512, 1, 2, 4, . . . , 512, 1, 2, 4, . . . , 512`. 为什么要这样配置呢？有两个intution：

+ 指数级增长(exponentially increasing)的扩大可以引起感受野指数级的增长。例如每一组`1;2;4;:::;512`这样的卷积模块都拥有1024大小的感受野， 可视为与1x1024卷积对等的更高效的（非线性）判别式卷积操作 。
+ 将多组这样的模块堆叠起来可以进一步增长模型的容量和感受野。

##### 2.2 SoftMax的分布

因为原始音频是按照16-bit（one per timestep）的整数值序列储存的，每个timestep，softmax层需要输出65536个概率值。为了便于运算，我们应用了**Law Companding Transformation**进行转换，将输出概率数目降低为256个。公式如下：

$$f(x_t)=sign(x_t)\frac{ln(1+\mu|x_t|)}{ln(1+\mu)},其中-1 < x_t < 1, \mu=255$$

##### 2.3 门控单元

本文也采用了类似于PixelCNN的门控单元：

<div align=center>
    <img src="zh-cn/img/ch7/04/p3.png" /> 
</div>

$$z=tanh(W_{f,k} \ast x) \odot \sigma(W_{g,k} \ast x)$$

+ $\ast$是卷积操作
+ $\odot$是矩阵element-wise的乘积
+ $\sigma$是sigmoid函数
+ $k$是层数索引
+ $f,g$是滤波器和Gate
+ $W$是卷积核

> 为什么不使用ReLU？

In our initial experiments, we observed that this non-linearity worked significantly better than the rectified linear activation function (Nair & Hinton, 2010)(ReLU) for modeling audio signals. 因为在音频信号分析中，sigmoid就是好一些。

<div align=center>
    <img src="zh-cn/img/ch7/04/p4.png" /> 
</div>

上面是ReLU的激活，丢失了信息。

##### 2.3 Residual and skip connections

<div align=center>
    <img src="zh-cn/img/ch7/04/p5.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch7/04/p6.png" /> 
</div>

文章中使用了residual 和skip connection技术来使模型更快收敛，并且使梯度能传到到更深层模型。

##### 2.4 Conditional WaveNet

接受额外输出的$h$，我们可以如下建模：

$$p(x|h)=\prod^{T}_ {t=1}p(x_ t|x_1,..,x_{t-1},h)$$

通过条件分布输入其他变量建模，我们可以使得WaveNet生成具有目标特点的音频。额外信息可以是：在多人语音我们输入一个人声音作为额外输入，在TTS(text-to-speech)的任务中，我们可以额外输入text的信息。文章中有两种建模的方法：全局方法(global conditioning) 以及局部方法 (local conditioning).

1. Global conditioning

全局建模方法接受单额外输入$h$，并且$h$在所有时间节点影响输出, 
$$z=tanh(W_{f,k}\ast x+V_{f,k}^{T}h) \odot \sigma(W_{g,k}\ast x + V_{g,k}^{T}h)$$

$V_{\ast,k}^T$在所有时间节点上传播,$V_{\ast,k}$是可学习的参数矩阵。

2. Local conditioning

我们有第二种时间序列$h_t$,可以通过对原始数据的降采样率获得（比如TTS模型中的线性特征）。我们首先通过transposed convolutional network（learned upsampling）$y=f(h)$将时间序列转换成和语音序列一样分辨率的新的时间序列。然后将其用于激活单元：

$$z=tanh(W_{f,k}\ast x+V_{f,k}^{T}y) \odot \sigma(W_{g,k}\ast x + V_{g,k}^{T}y)$$

如果采用了transposed convolutional network (learned upsampling),我们也可以直接使用 $V_{f,k}\ast h$
,但是没有`1x1`卷积好。

##### 2.5 Context Stacking

我们提到的增加感受野(receptive filed)的方法：

+ 增加空洞卷积的数目
+ 使用更多层数
+ 更大的卷积核滤波器(filter)
+ 更大的空洞因子

另外一种可以增加感受的野的补充方法是：使用一个独立的更小的上下文堆栈来处理语音信号的长跨度信息，并局部调试一个更大的WaveNet只用来处理语音信号的更短的局部信息（在结尾处截断）。可以使用多个变长的具有不同数量隐藏单元的上下文堆栈，拥有越大感受野的堆栈其每层含有的隐藏单元越少。上下文堆栈还可以使用池化层来降低频率，这使得计算成本被控制在合理范围，也与用更长的跨度对时间相关性建模会使体量更小的直觉相吻合。

#### 3.实验

主要进行了三个Tasks的实验：

+ 多说话人的语音合成（没有基于文本训练）
+ 文本合成语音
+ 语音音频建模

关于实验部分：请见[deepmind的post](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)

**Multi-Speaker Speech Generation**

使用的多人语音材料是来自于: CSTR voice cloning toolkit (VCTK)，基于说话人的条件进行了建模，数据集一共包括44个小时的109个位不同的说话人。

因为不是基本文本信息建模，所以这个也会产生一些现实中并不存在但是听起来还是不错的一些人类语言。这就像语言生成图片一样，乍一看还不错，细瞅瞅就不行了。部分由于感受野的大小，这些语音在长跨度上缺少连贯性，基本上每次只能记住2-3个场景信息。

通过one-hot编码，单个的WaveNet可以建模任意一个说话者语音。我们发现，与在单人的数据集训练上相比，增加训练集说话人的数量可以在验证集上获得更好的效果。这表示：WaveNet的说话表达是多人共享的。同时WaveNet也会捕捉其他语音信息，例如声音质量，说话人的呼吸以及嘴部动作等。

**Text-to-Speech**

数据集来自于Google’s North American English and Mandarin Chinese TTS systems。

在TTS 的任务中，WaveNet只考虑了(were locally conditioned on)语言学特征。同时也考虑了语言学特征和对数基频(logarithmic fundamental frequency, $logF_0$)作为额外的feature考虑到模型生成中去。这个模型的感受野是250毫秒，还构建了HMM单元选择拼接(Gonzalvo et al., 2016)语音合成器作为基于例句的基线，以及LSTM-RNN统计参数(Zen et al., 2016)语音合成器作为基于模型的基线。整个评价采用了主观评分模式。五分制度（5为最高分）

<div align=center>
    <img src="zh-cn/img/ch7/04/p7.png" /> 
</div>

**Music**

数据集来自于：

+ the MagnaTagATune dataset (Law&Von Ahn, 2009), which consists of about 200 hours of music audio. Each 29-second clip is annotated with tags from a set of 188, which describe the genre, instrumentation, tempo, volume and mood of the music.
+ the YouTube piano dataset, which consists of about 60 hours of solo piano music obtained from YouTube videos. Because it is constrained to a single instrument, it is considerably easier to model.

我们发现，对于让一段音乐获得音乐性，增加感受野的长度是非常重要的。即使感受野增加到几秒钟，这个模型也没表现出长的连续性。我们发现，即使是非条件建模，这个模型还是不错的，特别在合声部分。

但是，我们希望进行条件建模，例如说题材和乐器等。类似于条件语音生成一般，我们插入了依赖于与每个训练片段相关联的标签的二进制向量表示的bias。当我们采样的时候，这就可以使我们能够去控制我们的输出样本，并且效果还不错。

**Speech Recognition**

WaveNet也可以用在语音识别里面。总结了下前人的工作，近期的研究转向到原始语音数据建模。例如：(Palaz et al., 2013; T¨uske et al., 2014; Hoshen et al., 2015; Sainath et al., 2015). LSTM-RNNs 是这些工作的核心部分，LSTM使得我们可以建立更大的感受野但是同时我们可以使用更小的代价函数。

#### 4.结论

本文提出了WaveNet，利用自回归且结合了空洞因果卷积增加感受野，这对于长时序问题建模的依赖非常重要。WaveNet可以通过两种方式建模，global (e.g. 语音识别)或者local way (e.g 语言特征的获取)。

------

### 5. WaveRNN

!> https://arxiv.org/abs/1802.08435v1

!> https://github.com/fatchord/WaveRNN

<!-- https://zhuanlan.zhihu.com/p/464033874 -->




### 6. WaveGrad

!> https://arxiv.org/abs/2009.00713


### 7. HiFiGAN V1/V2

!> v1: https://arxiv.org/abs/2006.05694

!> v2: https://ieeexplore.ieee.org/document/9632770

!> v2: https://daps.cs.princeton.edu/projects/Su2021HiFi2/



### 8. UnivNet

!> https://arxiv.org/abs/2106.07889



### 9. WaveGlow

!> https://arxiv.org/pdf/1811.00002.pdf
