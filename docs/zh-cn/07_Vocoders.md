
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

### 5. WaveRNN(\*): Efficient Neural Audio Synthesis

!> https://arxiv.org/abs/1802.08435v1

!> https://github.com/fatchord/WaveRNN

<!-- https://zhuanlan.zhihu.com/p/464033874 -->

!> 不是很懂这篇paper!只知道DeepMind使用RNN方式代替了WaveNet,里边大部分都是工程上优化的Trick，其核心思想是怎么让WaveRNN在工程上算的更快！

#### 摘要

序列模型在估计数据分布和生成高质量样本方面在音频、视觉和文本领域实现了最先进的结果。然而，此类模型的有效采样仍然是一个难以捉摸的问题。以文本到语音合成为重点，我们描述了一组通用技术，用于在保持高输出质量的同时减少采样时间。

我们首先描述了一个单层循环神经网络 WaveRNN，它具有与最先进的 WaveNet 模型的质量相匹配的双 softmax 层。

网络的紧凑形式使得在 GPU 上生成 24kHz 16 位音频的速度比实时速度快 4 倍。其次，我们应用了**权重修剪技术**来减少 WaveRNN 中的权重数量。

我们发现，**对于恒定数量的参数，大型稀疏网络比小型密集网络表现更好**，并且这种关系适用于超过 96% 的稀疏度水平。稀疏 WaveRNN 中的少量权重使得在移动 CPU 上实时采样高保真音频成为可能。

最后，我们提出了一种基于 **subscaling** 的新一代方案，将一个长序列折叠成一批较短的序列，并允许一次生成多个样本。Subscale WaveRNN 每步产生 16 个样本而不会损失质量，并提供一种正交方法来提高采样效率

#### 1.简介

顺序生成模型在包括自然语言 (Wu et al., 2016; Vaswani et al., 2017)、自然图像 (van den Oord et al., 2016b; Reed 等人，2017）和视频（Kalchbrenner 等人，2017）以及语音和音乐（van den Oord 等人，2016a；Mehri 等人，2016；Simon&Oore，2017；Engel 等人，2017） 。这些模型通过将分布分解为每个样本的条件概率的乘积来学习数据的联合概率。 这种结构使模型分配了显著的容量来估计每个条件因素，使它们在训练期间具有鲁棒性并且易于评估。结构中编码的排序也使得采样过程严格串行：只有在它所依赖的样本按照排序生成后，才能生成样本。

采样过程的串行方面会使使用这些模型生成语音和视频等高维数据变得缓慢且不切实际。我们的目标是在不影响质量的情况下提高序列模型的采样效率。**采样过程所花费的时间 $T(u)$ 是目标 $u$ 中的样本数量（例如，语音中的音频样本数量或图像中的像素数量）与生成每个样本所需的时间的乘积样本。**后者可以分解为模型 $N$ 层（操作）中每一层的计算时间 $c(op_i)$和开销 $d(op_i)$:
$$T(u)=|u|\sum^{N}_ {i=1}(c(op_i)+d(op_i)) (公式1)$$
在以下任何一种情况下，$T(u)$ 的值都会变得异常大：

+ 如果$|u|$与每秒由 24,000 个 16 位样本组成的高保真音频一样大；
+ 如果 $N$ 很大，因为使用了 WaveNet 等非常深的架构（van den Oord et al., 2016a）；
+ 如果 $c(op_i)$由于例如特别是宽层或大量参数；或者如果由于启动每个单独操作的成本而导致开销 $d(op_i)$很高。

以文本到语音合成为重点，我们提出了一组方法来使采样数量级更快。我们减少每个因素 $N$、 
$d(op_i)$、 $c(op_i)$和 $|u|$ 的贡献对生成的输出质量的损失最小。

我们在单人北美英语文本到语音数据集上对所有模型进行基准测试，其中输入由预测的语言特征向量组成，输出是原始的 24 kHz、16 位波形（第 5 节）。我们报告了模型在保留数据上达到的负对数似然 (NLL)、人类听众评定的一对模型之间的 A/B 比较测试结果以及样本的平均意见分数 (MOS)模型。

我们首先设计一个序列模型，每个样本需要少量 $N$ 次操作。我们利用循环神经网络 (RNN) 的核心特性，即应用于先前状态的单个循环层可以提供上下文的高度非线性变换

WaveRNN 模型是具有双 softmax 层的单层 RNN，旨在有效地预测 16 位原始音频样本。我们看到具有 896 个单元的 WaveRNN 实现了与最大 WaveNet 模型相当的 NLL 分数，根据 A/B 比较测试（表 1），音频保真度没有显着差异，MOS 也同样高。 WaveRNN 通过对每个 16 位样本只需要 $N = 5$ 个矩阵向量乘积来实现这一性能；为简单起见，我们从计数 $N$ 中排除了非线性和其他次要操作。

这与具有 30 个两层残差块的 WaveNet 形成对比，每层都需要一系列 `N = 30 * 2 = 60` 个矩阵向量乘积。即使 $N$ 很低，开销 $d(op_i)$ 仍然是 WaveRNN 采样的常规实现中的一个重要瓶颈。我们通过为采样过程实现自定义 GPU 操作 (Diamos et al., 2016) 来回避开销。这允许 WaveRNN 在 Nvidia P100 GPU 上每秒生成 96,000 个 16 位样本，相当于 4 倍实时高保真 24kHz 16 位音频。 作为比较，我们用于 WaveNet 模型的最佳 GPU 内核在同一平台上以大约 0.3 倍的实时速度运行。吞吐量随着批次 4 的增加而增加，其中内核达到每秒 39,0000 个样本（总吞吐量为 156,000 个样本/秒。）

减少网络中的参数数量会减少采样所需的计算量 $c(op_i)$
。考虑到这一点，我们的目标是最大化我们可以从给定数量的参数中获得的性能。(Gordon et al., 2017) 还考虑了在给定计算预算下最大化性能的问题，并使用基于神经元修剪的方法来解决这个问题。我们使用 (Narang et al., 2017a; Zhu & Gupta, 2017) 的权重修剪技术对 WaveRNN 中的权重进行稀疏化。对于固定的参数计数，我们发现大型稀疏 WaveRNN 的性能明显优于小型密集 WaveRNN，并且这种关系的稀疏度高达 96% 以上（图 2）。

Sparse WaveRNN 的高质量输出、少量参数和对内存带宽的低要求相结合，使得该模型非常适合在低功耗移动平台上高效实现（例如在手机中发现的那些）。 我们在移动 CPU 上实现并基准化了 WaveRNN 中使用的稀疏矩阵向量乘积和非线性（表 2）。

<div align=center>
    <img src="zh-cn/img/ch7/05/p1.png" /> 
</div>

尽管移动 CPU 上的计算量和内存带宽分别比 GPU 小三个和两个数量级，但我们在现成的移动 CPU 上进行的基准测试表明，资源足以在具有高质量稀疏 WaveRNN 的设备音频合成。 据我们所知，这是第一个能够在广泛的计算平台上进行实时音频合成的序列神经模型，包括现成的移动 CPU。

<div align=center>
    <img src="zh-cn/img/ch7/05/p2.png" /> 
</div>

最后，我们处理组件 $|u|$ 的贡献。 在方程式 1 中。最近的多种方法的目标是使序列模型的采样更加并行（Reed 等人，2017；Gu 等人，2017；van den Oord 等人，2017）。

然而，这些模型要么在生成的样本之间做出局部独立性假设，从而破坏了序列模型的主干，要么它们需要训练多个特定领域的网络，这些网络具有限制模型整体可用性的专门损失。

我们提出了一个基于**subscaling**的生成过程。一个尺度为$L$的张量被折叠成$B$个尺度为 $L/B$ 的子张量。
$B$个子张量按顺序生成，每个子张量都以先前的子张量为条件。 Subscaling 让我们可以在一个批次中一次生成多个样本。由于在以前的子张量上生成每个子张量的条件在实践中只需要相对较小的未来范围，因此下一个子张量的生成可能在前一个子张量的生成开始后不久开始。原则上可以尽管在实践中没有必要恢复遥远的未来和过去的依赖关系，超越地平线；

那么，批量采样的精确成本就是当前批次中样本之间的$B$个远距离依赖关系。正如 A/B 比较测试（表 1）所证明的，Subscale WaveRNN 能够在不损失音频保真度的情况下每步产生 `B = 16` 个样本。

<div align=center>
    <img src="zh-cn/img/ch7/05/p3.png" /> 
</div>

Subscale WaveRNN 的批量采样开辟了许多提高采样效率的正交方法。即使是我们对模型的常规 Tensorflow 实现也可以在 Nvidia V100 GPU 上实现实时采样速度。Subscale WaveRNN 的 Fused 变体还通过对 WaveRNN-896 的 GPU 内核稍作修改，在 Nvidia P100 GPU 上提供了 10 倍的实时采样速度。

#### 2.Wave Recurrent Neural Networks (WaveRNN)

卷积序列模型 (Kalchbrenner et al., 2016) 在语音合成方面取得了出色的性能 (Wang et al., 2017)，但它们的架构往往很深而且很窄，需要为每个样本执行一长串层。我们寻求一种架构，该架构能够提供同样具有表现力和非线性的上下文转换，但在每一步都需要少量操作。

通过拥有一个隐藏状态来维护已经压缩的上下文表示，RNN 特别适合此目的，因为它能够在单个转换中将上下文与输入结合起来。 WaveRNN 中的总体计算如下（为简洁起见，我们省略了偏差）：

$$x_t = [c_{t-1},f_{t-1},c_t]$$
$$u_t=\sigma(R_uh_{t-1}+I^{\ast}_ {u}x_t)$$
$$r_t=\sigma(R_rh_{t-1}+I^{\ast}_ {r}x_t)$$
$$e_t=\tau(r_t)\circ(R_eh_{t-1}+I^{\ast}_ {e}x_t)$$
$$h_t=u_t\circ h_{t-1}+(1-u_t)\circ e_t$$
$$y_c,y_f=split(h_t)$$
$$P(c_t)=softmax(O_2relu(O_1y_c))$$
$$P(f_t)=softmax(O_4relu(O_3y_f))$$

其中，$\ast$表示掩码矩阵，其中最后一个粗略输入$c_t$仅连接到状态$u_t$,$r_t$,$e_t$和$h_t$的精细部分，因此仅影响精细输出$y_f$。错略和精细部分$c_t$和$f_t$在[0,255]中被编码为标量，并缩放到区间[-1,1]。

由矩阵$R_u,R_r,R_e$形成的矩阵$R$ 被计算为单个矩阵向量积，以产生对所有三个门$u_t,r_t,e_t$的贡献（GRU 单元的变体，如 (Chung et al., 2014 ; Engel, 2016).) $\sigma,\tau$是标准的 sigmoid 和 tanh 非线性。

一种可能的架构变体是让$h_t$仅依赖于$x_{t-1}$并使用一个全连接层，然后使用求和或连接来将$h_t$置于$c_t$上；我们发现这个版本需要多出 20% 的参数，并且性能也差了 1-2 厘纳。

我们将 RNN 的状态分为两部分，分别预测 16 位音频样本的 8 个粗略（或更重要）位 $c_t$ 和 8 个精细（或最低有效）位 $f_t$（图 1）。

每个部分在相应的 8 位上馈入一个 softmax 层，并且 8 个精细位的预测以 8 个粗略位为条件。由此产生的 Dual Softmax 层允许使用两个小输出空间（每个 $2^8$ 个值）而不是单个大输出空间（具有 $2^16$ 个值）来有效预测 16 位样本。

图 1 直观地显示了这一点。我们注意到可以在所有 $2^16$ 个值上使用一个 softmax 进行训练，但是除了需要显着更多的参数、内存和计算之外，它的性能始终差 1-2 厘纳。

<div align=center>
    <img src="zh-cn/img/ch7/05/p4.png" /> 
</div>

##### 2.1 WaveRNN Sampling on GPU

上述架构将每个步骤所需的操作数 $N$ 从具有 16 位离散逻辑混合 (DLM) 输出（Salimans 等人，2017）的 WaveNet 的 $N = 60$ 减少到提议的 WaveRNN 的 $N = 5$双softmax。尽管操作数 $N$ 减少了，但 WaveRNN 采样的常规实现并不能直接产生实时或更快的合成。在 GPU 上，主要障碍不是采样所需的原始 FLOP；相反，困难是双重的：内存带宽的限制和启动 $N$ 个操作中的每一个操作所需的时间。关于前者，一个状态为 896 个单元的 WaveRNN（WaveRNN-896）有大约 3M 个参数。对于 24,000 个样本中的每一个样本，按顺序分别调用每个 WaveRNN 操作的常规采样实现在每个步骤中将所有 WaveRNN 参数从内存加载到 GPU 寄存器中，总计约 `3e6 × 24e3 × 4 = 288 GB` 所需的内存带宽。这已经是 Nvidia P100 GPU 中可用内存带宽的三分之一以上，它本身就为常规采样实现提供了 3 倍实时的上限。

在 GPU 上单独启动每个操作的开销更大。虽然在 GPU 上启动操作具有 5 微秒的恒定开销，但每个步骤需要 `N = 5` 次此类操作，这意味着仅启动开销就导致每秒 40,000 个样本的上限。对于每个样本需要（至少）`N = 60` 次操作的 WaveNet 架构，启动开销导致每秒 3,300 个样本的上限。这没有考虑实际计算操作所花费的时间。

在实践中，定期实施抽样，例如对于 WaveRNN-896 和 WaveNet，Tensorflow 每秒分别产生大约 1600 和 170 个样本。我们通过将采样过程直接实现为单个持久 GPU 操作来减少这两个因素。避免了内存带宽瓶颈，因为参数仅在采样开始时加载到 GPU 寄存器中一次，并在整个过程中持续存在于寄存器中。这是可能的，因为 P100 GPU 有 367 万个全精度寄存器，足以存储超过 700 万个半精度参数，即 WaveRNN 896 所需数量的两倍多。也避免了操作启动瓶颈，因为一个话语的整个采样过程是作为单个 GPU 操作执行的。

896 的状态大小被专门选择以适合具有 56 个多处理器的 P100 GPU。必须分配给每个多处理器以访问 GPU 的完整寄存器文件的最小扭曲数为 8。如果我们将每个扭曲分配给状态计算，那么状态大小必须是 `56 * 8 = 448 `的倍数，并且适合可用寄存器空间的最大倍数是 896。生成的用于 WaveRNN 采样的 GPU 内核比常规采样实现效率高两个数量级，WaveRNN-896 达到 96,000 个样本/秒。

WaveNet 的相应操作达到 8,000 个样本/秒。新的开销 $d(op)$ 现在由 GPU 中的数千个内核的同步给出（Xiao & c. Feng, 2010），每次同步只需 500 纳秒，而不是每次启动操作所需的 5 微秒。

#### 3.Sparse WaveRNN

WaveRNN 架构显着减少了所需操作 $N$ 的数量，并将采样作为单个 GPU 操作实现消除了大部分原始计算 $c(op_i)$ 和开销 $d(op_i)$ 瓶颈。接下来，我们提出了一种直接减少每个操作所需的计算量 $c(op_i)$ 的技术。减少隐藏单元的数量会减少计算量，但这会带来质量的显着损失（表 3）。

相反，我们通过稀疏化权重矩阵来减少网络中非零权重的数量，同时保持较大的状态大小和各自的表示能力。 这减少了 $c(op_i)$
，因为非零权重的数量与$c(op_i)$成正比（表 4）。

<div align=center>
    <img src="zh-cn/img/ch7/05/p5.png" /> 
</div>

##### 3.1 Weight Sparsifification Method

我们使用基于权重大小的剪枝方案，随着训练的进行会增加稀疏性（Narang 等人，2017a；Zhu & Gupta，2017）。我们维护一个二进制掩码，指定权重矩阵的稀疏模式。在训练开始时，权重矩阵是密集的。 每 500 步，每个稀疏层内的权重按其大小排序，并且通过将$k$ 个最小大小的权重归零来更新掩码。数字$k$ 被计算为权重总数的分数 $z$，作为训练步骤 $t$ 的函数，它从 0 逐渐增加到目标稀疏度 $Z$：
$$z = Z(1-(10\frac{t-t_0}{S})^3)$$

其中$t_0$是权重修剪开始的步骤，$S$ 是修剪步骤的总数。我们使用 $t_0 = 1000, S = 200k$ 并为所有模型训练总共 500k 步。这样的方案实用，易于集成到现有模型中，并且不增加训练时间。 我们分别稀疏化 GRU 单元内的三个门矩阵。

##### 3.2 Structured Sparsity

我们需要以允许的方式对稀疏掩码进行编码高效的计算。标准压缩稀疏行格式使用与存储参数相同的存储量来编码稀疏掩码。
与 Viterbi 剪枝 (Lee et al., 2018) 等面向硬件的方法不同，我们探索结构化稀疏性作为减少内存开销的一种手段。
我们考虑的稀疏掩模中的结构是不重叠的权重块的形式，这些权重块根据块内权重的平均大小被修剪或保留在一起。
我们发现 `m = 16` 权重的块在非结构化稀疏性方面损失很小的性能，同时将存储稀疏性模式所需的内存量减少到非结构化掩码所需的 `1m`。

除了我们发现工作良好的矩形 `4 × 4` 块（Gray 等人，2017；Narang 等人，2017b）外，我们还采用了 `m × 1 `形状的块，这会导致更低的内存带宽开销。在 `m × 1 `块的情况下，只需从隐藏状态中检索单个激活值即可执行点积。这与方形块形成对比，方形块中每个块需要从隐藏状态中检索 4 个激活值。我们报告了 `16 × 1` 和 `4 × 4` 块的结果。基准测试证实了 `16 × 1` 块的速度更快（表 4）。

##### 3.3 Sparse WaveRNN Sampling on Mobile CPU

我们利用稀疏 WaveRNN 所需的低计算和内存带宽来实现在移动 CPU 上进行采样所需的矩阵向量操作。为了最大化内存利用率，权重以 16 位浮点数存储，并在用于计算之前转换为 32 位浮点数。激活和计算保存在 32 位浮点中。小块提供的低内存开销允许稀疏矩阵向量产品与具有相同参数计数的密集矩阵向量产品的性能相匹配。因此，每秒顺序矩阵向量乘积的数量几乎完全由网络中的参数数量决定。

#### 4.Subscale WaveRNN

我们已经描述了在高保真音频生成中减少采样时间的两种方法：减少 $N$ 和 $d(op)$ 的 WaveRNN 和减少 $N$ 和 $c(op)$ 的 Sparse WaveRNN。最后，我们减少因子 $|u|$ 的贡献。 在等式 1 中。这个因素取决于话语 $u$ 的大小，直接减小$u$本身的大小（例如从每个样本 16 位到 8 位）会对音频质量产生负面影响。 相反，我们提出了一种每一步生成一批 $B$ 个样本的方法，而不仅仅是一个：

$$T(u)=\frac{|u|}{B}\sum^{N}_ {i=1}(c(op^B_i)+d(op^B_i)) (公式3)$$

在许多情况下，一批 $B$ 个示例的计算时间 $c(op^B_i)$
 在单个示例 $c(op_i)$的计算时间中呈亚线性增长，因为权重被重用并且可用的备用计算能力。批处理样本的能力还可以跨多个处理器生成，并减少与处理器数量成线性关系的总采样时间。以前在顺序模型中每一步生成多个样本的工作需要打破局部依赖关系（Reed 等人，2017 年）：两个彼此强烈依赖的相邻样本是独立生成的，可能以其他样本为条件。我们引入了一种通用方法，该方法允许我们用少量恒定数量的遥远过去和未来依赖关系来换取每一步生成 $B$ 批样本的能力。

##### 4.1 Subscale Dependency Scheme

首先从张量 $u$ 中提取一组 $B$ 个子张量，其频率或尺度小于 $B$ 倍。每个子张量对应于$u$ 的一个子尺度切片（参见图 3）。
如果 $u$ 是 24kHz 的音频话语并且 $B$ 是 16，那么每个子张量对应于 `24/16=1.5kHz` 的话语。这与从$u$ 中提取的不同子张量具有递增尺度的多尺度方案形成对比。Subscaling 对 $u$ 中的变量的依赖关系产生以下排序，这相当于关节的标准分解：

<div align=center>
    <img src="zh-cn/img/ch7/05/p6.png" width=40%/> 
</div>

给定 $(i, s)$ 的样本 $u_{Bi+s}$
取决于 $z < s$ 和 $k ≥ 0$ 的所有样本 $u_{Bk+z}$。$u$ 的生成过程如下：首先生成第一个子张量，然后以第一个子张量为条件生成第二个子张量，然后以前两个子张量为条件生成第三个子张量，以此类推。生成给定子张量的 Subscale WaveRNN 以先前子张量的未来上下文为条件，使用带有 relus 的掩码扩张 CNN 和应用于过去连接而不是未来连接的掩码。与多尺度方案一样，子尺度方案同样适用于多维张量。

##### 4.2 Batched Sampling

与多尺度方案相比，子尺度化可以在单个步骤中生成 $B$ 个样本。在等式 4 中，对于某些未来视野 $F$ 的 $k > i + F $值，$u_{Bi+s}$ 对未来样本 $u_{Bk+z}$ 的依赖性$（z < s）$变得非常弱（图 3）。Subscale WaveRNN 中的调节网络本身只能看到来自先前子张量的有限且通常少量的未来样本。子张量的采样可以在前一个子张量的前 $F$ 个样本生成后立即开始。因为 Subscale WaveRNN 在所有子张量之间共享，所以可以批量输入，并且在 `B * F `步之后，Subscale WaveRNN 的总批次为 $B$。由于 $F$ 的值（通常为 64 或 128）与 $u$ 的规模和长度相比相对较小，即使对于相对较大的 $B$ 值（例如 16），`B * F` 步的总滞后对于总采样延迟仍然可以忽略不计。尽管需要对每批样本执行调节网络，但计算调节网络不会影响 Subscale WaveRNN 的因子 $N$，因为该网络可以针对选定数量的 $L$ 个未来样本并行执行。这将总采样延迟增加了 `B * L` 步，即使对于 `L = 100` 的值，它仍然可以忽略不计。由于批量采样，即使我们在 Tensorflow 中的常规实现对于具有 1024 个隐藏状态单元的 Subscale WaveRNN 16× 也几乎达到了实时速度（24,000 个样本/秒）。

##### 4.3 Recovering Future and Past Dependencies

删除 $k > i + F$ 的遥远未来依赖关系原则上也允许我们恢复几乎相等数量的遥远过去依赖关系。
继当前子张量 $s$ 之后的子张量 $z$ 比 $s$ 落后 $(z − s)(F + 1)$ 步，但留下了遥远过去样本的痕迹。 在训练和采样期间，可以访问这些遥远的过去样本
条件当前通道 $s$ 的生成。
类似地，来自 $s$ 之前的子张量的 $i + F$ 之外的恒定数量的未来遥远样本也可用于附加条件。
使用子缩放和批量采样的确切依赖方案包括这些远距离依赖； 然而，在实践中，选择较大的值 $F$ 似乎比嵌入远距离依赖项更简单。

##### 4.4 Fused Subscale WaveRNN

我们使用 Subscale WaveRNN 背后的方案在 WaveRNN 本身中每步直接生成超过 16 位。
我们采用 Subscale WaveRNN 2× 模型，而不是批量处理 2 个子张量，我们将 WaveRNN 的隐藏状态分成两部分。 然后我们使用 8 个每个 4 位的 softmax 和一个只有 2 的 $F$ 值。 来自子张量的样本直接作为输入提供给 WaveRNN，而不使用调节网络。
生成的 Fused Subscale WaveRNN 2x 仅在输出质量上实现了小幅下降（表 3），但很好地映射到 WaveRNN GPU 自定义操作上。
与以 4 倍实时运行的 WaveRNN 相比，该模型每步生成 32 位并且需要更少的同步，从而实现 10 倍实时的采样速度。

我们注意到，与 Subscale WaveRNN 相比，因为融合需要拆分隐藏状态，音频质量在 Fused Subscale WaveRNN 中超过 2 倍时会迅速下降。

-----

### 6. WaveGrad: Estimating Gradients for Waveform Generation

!> wavegrad: https://arxiv.org/abs/2009.00713

!> wavegrad2: https://arxiv.org/abs/2106.09660

!> Diffusion Model在Vocoder中的应用

!> WaveGrad 源码 https://github.com/lmnt-com/wavegrad

<!-- !> FastDiff, NaturalSpeech2都是近期基于Diffusion的Vocoder! -->

<!-- https://zhuanlan.zhihu.com/p/417306113 -->
<!-- https://liu-feng-deeplearning.github.io/2022/08/22/ddpm%E4%B8%8EwaveGrad%E8%A7%A3%E6%9E%90/ -->

<!-- https://aaaaaalan.github.io/2021/08/10/Wavegrad2/ -->

#### 1.DDPM方法简介

!> lilan大神关于 DDPM 更系统的讲解 https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice

!> 苏神关于 DDPM 的通俗讲解:盖楼与拆楼 https://spaces.ac.cn/archives/9119

DDPM(Denoising Diffusion Probabilistic Model),是最近生成领域非常火的一类模型。 抛弃严谨的数学推导， 我用自己的理解大概讲一下：传统方案下，需要从无到有/从高斯分布的噪声生成对应的图像， 这个过程非常困难，需要模型从中学到很多东西。扩散模型希望把整个过程拆成若干步， 使用模型每次重建一点点，逐次生成最终的模型。为了能够做到这个事情，需要定义一大堆的中间状态。 考虑生成问题的逆问题，从一张图片生成一个高斯分布，非常简单。每次往图片中随机加一点噪声， 重复足够多的次数，就得到了最终的高斯噪声。记录这个中间过程，想办法逆向完成这个加噪的过程， 即可完成扩散模型。

<div align=center>
    <img src="zh-cn/img/ch7/06/p1.png" width=70%/> 
</div>

#### 2.Vocoder

声码器是语音领域里一个经典的生成问题: `对于给定一个mel谱 x，怎样生成对应的信号?`。 `y -> x` 过程非常简单，但逆问题想做到高精度/效率兼顾却比较难。另外，声码器和图像生成有一个显著的区别。图像生成中，condition 一般是一个类别标签。 对于同一类别标签，显然有各种风格迥异的图片生成。但对于声码器，mel 是稠密的特征，一旦给定 ，音频信号大体上确定了。和音频信号信息相比，谱图中只有相位信息以及 fft 高频截断信息被丢掉了。 而高频信息其实人耳听上去的主观感受并不明显。这表明在 cond 条件下，音频生成的丰富度，远低于图像生成问题。 声码器更近似与一个 1 vs 1 的问题，而不是 1 vs N。 一些生成问题中常见的 badcase 现象(例如 model-collasp/over-smooth) 在这里表现的并不明显。

因此类似 hifigan 等 gan-based 方法在效果方面表现并不差。大家更关注的还是效果/性能之间的平衡。

**WaveGrad: ddpm for vocoder：**

<div align=center>
    <img src="zh-cn/img/ch7/06/p2.png" /> 
</div>

WaveGrad 应该是一个比较早使用 DDPM 的声码器框架，作者开源了其源码，结合论文和代码来做一些解析。首先来看下算法的基本架构，

<div align=center>
    <img src="zh-cn/img/ch7/06/p3.png" /> 
</div>

基本上是一个比较经典的 DDPM 流程，包括 train/sample 两部分

+ train

在训练部分，作者设计了一个神经网络模型, nnet_model, 每次从不同scale中， 带噪语音中预测噪声（进而可以得到干净的语音），使得预测噪声和原始生成随机噪声尽可能接近。

```
noise ~ N(0, 1)
noise_sig = nosie * noise_scale + clean_sig
pred_noise = nnet_model(noise_sig, mel, noise_scale)
Loss = ce(pred_noise, noise)
```

训练过程中，注意对 batch 中每个样本，都要分别产生对应的高斯分布。 以及产生随机的 noise-scale。

+ model 设计

模型本身的设计上，没太多好讲的。包括了 Ublock/Film/Dblock 等模块。不过在 Film的时候， 里面加入了 Position encoding，我记得原版是没有的？这里可能要额外注意一下。

<div align=center>
    <img src="zh-cn/img/ch7/06/p4.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch7/06/p5.png" /> 
</div>

+ inference/sample

在推理过程中，从随机高斯噪声出发进行采样，迭代求解。注意每次迭代求的时候， 也要和前向过程类似加入随机噪声(这是为什么呢？请思考生成式模型多样性的解决办法！）。具体的公式可以参考原始论文/代码进行推导。

WaveGrad 的一个优化是对推理步骤，正常来讲 `T=1000`，这样一个速度是无法接受的。 代码里对此进行了精简，只用 `6` 次推理。为了保证 `T` 降低后效果损失较小，作者选取了一个验证集， 在验证集上进行测试，选取最优参数。(这应该也是一个相对比较容易想到和实现的方案了)

#### 3.实验结果

这里就不放论文本身的结果了，但凡能发论文的，大家都会说自己的方法吊打他人。我自己重新训练和测试了一轮，包括解码时候仔细调了一下参数。我使用了 bzn 的经典数据集 (1w句)， 效果很不错，至少不会比 HiFi-GAN 差（6steps）, 不过确实解码速度比较慢，即使只用6steps， 依然会比hifigan 慢一个数量级。

性能依然是 WaveGrad 方法的一个瓶颈。

#### 4.总结

整体上来说，因为作者放了论文的源码，复现起来比较顺利。bzn 数据质量很高，真是做声码器问题的黄金数据。

遇到的瓶颈主要在于公式推导上，有两天时间都在纠结 朗之万动态采样公式中随机噪声的系数取值， 后来看苏神的博客说，直接取和正向相同的 beta 即可。我自己在推理过程中，大概测试了一下，发现其实不敏感。 （系数调大一倍对结果基本不影响） 另外，花了一点时间在研究 sgd 和 langevan dynamic 之间的关系， 数学上很多推导虽然没太看懂，但看得很有乐趣。 再次认识到，我果然只适合写写偏应用层面代码，没有研究偏理论算法的天赋。

**WaveGrad 更多是一个比较有启发意义的工作，距离使用还有距离**。毕竟是20年底的工作了， 还有一些更新的 DDPM for Vocoder 的工作在解决性能的相关问题。

------

### 7. HiFiGAN V1/V2: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

!> v1: https://arxiv.org/abs/2006.05694

!> v2: https://ieeexplore.ieee.org/document/9632770

!> v2: https://daps.cs.princeton.edu/projects/Su2021HiFi2/

!> V2: https://pixl.cs.princeton.edu/pubs/Su_2021_HSS/Su-HiFi-GAN-2-WASPAA-2021.pdf

!> https://github.com/jik876/hifi-gan

!> https://jik876.github.io/hifi-gan-demo/

<!-- https://zhuanlan.zhihu.com/p/420863816 -->

#### 1.背景

GANs已经被用于raw waveforms的生成（vocoder阶段，类似于从梅尔谱到最终的音波，声音文件的阶段）。但是：效果上还赶不上（1）自回归vocoder，(2)流（flow）方法。本论文提出来HiFi-GAN，其（1）高效，（2）高保真，地实现“语音合成”。核心的点：`modeling periodic patterns of an audio -> enhancing sample quality`，即：对语音中的“周期模式”进行建模，从而可以提升样本质量。

效果：MOS上，可以类比人的水平！并且可以在单个V100 GPU上，生成比真实时间快167.9倍的22.05kHz的语音！也就是说，生成1秒的语音，只需要大概1/168=0.006秒的时间。这个真实太赞了！是面向实际应用的！此外，即使是在CPU上，（缩减版的）HiFi-GAN也可以产生比真实时间快13.4倍的语音。而且这个语音的质量可以媲美“自回归模型”。

#### 2.HiFi-GAN

HiFi-GAN由一个生成器，两个判别器构成。一个判别器负责multi-scale，多尺度；一个判别器负责multi-period，多周期。生成器和两个判别器是通过对抗学习的方法训练的，新增加了两个损失函数来提高训练的稳定性和提高模型的性能。

##### 2.1 生成器

生成器是一个“纯的”卷积神经网络。输入是梅尔谱，然后一直用卷积（transposed convolutions）来上采样，一直到输出的序列的长度和原始波形的时域分辨率相同为止。每个transposed convolution的后面，接的是一个multi-receptive field fusion (MRF)【多感受野融合】模块。下图是生成器的示意图。注意，白噪声没有作为该生成器的输入。（即，输入只有梅尔谱）

<div align=center>
    <img src="zh-cn/img/ch7/07/p1.jpg" /> 
    <p>HiFi-GAN中的生成器的示意图；注意：这是在说，Dr是有三个维度，具体取值可以参考Table 5，例如[[1,1], [3,1], [5,1]])</p>
</div>

**生成器中的MRF:** Multi-Receptive Field Fusion(MRF)，多“感受野”融合模块。

+ MRF输出的是，多个残差块（$|k_r|$个块）的输出的“相加”（上图中的那个“圆圈+”号！）；
+ 每个残差块，使用不同的kernel size，以及膨胀率(dilation rates），从而构成多样的“感受野”模式。

上图中，也绘制了，第$n$个残差块的样子（最右边）： 
+ 首先，$m$遍历从$1$到$|D_r[n]|$（第$n$个残差块的最大膨胀率）；其次，$l$遍历从1到$|D_r[n, m]|$； 
+ 然后，历经一个Leaky ReLU，和一个卷积，其kernel size为：$k_r[n] \times 1$，而且膨胀率为$D_r[n, m, l]$(注意：这是在说，$D_r$是有三个维度，具体取值可以参考Table 5，例如`[[1,1], [3,1], [5,1]]`)；
+ 最后，有个“圆圈+”的残差运算。

这里面，有几个可调节的参数：

+ 隐层维度，$h_u$；
+ transposed卷积的核函数尺寸，$k_u$；
+ MRF中的卷积的核函数尺寸，$k_r$；
+ MRF中的膨胀率$D_r$。

这几个参数的取值，都可以根据“合成效率”和“样本质量”的平衡来进行调节。

> 这个MRF的设计，还是很赞的，感觉糅合了多个size的卷积，以及多种膨胀率，从而可以更好地从不同的窗口来提炼梅尔谱的特征信息，用于生成最后的原始音波。简单总结：“很胖”！

不少细节，还不清晰,看看他们的附录A里面的图，和关于四种参数的取值的例子。

<div align=center>
    <img src="zh-cn/img/ch7/07/p2.jpg" /> 
    <p>生成器的图的细化版本。把MRF的周边，绘制的更详细了</p>
</div>

上面是细化后的生成器的图。可以看到，把MRF的周边（前后）绘制的更加详细了。特别是：
+ 进$|k_u|$个模块之前的，一个`7*1`的卷积层，其膨胀率为1，$h_u$个channel；
+ 出来的时候，先走Leaky ReLU；
+ 然后又是一个`7*1`的卷积，1个channel；
+ 最后走一个非线性激活函数:$Tanh(\cdot)$。

然后看一下四种参数的取值：

<div align=center>
    <img src="zh-cn/img/ch7/07/p3.png" /> 
    <p>三种不同配置下的生成器，以及分别的hu, ku, kr, Dr的取值</p>
</div>

可以看到：
+ 隐层维度$h_u$；这个的取值，是一个标量，例如512, 128, 以及256
+ Transposed卷积的核函数尺寸$k_u$；这是一个列表集合，那么生成器中的$l=1, 2,3,4$，即一共四个`[leaky ReLU + Transpose 卷积+MRF]`的“块”，然后，每个“块”里面的卷积的kernel size的取值为`16, 16, 4, 4`。即：

<div align=center>
    <img src="zh-cn/img/ch7/07/p4.jpg" /> 
    <p>V1配置下的生成器的展开示意图</p>
</div>

+ MRF中的卷积的核函数尺寸$k_r$；列表集合。这是“并行”三种卷积，卷积核大小分别为`3,7,11`。卷积之后，相加。
+ MRF中的膨胀率$D_r$。这个值是三个维度为`(3,3,2)`的张量。

把上面两个点结合起来，并且进一步在V1配置下，展开生成器的一个MRF的构造为：

<div align=center>
    <img src="zh-cn/img/ch7/07/p5.jpg" /> 
    <p>一个MRF的展开示意图，三个残差块，每一块又分别展开成三个部分，每个部分里面，又包括了“两次”“leay ReLU+卷积层”的重复</p>
</div>

上图，是一个MRF的展开示意图：
+ 三个残差块，
+ 每一块又分别展开成三个部分，
+ 每个部分里面，又包括了“两次”“leay ReLU+卷积层”的重复。卷积层的卷积核的size，取值分别为`3，5，11`。膨胀系数分别为`1，3，5`。

如此，就实现了“多感受野”下，对最初的输入的梅尔谱的特征进行建模的目的。

##### 2.2 判别器

识别真实语音中的“长距离依赖”是TTS中比较关键的点。例如，一个“音素”可能持续100ms（0.1秒），这就涉及到2200个采样点的相互关联的建模的问题。也就是说，第一个采样点和最后一个采样点，都属于这个“音素”，那么他们之间也存在所谓“长距离依赖关系”。之前的工作，为了解决上面的长距离依赖问题，有使用扩张“感受野”的方法。但是，本文中，关注的是另外一个问题！而这个问题目前为止还没有被好好研究对待过。这就是，多种周期的正弦波，是合成一个语音的核心要素！（考虑傅里叶变换的时域到频域的转换！）即：**语音数据里面的多样的周期模式（diverse periodic patterns），需要被建模**。

故此，本文提出了**MPD: multi-period discriminator（多周期判别器）**。它包括几个“子判别器”，每个判别器负责一部分周期信号。更进一步，为了更好地捕捉连续的patterns，以及长距离依赖，本文也使用**multi-scale discriminator（MSD）**，这个是在MelGAN (Kumar+2019)里面被提出来的。

<div align=center>
    <img src="zh-cn/img/ch7/07/p7.png" /> 
    <p>MPD with period $p$</p>
</div>

通过MPD和MSD的相结合，本文的判别器，和生成器一起，取得了更好的效果。再回顾一下MelGAN的架构（生成器和判别器的样子）：

<div align=center>
    <img src="zh-cn/img/ch7/07/p6.jpg" /> 
    <p>MelGAN中的生成器和判别器</p>
</div>

这个生成器和判别器（若干个）中规中矩。生成器中主要是残差网络+上采样卷积的架构。判别器中，主要是下采样加卷积。

然后，本文使用的特征匹配Loss (feature matching loss)，在MelGAN中也更早用到了：

$$\mathcal{L}_ {FM}(G,D_k)=\mathbb{E}_ {x,s\sim p_{data}}[\sum^{T}_ {i=1}\frac{1}{N_i}||D^{(i)}_ {k}(x)-D^{(i)_ {k}(G(s))}||_ 1]$$

**判别器中的MPD**: MPD是若干“子判别器”的混合。每个“子判别器”只接受一个输入语音的“等间隔的样本”（equally spaced samples? 这个是啥意思？）这个space的“周期”被给定为p【何意？】。 ”子判别器“设计的目的，就是捕捉一个输入语音的不同部分的不同的（区分于其他部分的）隐含的结构。这里的周期被设置为`[2,3,5,7,11]`，从而避免重叠。

如下图(b)所示，我们首先把一维度的长度为$T=12$的（例如，$T$个frames）原始语音信号，塑型为二维的数据，其宽度为$p=3$，高度为$T/p=12/3=4$（只不过，这里的“宽度”是行的个数；“高度”是列的个数。）。然后，在这个二维数据上应用2D卷积。

<div align=center>
    <img src="zh-cn/img/ch7/07/p8.jpg" /> 
    <p>MSD和MPD的“子模块”的示意图</p>
</div>

在应用二维卷积到这个`[3, 4]`的矩阵的时候，我们限制在width轴上（即宽度上）的核函数size=1，从而保证每行之间“互不干扰”。这也对应了，”独立地处理周期性的样本（process the periodic samples independently）“这个”初心“。形式化的表示就是：

<div align=center>
    <img src="zh-cn/img/ch7/07/p9.jpg" /> 
    <p>右上部分，给出了`k=3`的时候，所谓`k*1=3*1`的卷积的应用范围（红色矩形框)</p>
</div>

每个”子判别器“中，堆砌了若干`【带步幅的卷积层，以及使用Leaky ReLU激活】`。随后，weight normalization应用于MPD。通过把输入语音信号重塑为二维数据，而不是对语音的周期性信号进行采样，MPD的梯度可以被传播到输入语音的每个时间步。【也就是说，如果是采样的话，那么无法保证每个时间步都被选择，从而也无法保证每个时间步都被更新梯度】

**判别器中的MSD:** 

`MSD=multi-scale discriminator`，多刻度判别器。 鉴于MPD只接受没有交集的（不同周期的）样本点，我们还需要引入MSD从而可以连续地评估（real/fake分类）语音序列。MSD包括了三个”子判别器“，分别在不同的输入刻度上运算：
+ 原始语音；
+ 进行了$\times 2$ average-pool的语音；
+ 进行了$\times 4$ average-pool的语音。

如上面的图中的(a)所示。【数了一下，上图中，应该是4个点进行一次”平均池化“，那不应该是第三个”子判别器“吗？】。 MSD中的每个子判别器，是若干，带步幅的，group的卷积层，加上Leaky ReLU激活函数。判别器的size，通过减小步幅以及增加更多的层数，来被增大。

第一个子判别器，直接作用到原始的语音信号；而且这个子判别器中没有用weight normalization（权重标准化），而是用的谱标准化（spectral normalization，Miyato+ 2018)。据报告，使用谱标准化，可以使得训练更加稳定。【赞！稳定最重要，一般训练GAN，经常出现不稳定的问题。。。】。后续的两个”子判别器“，都应用了权重标准化。

也就是说，MPD是在原始音波的无交集的采样点上执行运算；但是，MSD是在平滑后的波形上面执行运算。两者有相辅相成之妙。

<div align=center>
    <img src="zh-cn/img/ch7/07/p10.jpg" /> 
    <p>MPD中的周期为p的一个”子判别器“的展开图</p>
</div>

上图中，给出了MPD中的周期为p的一个”子判别器“的展开图。鉴于$l=1,2,3,4$，所以有四个”卷积+Leaky ReLU“的模块的堆积。（居然没有残差。。。）。之后，是一个`5*1`的卷积，而且channels=1024。再接一个leaky ReLU和一个`3*1`的卷积（输出channel=1，对应了real/fake的判别）之后结束。

#### 3.损失函数

+ 判别器：真实samples分类为1，而来自生成器的分类为0；
+ 生成器：尽量欺骗判别器，使得自己产出的波形被判定为越接近1越好。【类似于，”假钞“造的越逼真越好！】

<div align=center>
    <img src="zh-cn/img/ch7/07/p11.jpg" /> 
    <p>被拆开的两个损失函数，都是期望最小化</p>
</div>

上面是通过注释的方式，简单解释了一下，论文中的两个损失函数。

+ $L(D; G)$是从$D$，即判别器，的角度来看的；
+ $L(G; D)$是从$G$，即生成器，的角度来看的。

**梅尔谱Loss:**

<div align=center>
    <img src="zh-cn/img/ch7/07/p12.jpg" /> 
    <p>梅尔谱损失函数，比较的是reference 语音的梅尔谱和通过生成器自动生成的梅尔谱之间的1-范数</p>
</div>

上面给出了，梅尔谱损失函数，比较的是reference 语音的梅尔谱和通过生成器自动生成的梅尔谱之间的1-范数。这个损失函数，有助于生成器来根据一个输入的条件（an input condition）来合成一个真实世界的音波。并且可以稳定训练过程的早期阶段。这个Loss只用于G，即生成器的”质量控制“。

**特征匹配Loss:**

<div align=center>
    <img src="zh-cn/img/ch7/07/p13.jpg" /> 
    <p>特征匹配Loss，对于G来说，当然是希望求和公式里面的1-范数的值，越小越好</p>
</div>

上面的公式中，$T$表示判别器中的层数。（这个有点不太好，最好用其他字母表示。。。因为前面，$T$已经是输入的音波的长度了。。。）。目的也是尽量让判别器在中间阶段，都不要”跑偏“：即，D的每个神经网络层都分不清”真币“和”假币“，那才对G最好了。

**最终的Loss：**

<div align=center>
    <img src="zh-cn/img/ch7/07/p14.jpg" /> 
    <p>生成器的Loss是三合一；判别器的不变</p>
</div>

这里面有两个超参数，其中$\lambda_{fm}=2, \lambda_{mel}=45$。（45，这个有点诡异。。。）

考虑到，本文是使用了若干个（$K$个）判别器，那么上面的损失函数，还需要考虑多个D的情况。也就是说，我们需要在每个D出现的时候，把一个D，扩展为$k=1...K$个D。也就是说：

<div align=center>
    <img src="zh-cn/img/ch7/07/p15.jpg" /> 
    <p>Loss的一个D被扩展为$k=1...K$个D的展开式</p>
</div>

从公式（5），（6），过滤到（7），（8）没啥难度。

#### 4.实验

数据：LJSpeech data，这个是经典的英文数据了。包括13100个短的audio clips，大概24小时（单人女性）。22kHz，而且是16-bit PCM。

SOTA基线包括了：
+ MoL wavenet;
+ waveglow
+ melgan

单张NVIDIA V100 GPU.

精度和速度：

<div align=center>
    <img src="zh-cn/img/ch7/07/p16.png" /> 
    <p>HiFi-GAN和几个SOTA基线的对比，可看到，在精度和速度上都有优势</p>
</div>

上面的表格，给出了MOS和速度的评估。HiFi-GAN和几个SOTA基线的对比，可看到，在精度和速度上都有优势！不过明显WaveNet和WaveGlow的结果，比他们的各自论文里面的结果要差一些的样子。


消融实验：

<div align=center>
    <img src="zh-cn/img/ch7/07/p17.png" /> 
    <p>消融实验分析，去掉MPD之后，效果下降最厉害</p>
</div>

上面的表格是消融实验分析。可以看到，MPD的作用最不能被无视！

扩展到未知speakers：

VCTK数据集下，9个speakers的50个语音。

<div align=center>
    <img src="zh-cn/img/ch7/07/p18.png" /> 
    <p>新speakers下的效果评估，HiFi-GAN已经非常接近于真实的Ground Truth了</p>
</div>

可以看到，新speakers下的效果评估，HiFi-GAN已经非常接近于真实的Ground Truth了，赞一个！

端到端语音合成：

效果在上面的Table 4中。这里是用了Tacotron2作为生成梅尔谱的工具。可以看到，HiFi-GAN还是非常接近ground truth的！

#### 5.总结和思考

本文有几个特点，
+ 一则是大量用了卷积网络，大大加快了解码速度，并且还能保证精度。
+ 二则，提案了基于periods的MPD判别器，可以很好地对和周期相关的特征进行建模。并且引入了多尺度判别器。两者结合，MPD+MSD，效果更好。
+ 三则，在生成器中，新增加了两个损失函数，一个是梅尔谱的相似度计算，另外一个是判别器的内部各层的特征值的比较，从而尽量保证，让判别器的各层都”犯迷糊“！
+ 最后，无论是精度还是速度，特别是参数规模，都显著好于几个SOTA的baselines。

本文的idea值得用！

------

<!-- ### 8. UnivNet

!> https://arxiv.org/abs/2106.07889
 -->
### 8. WaveGlow: A Flow-Based Generative Network for Speech Synthesis

!> https://arxiv.org/pdf/1811.00002.pdf

!> https://github.com/NVIDIA/waveglow

!> 基于流的生成模型： https://zhuanlan.zhihu.com/p/351479696

<!-- https://zhuanlan.zhihu.com/p/355219393 -->
<!-- https://blog.csdn.net/weixin_42721167/article/details/115493648 -->

#### 1.引言和简介

首先看名字waveglow，融合了wavenet和glow两个工作的新的神经网络架构。其中wavenet中的膨胀卷积层，仍然在waveglow中被使用。而glow的思想也被使用：基于流模型的（基于梅尔谱为memory指导的）一系列对正态噪音的建模构建最终的语音信号的输出。 目的：to provide fast, efficient, and high-quality audio synthesis。更快，更高效，以及更高质量的语音信号的输出。

可圈可点之处:

+ 其一，只有一个损失函数，而且其就是最大化training data的分布的likelihood——最大似然估计。而且是直接针对训练数据的最大似然估计！这就使得整个训练过程简单而稳定。（相对应的，例如GAN这样的生成式网络的训练，就存在训练不稳定的问题）；
+ 其二，inference的速度足够快，例如基于开源pytorch的实现，在NVIDIA V100 GPU上，可以达到实时生成最大500kHZ的采样率的声音；
+ 其三，并且训练数据的构造，是无监督的方法=有了语音之后，就可以自动地从语音截取片段（例如一秒长度），然后转换成梅尔谱。之后，<梅尔谱，语音信号>的pairs，就可以作为waveglow的训练数据了。是不是想要多少要多少？！（因为不涉及文本方面的输入！）；关于“梅尔谱”的细节，可以参考：<https://zhuanlan.zhihu.com/p/356364039>
+ 其四，效果足够好，在MOS (mean opinion scores - 人工评价得分）下和已有的wavenet的实现效果基本持平（实际根据情况而定，还有若干差距）。好在速度足够快。

#### 2.理论基础

直接上关于waveglow的模型介绍，会有若干公式，尽量通过例子来把他们讲清楚，理解清楚。出场人物：

其一，$z\sim N(z;0,I)$ ，符合（高维）标准高斯分布的白噪声输入，其均值为0，方差矩阵为单位矩阵。维度未定，例如100维，等等；

其二，$x=f_0 \circ  f_1 \circ ... f_k(z)$，语音信号的输出，这里省略了conditioned on a mel-spectrogram（基于输入的梅尔谱作为限制条件，从白噪声z，来逐步生成语音信号序列x=inference的过程）。

其三，$f$ 从$k$到$0$，是一系列的神经网络（函数），一个$f_i$是负责把一个输入张量$z_i$进行一次非线性变换得到$z_{i-1}$。类似如下的示意图：

<div align=center>
    <img src="zh-cn/img/ch7/08/p1.jpg" /> 
</div>

上图展示的是inference的过程！即，从白噪音逐步得到想要的语音信号的过程。特别需要注意两个边界条件，左边的一个是起点，有 
$z=z_k$，纯白噪声（高斯分布），最右边的一个是根据最后一个函数$f_0$ ，其负责把$z_0$再次进行非线性变换，得到最后的 
$x$作为输出。

而训练的过程，正好和上面的inference的过程相反，训练是给定$x$，去训练一系列函数$f$的逆函数（网络）中的参数，并使得每个$x$经过这一系列（排好顺序的）网络之后，最终得到的都是接近于“高斯分布”的向量。

下面是论文中给出的公式（有瑕疵）：

<div align=center>
    <img src="zh-cn/img/ch7/08/p2.jpg" /> 
</div>

上面的公式(4)是ok的，即类似于下图的示意：

<div align=center>
    <img src="zh-cn/img/ch7/08/p3.jpg" /> 
</div>

需要注意的是，如果$f_i$的输入是$z_i$，输出是$z_{i-1}$；则其逆函数的输入则是$z_{i-1}$，输出是$z_i$。如上图的黄色箭头所示。

问题出现在公式(3)，首先肯定不是$i=1$到$k$，只有$k$个$f$函数。而是应该有$i=0$到$k$，有$k+1$个$f$函数。其次，不是每次都是用$x$作为输入的，所以 $f_i^{-1}(x)$中的$x$就很突兀了。真实情况如上图，不是每个$f$的逆函数，都是直接用$x$为输入的！

正确的形式应该类似于：
$$log p_{\theta}(x)=logp_{\theta}(z)+\sum_{i=0}^{k}log|det(\mathcal{J}(f_i^{-1}))|$$

上的公式中省略了每个$f$的逆函数的输入，具体的输入的值，可以参考上图。或者是：

$$log p_{\theta}(x)=logp_{\theta}(z)+\sum_{i=0}^{k}log|det(\mathcal{J}(f_i^{-1}(z_{i-1})))|$$

这里的$z_{-1}=x$,$z_i=f_i^{-1}(z_{i-1})$.

上面公式的第一项 $log p_{\theta}(z)$是高斯分布的对数似然度。第二项则是来自“change of variables"(变量变换定理）使用的是雅可比矩阵的行列式的值。

其实，即使有了上面的公式，距离实现真正的waveglow，中间还差着十万八千里。下面看论文中最神奇的一个图吧，很多其他的讲解，都是基于这个图进行解释的。不过坦白说，这个图，极其不容易理解：

首先需要说的是，这个是”训练“的阶段的图：

<div align=center>
    <img src="zh-cn/img/ch7/08/p4.jpg" /> 
</div>

上面的训练阶段示意图中的”登场人物“的简介:

+ 第一个，语音向量$x$，这个是来自训练数据的真实的语音向量数据；其会被切分成左右两段，左边的是$x_a$，右边的是$x_b$。（比较细化的问题是，是怎么具体切的呢？有没有具体的例子呢？容后续讲解到代码的时候，详细介绍。）

+ 第二个，梅尔谱输入，这里非要加一个upsampled mel-spectrogram（上采样之后的梅尔谱）就说明，在真实的从语音wav文件得到的梅尔谱的基础上，会有一次线性变换，来实现上采样；

+ 第三个，一系列的$f$函数的逆函数。注意的是，inference是正用$f$，从$x$到$z$；而训练的时候，是使用的$f$的逆函数，从$z$到$x$。这里边，更加细化为了两个网络层，3.1的可逆`1*1`的卷积，以及3.2的affine coupling layer（仿射耦合层），这两个网络结构，在我之前的文章中有详细的介绍，这里不再详细讲了。后续会结合代码详细介绍。

+ 第四个，`WN=wavenet`的实现，$x_a$经过wavenet之后，得到的是两种输出，$t$和$logs$，这里的$logs$其实也不容易理解。简单起见，可以认为输出的就是$t$和$s$，然后在后续使用$s$的时候，用的是 $e^s$
。如下面的几个公式所示：

<div align=center>
    <img src="zh-cn/img/ch7/08/p5.jpg" /> 
</div>

这里的公式(6)也是有问题的，左边的输出，应该是$(t, logs)$，而不是现在的$(logs, t)$，即得到的从WN的输出里面，前半段给了t，后半段给了logs。

+ 第五个，第六个分别是$t$和$logs$了，这个后续在代码阶段详细介绍。
+ 第七个，affine transform，仿射层（可以简单理解为一个linear layer），对应的公式是上面的公式(7)。
+ 第八个，架构图中标注为8.1和1.1的张量相同，都是$x_a$ ，直接copy的结果；另外8.2  $x_{b^{'}}$
 （图中被绘制成了 $x_b^{'}$，其实应该是 $x_{b^{'}}$）就是根据公式(7)计算得到的了。

在对$x_a$ 和 $x_b$进行串联之后，就是3.2这个仿射耦合层的输出了。

按照个人理解，把上面的四个公式整理为：

<div align=center>
    <img src="zh-cn/img/ch7/08/p6.png" /> 
</div>

和普通的基于自回归的wavenet不同的是，这里的wavenet，并没有使用自回归的方式来逐步构造输出序列。后续在介绍代码的时候，会详细介绍其架构。论文中的下面一段讲述，其实并不容易理解：

<div align=center>
    <img src="zh-cn/img/ch7/08/p7.jpg" /> 
</div>

可以参考上面的这个雅可比矩阵的样子来理解上面的公式，左上角是单位矩阵，因为$x_a$的前半部分是直接copy到输出的，所以输出相对于输入的偏微分矩阵，得到的就是一个单位矩阵。然后看右上角，是$x_a$（输出）对于$x_b$（输入）的偏微分，因为$x_b$对于$x_a$的构造，没有参与，所以这部分是全0矩阵。当右上角是全零矩阵的时候，左下角是啥，我们也不关心了，反正和右上角相乘之后为0。最后剩下的就是右下角这个关键的部分了。可以简单理解为$x_{b^{'}}$对$x_b$的偏微分矩阵，则基于上面的公式(7)，得到的就是$s$为对角线的（其他非对角线元素都为0）的矩阵。这是因为：

<div align=center>
    <img src="zh-cn/img/ch7/08/p8.png" /> 
</div>

从而$x_{b^{'}1}$对 $x_b$的$n$个元素分别进行偏微分的时候，得到的是向量$[s_1,, 0, ..., 0]$，即只有第一个元素为$s_1$ 
 (向量$s$中的最左边/最上面的元素），其他的元素都为$0$。

继续看`1*1`卷积方面，细节也可以参考我之前的文章。这里直接解释一下论文中的公式：

<div align=center>
    <img src="zh-cn/img/ch7/08/p9.png" /> 
</div>

上面的公式(10)就是`1*1`卷积的表示，即直接对$x$进行一个线性变换，就得到结果了（卷积和线性变换的相通性）。这样的话，在计算这个网络（`1*1`卷积）的雅可比矩阵的时候，就是如(11)公式所展示的那样了，就是可训练参数（矩阵）$W$的对应的行列式值的绝对值。

下面的整体对数相似度函数的公式略微有瑕疵：

<div align=center>
    <img src="zh-cn/img/ch7/08/p10.png" /> 
</div>

上面的红色文字和线，代表了修正之后的部分。首先$j$和$k$都应该是从1开始，一共12层”仿射耦合层“，和12层”`1*1`卷积层“。然后这里的$(x, mel\_spectrogram)$是不需要的，只要有 $s_j$的1-order范数的值的绝对值就可以了。（这里估计不容易理解，还是后续结合代码的时候，再详细讲解。）。当然，$W_k$ 
 这个矩阵的行列式的值，也需要加了绝对值之后，才能传给log函数。


#### 3.Waveglow整体代码分析【训练视角】

分别（相对独立地）讲述Train的算法，和Inference的算法：

**先看Train的部分：**

<div align=center>
    <img src="zh-cn/img/ch7/08/p11.jpg" /> 
</div>

假设一次mini-batch取12个wav文件，waveglow里面可以设定一个segment的大小，例如default就是16000个采样点（即，无论wav的长度是多少，都是从中随机选择一个长度为16000个点的声音片段）。

得到shape为[12, 16000]的矩阵之后，兵分两路（左右两个箭头）：

左边，经过STFT（短时傅里叶变换，default frame size=1024, hop size=256, 使用hann window function，window size=1024），然后抽取80个mel-scale的filterbanks，可以计算出长度为16000个点的wav，一共有`(16000-frame size)/hop_size + frame.size/hop_size=63`个frame。从而得到的梅尔谱的张量shape为(12, 80, 63)。

继续左边，经历“逆卷积上采样”（可以简单理解为一个upsampling的卷积，或者简化为special linear layer），这里出现了16896这个奇怪的数字，其对应的代码和计算公式为：

```python
self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, 
                                                 n_mel_channels, 
                                                 1024, stride=256) 
# (1) 80 conv_transpose_1d: 卷积转置1d! 
#     (反卷积)卷积操作的作用类似神经网络中的编码器，
#     用于对高维数据进行低维特征提取，
#     而 反卷积 通常用于将低维特征映射成高维输入，
#     与卷积操作的作用相反。同时也是一种基于学习的上采样实现方法。
# (2) 80 
# (3) kernel_size = 1024. 
#     (Lin-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
# =(Lin-1)*256-2*0+1*(1024-1)+0+1=256*Lin-256+1024=256*(Lin-1)+1024 =
# = 256 * (63-1) + 1024 = 16896. 
#  这里Lin=输入的sequence的长度。
```

关于“逆卷积”可以参考：<http://www.suzhengpeng.com/hello-pytorch-01#%E4%B8%80%E7%BB%B4%E5%8F%8D%E5%8D%B7%E7%A7%AF-torchnnconvtranspose1d>

得到16896维度之后，只要前16000个数值（类似对应到最初的16000个wav文件中的采样点），这样就得到了(12, 80, 16000)，其中80表示80个mel-scale filterbanks。这个张量可以（简单地）理解为一个minibatch中有12个序列，每个序列16000个点，每个点使用80维度的一个特征向量(feature vector)表示。

继续左边，再往下，就有点玄幻了，目前还不是很明白其中有什么道理，类似从`(12, 80, 16000)`转成了`(12, 640, 2000)`的张量。关联到的代码如下：

```python
spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) 
# TODO unfold是干啥的？ [12, 2000, 80, 8] <- [12, 80, 8, 2000]
spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
    .permute(0, 2, 1) 
# [12, 2000, 640] -> [12, 640, 2000]

audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) 
# [12, 2000, 8] -> [12, 8, 2000] 
# TODO为什么要分成2000*8呢？8个长度为2000的序列？
```

得到了这个shape为(12=batch.size, 640, 2000)的代表了“梅尔谱”的张量之后，它会被后续的WaveNet使用，即和经历了`1*1`卷积的wav信息一起，作为WaveNet的输入。

刚才说到花开两朵，各表一枝。现在看右边的这朵花。(12, 16000)这个描述了12个序列，每个序列16000个采样点的数据，经过unfold，以及permute之后，得到shape为(12, 8, 2000)的张量。类似可以理解为每个序列，有8个通道，每个通道上2000个采样点。

右边，之后，会扔给一个“`1*1`的卷积”，里面的Weight的shape是`8*8`的.

经历了这个卷积之后，得到的张量的shape仍然是(12, 8, 2000)，同时，卷积网络会返回`log_det_W`这个值，代表的是$W$矩阵的行列式值的绝对值的对数。这个值是计算loss的时候需要用的。

然后，我们分别按照第一个维度（即，batch size的后面的那个长度为8的维度）进行切割：前半部分`[:, :4, :]`是 $x_a$；后半部分`[:, 4:, :]`是 $x_b$，这两个张量的大小都是(12, 4, 2000)。

这个时候，之前的公式就发挥作用了，我们再看下：

<div align=center>
    <img src="zh-cn/img/ch7/08/p12.png" /> 
</div>

其对应的代码是：

```python
output_audio = []
log_s_list = [] # 为了最后计算loss
log_det_W_list = [] # 为了最后计算loss

for k in range(self.n_flows): # 12，类似从f0到f11
    # k=4, 8 的时候，会有对结果的“收割”：
    if k % self.n_early_every == 0 and k > 0: # n_early_every=4
        output_audio.append(audio[:,:self.n_early_size,:]) 
        # n_early_size=2
        audio = audio[:,self.n_early_size:,:]

    audio, log_det_W = self.convinv[k](audio) # 1*1的卷积
    log_det_W_list.append(log_det_W)

    n_half = int(audio.size(1)/2)
    audio_0 = audio[:,:n_half,:] # audio_0是前半，即x_a
    audio_1 = audio[:,n_half:,:] # audio_1是后半，即x_b

    output = self.WN[k]((audio_0, spect)) 
    # spect=梅尔谱(12, 640, 2000)，WN=WaveNet

    log_s = output[:, n_half:, :] # log_s是后半！
    #另外，俺只是名字叫log_s，不代表俺就是调用了log之后的结果！

    b = output[:, :n_half, :] # b是前半！！！即论文中的t

    audio_1 = torch.exp(log_s)*audio_1 + b # 得到x_b'

    log_s_list.append(log_s) # log_s的shape是(12, 4, 2000)

    audio = torch.cat([audio_0, audio_1],1)
    # 得到的audio的shape为：[12, 8, 2000]

output_audio.append(audio)
```

特别的，这段代码中用到了12个`1*1`卷积网络，以及12个WaveNet网络。这两个网络的细化的代码，后续会逐步介绍。特别需要注意的是WaveNet，同时接受来自audio的信息和来自梅尔谱的信息：进一步，在训练的时候，接受的是reference audio信息；而在inference的时候，接受的是noisy audio信息（白噪音）。

一层WaveNet，的输出的张量shape是(12, 8, 2000)，再次对长度为8的那个维度，对半拆开。截取出前半部分是代码中命名为b，且论文中命名为t。其shape为(12, 4, 2000）。而截取出后半部分，命名为`log_s`（不代表就经过了log函数，其实log_s还是一个float参数，和log函数没有直接关系！！）。

再之后，就是使用如下的公式（其有个“高！大！上！”的名字：affine transform - 仿射变换）：

<div align=center>
    <img src="zh-cn/img/ch7/08/p13.png" /> 
</div>

来计算新的 $x_{b^{'}}$。这个张量的shape也是(12, 4, 2000)，然后会和最初的 按照`dim=1`进行串联，得到的audio的shape为(12, 8, 2000）。

“天道好轮回”！--- 是否可以注意到，这个在图中被标识为绿色底色的变量，和最初扔给`1*1`卷积的那个张量的形状是一样的呢？！也就是说，之后可以开启`k=1`到`11`的`11`个循环，都是先后经过`1*1`卷积，以及结合了梅尔谱输入的WaveNet。

下面的图，展示了从`k=0`的输出，可以扔给`k=1`的输入；以此类推，`k=1`的输出，扔给`k=2`的输入；`k=2`的输出，扔给`k=3`的输入。`k=3`执行完毕之后，我们得到的audio的shape仍然是(12, 8, 2000)。不过这个audio就是“充分”融合了梅尔谱信息的audio：

<div align=center>
    <img src="zh-cn/img/ch7/08/p14.jpg" /> 
</div>

在进入`k=4`的时候，会有一波“收割”（即把audio的维度为8的，前两个维度的值，保存到最终结果中，类似于(12, 2, 2000)存入output_audio），收割之后，剩余的是(12, 6, 2000)会继续参与到`k=4, 5, 6, 7`的四层block中来：

```python
# k=4, 8 的时候，会有对结果的“收割”：
if k % self.n_early_every == 0 and k > 0: # n_early_every=4
    output_audio.append(audio[:,:self.n_early_size,:])
    # n_early_size=2
    audio = audio[:,self.n_early_size:,:]
```

这里的output_audio存放的就是最终网络执行完毕的时候的输出（对应到audio张量）。

<div align=center>
    <img src="zh-cn/img/ch7/08/p15.jpg" /> 
</div>

上图中的红色长方块，即表示了(12, 2, 2000)会扔给最终的audio_output，然后剩下的(12, 6, 2000)会扔给`k=4,5,6,7`的四层blocks，（包括一个`1*1`卷积，和一个结合梅尔谱的WaveNet网络）。类似的`k=4`的输出，会作为`k=5`的输入，`..., k=7`的输出，会再次被截断，`[:, :2, :]`会被保存到audio_output，然后剩下的`[:, 2:, :]`会被扔给`k=8,9,10,11`这四层blocks。

相应的，卷积网络中的weight的大小，在`k=8 - 11`的时候，是`4*4`的。类似的图如下：

<div align=center>
    <img src="zh-cn/img/ch7/08/p16.jpg" /> 
</div>

当`k=11`执行完毕之后，最后得到的audio的shape是(12, 4, 2000)，它会被扔给output_audio。

如此，output_audio中就包括了三个张量：

第一，`k=3`的输出的前一部分(12, 2, 2000),

第二，`k=7`的输出的前一部分(12, 2, 2000)，以及，

第三，最后`k=11`的输出的(12, 4, 2000)。

然后可以调用`torch.cat(output_audio, dim=1)`这样的方法，得到最终的forward函数的输出张量，shape为（12， 8， 2000），也就是所谓`z`了（服从高维高斯分布的“確率変数”）【输出一】。

除了这个张量，forward函数，还有另外两个输出，分别是：

【输出二】：`log_s_list`，包括了12个`log_s`（张量）：前四个的shape是(12, 4, 2000)；中间四个的shape是（12，3，2000）；最后四个的shape是（12，2，2000）！（为计算loss用的）

【输出三】：`log_det_W_list`，包括了12个标量，每个标量都是一层`1*1`卷积中的weight矩阵的行列式的值的绝对值的对数！（为计算loss用的）

话赶到这里了，正好顺带把waveglow的loss function给过一遍得了：

如下这个损失函数class，在`glow.py`文件里面：

```python
class WaveGlowLoss(torch.nn.Module): 
    # 损失函数, negative log likelihood (NLL)
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma #default = 1.0

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
　　　　# 重新温习，z，就是(12, 8, 2000)这样的服从高维球面高斯分布的“自由变量”

        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s) # log_s_total记录log_s求和结果
                log_det_W_total = log_det_W_list[i] 
                # log_det_W_total记录log_det_W求和结果
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                # 直接对log_s这个张量中所有元素求和！（没有绝对值啥的骚操作）
                # 哦！扔给log函数之前，没有用绝对值？！
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) 
                - log_s_total - log_det_W_total 

        # (e.g.,) loss = tensor(694.3699, grad_fn=<SubBackward0>)
        return loss/(z.size(0)*z.size(1)*z.size(2)) 
        # 对最终输出的loss值进行normalization处理。
        # (e.g.,) loss = tensor(0.0036, grad_fn=<DivBackward0>), 694.3699/(12*8*2000)=0.0036
        # 12=batch.size, 8=seq.length/channel_size, 2000=hidden.size
```

而上面这个类的forward函数，恰恰就对应了如下的公式：

<div align=center>
    <img src="zh-cn/img/ch7/08/p10.jpg" /> 
</div>

特别需要注意一点：代码中并没有所谓$log|s_j|$这样的，先加绝对值，然后扔给log函数的操作啊？为啥呢？这是因为`log_s_j`本身就是一个变量的名字，这个变量就是“一步到位”的（1）保证 $e^{log\_s\_j}$
 为正，（2）又保证了不需要直接求 $s_j$
 （定义个名为`log_s`的变量，然后以后使用 $e^{log\_s}=s$就好了嘛！nb的实现方法！

那$log det|Wk|$需要在det外部有个绝对值吗？还不知道，因为还没有分析WaveNet在这里的详细的架构。

##### 4.【训练视角】详细分析WaveNet的架构和代码

对于WaveNet不是很了解，或者没有仔细研究过WaveNet代码的童鞋，这块其实还是相对比较难的，独立出来一个章节。
分析的类，是`glow.py`里面的`class WN(torch.nn.Module)`。
在前面用到了12个WN，从`WN[0]`到`WN[11]`。
在WaveGlow类中，有12个WN的初始化的方法：

```python
for k in range(n_flows): 
    # k=0 to 11. 类似于：0, 1, 2, 3,     
    #  4[diff], 5, 6, 7,     8[diff], 9, 10, 11

    if k % self.n_early_every == 0 and k > 0: # n_early_every=4
        n_half = n_half - int(self.n_early_size/2)
        n_remaining_channels = n_remaining_channels - self.n_early_size

    self.convinv.append(Invertible1x1Conv(n_remaining_channels))
    # k=0/1/2/3, Invertible1x1Conv(8)
    # k=4/5/6/7, Invertible1x1Conv(6)
    # k=8/9/10/11, Invertible1x1Conv(4)

    self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
    # k=0/1/2/3, n_half=4, WN(4, 80*8, 
    #        {'kernel_size': 3, 'n_channels': 256, 'n_layers': 8})
    # k=4/5/6/7, n_half=4-1=3, WN(3, 80*8, 
    #        {'kernel_size': 3, 'n_channels': 256, 'n_layers': 8})
    # k=8/9/10/11, n_half=3-1=2, WN(2, 80*8, 
    #        {'kernel_size': 3, 'n_channels': 256, 'n_layers': 8})
```

我们先看前4个WaveNet，中的`k=0`的情况：

```python
# class WN(torch.nn.Module)中的forward方法：
def forward(self, forward_input):
    audio, spect = forward_input 
    # audio=[12, 4, 2000] (i.e., half part), spectrogram=[12, 640, 2000]
    # 12=batch size; 4=channel number; 2000= hidden size
    audio = self.start(audio) 
    # [12, 256, 2000], self.start是4 channels到256 channels的conv1d的网络!

    output = torch.zeros_like(audio) # output=[12, 256, 2000]
    n_channels_tensor = torch.IntTensor([self.n_channels]) # 256

    spect = self.cond_layer(spect) 
    # self.cond_layer = Conv1d(640, 4096, kernel_size=(1,), stride=(1,)). 
    # 因此spect是从[12, 640, 2000]到[12, 4096, 2000]
    # 4096的时候：4096=16*256；所以分成了八层，
    # 每一层对应的是2*256个“点”（position?)

    for i in range(self.n_layers): # n_layers=8，i是从0到7
        spect_offset = i*2*self.n_channels 
        # 谱的offset, i=0时0; i=1时2*256; i=2时4*256; i=3时6*256; 
        # i=4时8*256; i=5时10*256; i=6时12*256; i=7时14*256.

        acts = fused_add_tanh_sigmoid_multiply(
            self.in_layers[i](audio), 
            # audio=[12, 256, 2000] -> [1, 512, 25472]
            spect[:,spect_offset:spect_offset+2*self.n_channels,:], 
            # [1, 512, 25472]
            n_channels_tensor) # 256
        # output acts=[1, 256, 25472]
        res_skip_acts = self.res_skip_layers[i](acts) 
        # res_skip_acts=[1, 512, 25472]

        if i < self.n_layers - 1: # i<7
            audio = audio + res_skip_acts[:,:self.n_channels,:] 
            # audio和res_skip_acts的前半段叠加：得到[1, 256, 25472]
            output = output + res_skip_acts[:,self.n_channels:,:] 
            # output和res_skip_acts的后半段叠加，得到[1, 256, 25472]
        else:
            output = output + res_skip_acts

    return self.end(output) 
    # self.end = Conv1d(256, 4, kernel_size=(1,), stride=(1,)). 
    # output=[1, 256, 25472] -> self.end -> [1, 4, 25472]
```

<div align=center>
    <img src="zh-cn/img/ch7/08/p17.jpg" /> 
</div>

我们分析的就是上面图中的`self.WN[0]`网络。该网络的输入包括，`xa:audio_0`，即前半部分的audio wav过来的信息，以及`up-mel-spec`，即经历过上采样之后的“梅尔谱”信息（类似于memory）。

虽然不太准确，不过这两个输入张量，可以分别简化理解为，audio这边是`batch.size=12`，每个batch有12个序列，每个序列2000个“点”，然后，每个点是用4维特征向量表示；对于`up-mel-spec`，也是类似，`batch.size=12`, 每个batch有12个序列，每个序列2000个“点”，然后每个点使用一个640维度的特征向量表示。

输出的是shape是(12, 8, 2000)的张量，类似于对输入的来自audio的4维特征向量，和来自梅尔谱的640维度特征向量，进行非线性变换，得到一个8维度的新的特征向量。

看一下细化之后的，`self.WN[0]`的网络示意图：

<div align=center>
    <img src="zh-cn/img/ch7/08/p18.jpg" /> 
</div>

一个WaveNet网络里面，有8个`in_layers`（其中有“膨胀卷积”）层，以及8个`res.skip.layers`（普通“卷积网络”）层。上图给出的是从forward函数接受输入，到使用了0-th的`in.layers[0]`以及`res.skip.layers[0]`之后的结果。

这里有个比较特殊的函数：`fused_add_tanh_sigmoid_multiply`
其负责的是，语音数据和梅尔谱数据之间的fusion-融合。融合之后，会经历一次卷积，即使用`res.skip.layers[0]`进行一次2倍化扩展，扩展之后，再次对半切分，前半部分会element-wise加到最初的audio上面，得到新的audio，其shape为(12, 256, 2000)。另外，后半部分会element-wise加到output张量上面。output是全0的，shape为(12, 256, 2000)的张量。

下面看这个特殊的函数：`fused_add_tanh_sigmoid_multiply`的图示化细节：

<div align=center>
    <img src="zh-cn/img/ch7/08/p19.jpg" /> 
</div>

上面的浅黄色块（下方），即是这个特殊函数的具体操作方法：先经历一次element-wise加，然后对半截取（按照feature 维度，从512，对半为两个256）。截取之后，分别经过tanh（其取值范围为`[-1, 1]`）和sigmoid（其取值范围为`[0,1]`），然后再经历一次element-wise的乘积。即得到融合后的张量。

在`i=1`的时候，会把`in_layers[1]`中的dilation设置为2，然后padding也设置为2。如下图所示：

<div align=center>
    <img src="zh-cn/img/ch7/08/p20.jpg" /> 
</div>

`k=0, n_half=4, n_mel_channels*n_group=640, WN_config={'n_layers': 8, 'n_channels': 256, 'kernel_size': 3}`

依此类推，8个`in_layers`的卷积的定义类似于：

<div align=center>
    <img src="zh-cn/img/ch7/08/p21.jpg" /> 
</div>

以及，`res.skip.layers`的8个网络（翻倍channel numbers；注意i=7的时候，是从256到256 channels）类似于：

<div align=center>
    <img src="zh-cn/img/ch7/08/p22.jpg" /> 
</div>

最后是`i=6`到`i=7`的时候，以及`i=7`的全过程：

<div align=center>
    <img src="zh-cn/img/ch7/08/p23.jpg" /> 
</div>

特别的，当`i=7`的时候，`res.skip.layers[7]`是从256 channels到256 channels。输出结果直接叠加到output张量。之后会走一个self.end，conv1d的卷积，得到的是（12, 8, 2000）的张量。这个张量，就是之前介绍过的下图中的output:

<div align=center>
    <img src="zh-cn/img/ch7/08/p24.jpg" /> 
</div>

前四个wavenet，`k=0,1,2,3`都是上面的架构。

到了`k=4,5,6,7`的时候，即中间四个WaveNet，输入的audio方面都修改为了shape为(12, 3, 2000)的张量，相应的在wavnet中，只有`self.start`被修改为从3映射到256 channels的卷积，以及`self.end`是从256映射到6 channels的卷积。其他网络结构保存不变。即，8个`in.layers`，和8个`res.skip.layers`的架构保存一样。

再后来，到了`k=8,9,10,11`的时候，即最后四个WaveNet，输入的audio方面都修改为了shape为(12, 2, 2000)的张量，相应的在wavnet中，只有`self.start`被修改为从2映射到256 channels的卷积，以及self.end是从256映射到4 channels的卷积。其他网络结构保存不变。即，8个in.layers，和8个`res.skip.layers`的架构保存一样。

#### 5.【inference视角】从梅尔谱和高斯噪声生成wave

前面两个部分分别介绍了训练视角的整体waveglow和分块wavenet的网络结构和代码。本部分详细介绍inference的代码分析，即输入的是来自tacotron2得到的梅尔谱，以及高斯噪音，输出的是audio wav文件。

核心思想是对前面的`k=0`到`k=11`个blocks的反向使用，即先经过`k=11`，然后逐步`k-=1`，最后到`k=0`，再经过若干变换，得到最后的audio，保存成`.wav`文件。

<div align=center>
    <img src="zh-cn/img/ch7/08/p25.jpg" /> 
</div>

上图给出了，从读取一个梅尔谱文件`（.pt文件）`到分别调用`k=11`的两个子网络：`self.WN[k=11]`，以及`self.convinv[k=11]`（注意这里的`1*1`卷积是`reverse=True`的设置！)。

在类`mel2samp.py`中，有关于如何从wav生成梅尔谱文件`.pt`的方法，这个文件有main函数，可以直接执行。

```python
audio, sr = load_wav_to_torch(filepath) # 读取.wav文件
melspectrogram = mel2samp.get_mel(audio) 
# 调用stft->mel-spec的库方法，得到梅尔谱

filename = os.path.basename(filepath)
new_filepath = args.output_dir + '/' + filename + '.pt' # 输出.pt文件名
print(new_filepath)
torch.save(melspectrogram, new_filepath) # 保存文件
```

这其中的`mel2samp.get_mel`方法如下：

```python
def get_mel(self, audio): 
    # from wav file to mel-spectrogram, 从wav文件到梅尔谱
    print('get_mel, audio.shape={}'.format(audio.shape)) 
    # [16000] (不到一秒钟吗?) 如果是22050的话，一秒应该是22050个点！
    # 和采样率没有直接关系；就是拿出来16000个点

    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = self.stft.mel_spectrogram(audio_norm) # 库方法
    melspec = torch.squeeze(melspec, 0)
    print('get_mel, melspec.shape={}\n----\n'.format(melspec.shape)) 
    # [80, 63]
    return melspec
```

更进一步，`self.stft`，来自TacotronSTFT类的方法，其初始化为：

```python
self.stft = TacotronSTFT(filter_length=filter_length, 
                         # 1024, frame length
                         hop_length=hop_length, # 256, jump length
                         win_length=win_length, # 1024, 
                         #window-size for hann window
                         sampling_rate=sampling_rate, # 22050
                         mel_fmin=mel_fmin, mel_fmax=mel_fmax) 
                         # 0.0, to, 8000 (Hz) frequency
```

然后就是一顿操作猛如虎了，例如上采样，之后去掉最后的768个元素，再进行unfold, permute, congituous, view, permute一系列花样，得到是(1, 640, 25472)这样的梅尔谱表示张量。

根据25472，我们可以初始化高维高斯噪音张量了，大小为(1, 4, 25472）。然后就是之前的熟悉的k=0到11的12个blocks的反向使用。这里从k=11开始，逐步经过wavenet，以及`1*1`的卷积的反向使用。最后得到的是张量audio，形状为(1, 4, 25472)。

这里面有意思的是，`log_s`和`b`的切割方法，其实和正向forward是一样的！多么神奇啊！可以参考下图（的F和H）加深理解：

<div align=center>
    <img src="zh-cn/img/ch7/08/p26.jpg" /> 
    <p>https://zhuanlan.zhihu.com/p/351479696</p>
</div>

特别的，当`k=8`的末尾的时候，会在audio的左边追加一个高斯噪音，shape为(1, 2, 25472)。得到新的`audio=torch.cat((z, audio), 1) -> (1, 6, 25472)`。然后继续经历，`k=7，6，5，4`。

当`k=4`的末尾的时候，会在audio的左边追加一个高斯噪音，shape为(1, 2, 25472)。得到新的`audio=torch.cat((z, audio), 1) -> (1, 8, 25472)`。然后继续经历，`k=3,2,1,0`。

最后经历了`k=11 to 0`之后，有下图:

<div align=center>
    <img src="zh-cn/img/ch7/08/p27.jpg" /> 
</div>

到此为止，就是把inference的算法框架介绍完毕了。

> 关于gpu, `apex=16`方面的相关知识，可以参考：<https://zhuanlan.zhihu.com/p/343891349>
> 
> 本论文解读参考来源于 知乎 迷途小书童 大佬的: <https://zhuanlan.zhihu.com/p/355219393>

------
