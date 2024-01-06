## 语音合成开源项目和数据集



### 1.GitHub开源的一些TTS项目


[1]. <https://github.com/NVIDIA/NeMo>

[1]. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/models.html#spectrogram-enhancer>

[2]. <https://github.com/mozilla/TTS>

[3]. <https://github.com/coqui-ai/TTS>

[4]. <https://github.com/TensorSpeech/TensorFlowTTS>

[5]. <https://github.com/sp-nitech/DNN-HSMM>

[6]. <https://github.com/dectalk/dectalk>

[7]. <https://github.com/tts-tutorial>

[8]. <https://github.com/as-ideas/TransformerTTS>

[9]. <https://github.com/k2-fsa/next-gen-kaldi-wechat/blob/master/2023-10-24-tts-runtime.md>

[10]. <https://github.com/wenet-e2e/speech-synthesis-paper>

[11]. <https://github.com/wenet-e2e/wetts>

[12]. [一文了解语音合成经典论文，架构及最新语音合成调研【20231213更新版】](https://blog.csdn.net/weixin_44649780/article/details/129402750)

[13]. <https://yqli.tech/page/tts_paper.html>



<!-- https://mp.weixin.qq.com/s/GHkc1DN4Dozzz50Fq8m3kQ -->



### 2.GitHub开源的一些语音克隆的项目

[1]. <https://github.com/suno-ai/bark>

[2]. <https://github.com/babysor/MockingBird>

[3]. <https://github.com/KevinWang676/Bark-Voice-Cloning>

[4]. <https://github.com/CorentinJ/Real-Time-Voice-Cloning>

[5]. <https://zhuanlan.zhihu.com/p/112627134>

[6]. <https://mp.weixin.qq.com/s/m5JY0bpcJYTR4xp_c7V5gg>

[7]. <https://github.com/Stardust-minus/Bert-VITS2>

[8]. <http://mp.weixin.qq.com/s?__biz=MzkwMzMzNTgzNQ==&mid=2247484509&idx=1&sn=1ddf2e5e74e6bc81fbbf42a5cbb919ab&chksm=c0969ea9f7e117bfe8a0b6fbaa868fdbc28ee574fa690ed2387d7124039505d9a62250cd5e26&mpshare=1&scene=1&srcid=1110ejmMWuV30Qhc9nmhSvEz&sharer_shareinfo=e042601aeafdc2dd754169116324346c&sharer_shareinfo_first=e042601aeafdc2dd754169116324346c#rd>

[9]. <https://github.com/netease-youdao/EmotiVoice>

[10]. <https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable>

[11]. <https://github.com/justinjohn0306/so-vits-svc-4.0>

[12]. facebookresearch/audiocraft: https://github.com/facebookresearch/audiocraft

[13]. w-okada/voice-changer: https://github.com/w-okada/voice-changer

[14]. espnet/espnet: https://github.com/espnet/espnet

[15]. babysor/MockingBird: https://github.com/babysor/MockingBird

[16]. CorentinJ/Real-Time-Voice-Cloning: https://github.com/CorentinJ/Real-Time-Voice-Cloning

[17]. neonbjb/tortoise-tts: https://github.com/neonbjb/tortoise-tts

[18]. https://mp.weixin.qq.com/s/OyxSSUPzGZWBLgCfQAI2wg


### 3.TTS常用开源数据集

!> http://yqli.tech/page/data.html

<!-- 6.语音合成开源数据
Nancy
TWEB
Spanish -->

+ TTS mandarin

|	|数据|	描述|	链接|
|---|----|------|-------|
|1	|baker标贝女声|	12小时|	<https://www.data-baker.com/open_source.html>|
|2	|Aishell-3|	85小时88035句多说话人数据	|<http://www.aishelltech.com/aishell_3>|
|3	|DiDiSpeech|	500人60小时	|<https://outreach.didichuxing.com/research/opendata/>|
|4	|OpenSLR|	提供各种语言的合成、识别等语料	|<https://www.openslr.org/resources.php>|
|5	|zhvoice|	3200说话人900小时，用于声音复刻，合成，识别等	|<https://github.com/fighting41love/zhvoice>|


+ TTS english


|	|数据|	描述|	链接|
|---|----|------|-------|
|1	|LibriTTS|	multispeakers,大约585小时|	<https://www.openslr.org/60/>|
|2	|LJ Speech|	大约24小时|	<https://keithito.com/LJ-Speech-Dataset/>|
|3	|VCTK|	109发音人，每人400句|	<https://datashare.is.ed.ac.uk/handle/10283/2651>|
|4	|OpenSLR|	提供各种语言的合成、识别等语料|	<https://www.openslr.org/resources.php>|
|5	|HiFi-TTS|	291.6小时，10发音人|	<http://www.openslr.org/109/>|
|6	|open speech corpora|	各类数据搜集|	<https://github.com/coqui-ai/open-speech-corpora>|
|7	|RyanSpeech|	10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation10小时conversation|	<http://mohammadmahoor.com/ryanspeech/>|

+ TTS emotion

|	|数据|	描述|	链接|
|---|----|------|-------|
|1	|ESD|	10位英语和10位中文发音人5种情感，主要应用VC,TTS|	<https://hltsingapore.github.io/ESD/>|
|2	|IEMOCAP|	12小时音视频情感|	<https://sail.usc.edu/iemocap/iemocap_release.htm>|
|3	|EmoV_DB|	english and french 5种情感|	<https://www.openslr.org/115/>|
|4	|Thorsten Müller|	single german speaker dataset (Neutral, Disgusted, Angry, Amused, Surprised, Sleepy, Drunk, Whispering) 175分钟|	<https://www.openslr.org/110/>|
|5	|TAL_SER|	4541条语音，总时长12.5小时，愉悦度和激情度两个维度。|	<https://ai.100tal.com/dataset>|

+ TTS dialect

|	|数据|	描述|	链接|
|---|----|------|-------|
|1	|RuSLAN|	31小时高质量俄语|	<https://ruslan-corpus.github.io/>|
|2	|M-AILABS|	1000小时，German,English,Spanish,Italian,Ukrainian,Russsian,French,Polish|	<https://www.caito.de/?p=242>|
|3	|OpenSLR|	提供各种语言的合成、识别等语料|	<https://www.openslr.org/resources.php>|
|4	|css10|	greek,spanish,finish,french,hungarian,japanese,dutch,russian,chinese数据	|<https://github.com/Kyubyong/css10>|


+ TTS frontend

| |数据|	描述|	链接|
|---|----|------|-------|
|1	|polyphone|	14 top多音字|	<https://drive.google.com/drive/folders/1ncEnpttZNxmNMXsQSmytgrK1_2wKujkX>|




