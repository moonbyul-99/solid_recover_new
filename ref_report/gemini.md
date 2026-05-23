单细胞多组学多样本整合与批次效应矫正深度调研报告
在单细胞多组学（Single-cell Multi-omics）与多样本联合分析中，由于测序平台、试剂批次、样本来源以及实验操作人员等非生物学技术因素引入的批次效应（Batch Effect），会严重遮蔽真实的生物学变异。高效去除批次效应，同时最大程度保留细胞异质性、细胞状态、发育轨迹等生物学变异，是构建大规模单细胞图谱（Single-cell Atlas）的前置核心任务。
本报告聚焦于单细胞多样本/多批次整合算法，广泛收集并深入梳理当前主流的技术路径，透彻剖析其底层的数学原理、公式推导、损失函数设计以及核心代码实现逻辑，辅以 scIB（Single-cell Integration Benchmarking）基准测试度量体系对比，为构建鲁棒、高效的单细胞整合工作流提供详实的方法学参考。
1. 单细胞多样本/多组学整合的核心挑战与技术演进
单细胞测序技术的飞速发展使得研究人员能够同时获取基因表达、染色质可及性、蛋白质丰度等多维度的信息。然而，多组学数据的联合分析面临着高度的技术异质性与复杂的批次混淆风险。
1.1 批次效应的来源与混淆机制
批次效应是指在不同时间、不同实验室、不同操作协议下收集的单细胞测序数据中，由于技术噪声引发的数据分布偏移。更具挑战性的是，当细胞类型分布在不同批次间高度相关（即某些细胞类型仅存在于特定批次中）时，技术批次效应与真实的生物学变异会产生深度混淆。若直接应用常规的全局对齐算法，极易将批次特异性的独特细胞类型强行混淆，造成过矫正（Over-correction），损害生物学异质性；而矫正不足（Under-correction）则会导致下游聚类和轨迹推断被技术批次所主导。
1.2 多组学整合的技术路径分类
针对单细胞多组学数据的整合任务，依据输入数据的结构和组学配对情况，主要分为三大集成路径：
非配对整合（Unpaired Integration）：处理来自同一组织但由不同细胞测得的非配对组学数据（如独立的 scRNA-seq 和 scATAC-seq 样本）。典型算法如 UnionCom 和 MMD-MA，通过匹配细胞的低维空间距离矩阵或特征分布实现对齐；而 scDART 和 scJoint 则分别引入基因活性函数或半监督学习框架来构建统一的潜空间。
完全配对整合（Paired Integration）：处理在同一个细胞内同时测定多种组学模态的数据（如 10X Multiome 同时测定 RNA 与 ATAC，或 CITE-seq 同时测定 RNA 与 ADT 表面蛋白）。代表性方法如 MOFA+、scAI 和 scMVP，利用矩阵分解或多通道变分自编码器（VAE）提取跨组学共享的低维表征。
配对引导与马赛克整合（Paired-guided & Mosaic Integration）：处理包含部分配对及部分单模态的数据集，利用配对细胞作为桥梁（Anchors）来对齐非配对的细胞。如 MultiVI 和 Cobolt，通过构建多模态联合潜空间实现信息互补。
1.3 批次矫正的博弈与控制：RBET 评估体系
为了在“批次消除”与“生物保留”之间寻求最佳平衡，方法学评估需要引入能够量化过矫正风险的基准工具。基于参考基因的批次效应测试（Reference-informed Batch Effect Testing, RBET）提供了一种鲁棒的指导方案。其底层机理在于，管家基因（Housekeeping Genes，如核糖体基因 RPS17、RPS18）在各种细胞类型和实验条件下均呈高水平、稳定的均一表达，几乎不受技术丢手（Dropout）事件的影响。完美的批次矫正应当精确对齐各批次间管家基因的平均表达水平（技术批次消除），而不应以抹杀批次内管家基因原有的表达方差为代价。RBET 通过在大样本数据中监控管家基因在矫正前后的均值与方差演变，能够敏锐检测出因算法正则化过强导致的过矫正行为，为不同整合算法的合理选型提供客观参考。
2. 概率生成模型与多模态似然函数设计
条件变分自编码器（Conditional Variational Autoencoder, cVAE）是当前单细胞数据降维与批次矫正的黄金标准之一。通过引入批次标签作为条件协变量，cVAE 能够迫使编码器在隐空间剥离批次偏差，同时解码器在重建时将批次信息还原。
2.1 条件变分下界（cELBO）的数学推导
设 $x_n$ 为第 $n$ 个细胞的观测多维分子计数向量，$s_n$ 为其对应的可观测批次（或系统）协变量标签。生成模型假设 $x_n$ 的生成过程依赖于低维连续隐变量 $z_n \in \mathbb{R}^d$ 且受到条件 $s_n$ 的调制。其联合条件概率分布写为 $p_\theta(x_n, z_n | s_n) = p_\theta(x_n | z_n, s_n) p(z_n | s_n)$ 。
由于真实的条件后验分布 $p_\theta(z_n | x_n, s_n)$ 无法直接积分解出，引入由参数 $\phi$ 控制的变分分布 $q_\phi(z_n | x_n, s_n)$ 进行变分推断。条件对数边际似然的推导过程如下：
$$\log p_\theta(x_n | s_n) = \log \int p_\theta(x_n, z_n | s_n) d z_n$$
$$\log p_\theta(x_n | s_n) = \log \int q_\phi(z_n | x_n, s_n) \frac{p_\theta(x_n, z_n | s_n)}{q_\phi(z_n | x_n, s_n)} d z_n$$
根据詹森不等式，将对数算子移至积分号内部，得到其证据下界（ELBO）：
$$\log p_\theta(x_n | s_n) \ge \mathbb{E}_{z_n \sim q_\phi(z_n | x_n, s_n)} \left[ \log \frac{p_\theta(x_n, z_n | s_n)}{q_\phi(z_n | x_n, s_n)} \right]$$
展开联合概率分布，重写为条件重构误差与 KL 散度约束两部分：
$$\mathcal{L}_{\text{cELBO}}(\theta, \phi; x_n, s_n) = \mathbb{E}_{z_n \sim q_\phi(z_n | x_n, s_n)} \left[ \log p_\theta(x_n | z_n, s_n) \right] - D_{\text{KL}}\left( q_\phi(z_n | x_n, s_n) \parallel p(z_n | s_n) \right)$$
模型优化的目标即为最大化该下界，或等价于最小化 $-\mathcal{L}_{\text{cELBO}}$ 。
2.2 针对单细胞稀疏性的似然分布选择
在单细胞转录组学（scRNA-seq）中，UMI 计数数据高度离散且包含大量的零值。通常采用负二项分布（Negative Binomial, NB）或零膨胀负二项分布（Zero-Inflated Negative Binomial, ZINB）作为重构似然 $p_\theta(x_n | z_n, s_n)$ 的底层假设。
对于 NB 分布，给定细胞 $n$ 的基因 $g$ 的均值 $\mu_{ng}$ 与基因特异性逆离散参数（Inverse Dispersion）$\theta_g$，其观测计数 $x_{ng}$ 的概率密度定义为：
$$p(x_{ng} | \mu_{ng}, \theta_g) = \frac{\Gamma(x_{ng} + \theta_g)}{\Gamma(x_{ng} + 1) \Gamma(\theta_g)} \left( \frac{\theta_g}{\theta_g + \mu_{ng}} \right)^{\theta_g} \left( \frac{\mu_{ng}}{\theta_g + \mu_{ng}} \right)^{x_{ng}}$$
在神经网络参数化中，解码器输出相对基因表达比例 $\rho_{ng}$（满足 $\sum_g \rho_{ng} = 1$），结合细胞文库大小 $l_n$，使得 $\mu_{ng} = l_n \rho_{ng}$ 。
若采用 ZINB 分布，则引入一个额外的零膨胀系数 $\pi_{ng} \in (0, 1)$ 来专门捕获技术性丢手带来的零值：
$$p(x_{ng} | \mu_{ng}, \theta_g, \pi_{ng}) = \pi_{ng} \delta_0(x_{ng}) + (1 - \pi_{ng}) \text{NB}(x_{ng} | \mu_{ng}, \theta_g)$$
其中 $\delta_0(y)$ 为狄拉克 $\delta$ 函数，在 $y=0$ 时取值 1，其余情况取值 0。
2.3 totalVI 的联合生成模型设计
totalVI 模型用于同时分析 CITE-seq 数据中的单细胞 RNA 与表面蛋白 ADT 丰度。其隐变量建模和条件生成过程如下：
隐空间先验：每个细胞独立抽取隐变量 $z_n \sim \mathcal{N}(0, I)$ 。
转录组建模：解码器神经网络输出 RNA 的尺度参数 $\rho_n = f_\rho(z_n, s_n)$ 。文库大小 $l_n$ 建模为对数正态随机变量 $l_n \sim \text{LogNormal}(l_\mu^T s_n, l_\sigma^2 T s_n)$，其中均值和方差从对应批次的实测对数文库大小中经验计算。RNA 观测 UMI 计数服从 $\text{NB}(l_n \rho_n, \theta_G)$ 分布，其中 $\theta_G$ 为基因专属逆离散参数。
表面蛋白质建模：由于蛋白质数据存在高度的非特异性抗体背景绑定，totalVI 引入了背景与前景的二组分混合高斯/负二项模型。解码器网络预测蛋白质的前景相对丰度比例 $\alpha_n = g_\alpha(z_n, s_n)$、背景比例参数 $\beta_n$ 以及当前细胞中该抗体表现为技术背景噪声的概率 $\pi_n = h_\pi(z_n, s_n)$ ：
$$f_\rho(z_n, s_n) : \mathbb{R}^d \times \{0, 1\}^K \to \Delta^{G-1}$$
$$g_\alpha(z_n, s_n) : \mathbb{R}^d \times {0, 1}^K \to。
对于单细胞 Multiome（同时测量 RNA 和染色质可及性 ATAC）的整合，liam 模型同样采用了多模态联合似然，其总损失定义为：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rna}} + \mathcal{L}_{\text{atac}} + \mathcal{L}_{\text{KL}} + \alpha \mathcal{L}_{\text{adv}}$$
其中 RNA 与蛋白质模态计算负二项似然损失 $\mathcal{L}_{\text{rna}}$，而高维稀疏的 ATAC 信号重构则采用负多项分布（Negative Multinomial）似然损失 $\mathcal{L}_{\text{atac}}$ 。
3. 强批次效应矫正的多样化技术路径与数学推导
在面对跨物种、跨测序协议（单细胞 scRNA-seq vs 单细胞核 snRNA-seq）等强批次效应场景时，常规的 cVAE 仅靠改变条件变量容易造成对齐不足；若一味增强 KL 正则化强度，则会无差别抹杀细胞类型特异性（导致隐空间崩溃）。因此，研究人员开发了多种新型正则化约束手段。
3.1 sysVI：VampPrior 与循环一致性约束
跨系统变分推断（sysVI）引入了可变分后验混合先验（VampPrior）和隐空间循环一致性（Cycle-Consistency），在实现强批次对齐的同时保护极精细的细胞变异。
3.1.1 变分后验混合先验（VampPrior）
在传统 VAE 中，先验分布 $p(z)$ 设定为各向同性的标准高斯分布 $\mathcal{N}(0, I)$，这构成了对隐空间极强的单峰平滑约束，难以保留多细胞类型的多峰异质性。VampPrior 将先验定义为一组可学习的伪输入（Pseudo-inputs）在变分后验上的混合分布：
$$p_\lambda(z) = \frac{1}{K} \sum_{k=1}^K q_\phi(z | u_k)$$
其中 $u_k \in \mathbb{R}^D$ 是由模型在原始输入空间中自主学习得到的 $K$ 个代表性细胞参数（伪输入）。当我们在对数边际似然变分下界中代入该先验并对全体训练集求期望时，KL 项可展开为：
$$- \mathbb{E}_{q(x)} = \mathbb{E}_{q(x)}[H[q_\phi(z|x)]] + \mathbb{E}_{q(z)}[\log p_\lambda(z)]$$
其中，第一项维持了单细胞变分后验的熵（保持隐表征的发散性），第二项促使累积后验分布（Aggregated Posterior）$q(z) = \frac{1}{N} \sum_{n=1}^N q_\phi(z | x_n)$ 与 VampPrior 先验实现高度匹配。由于伪输入数量 $K$ 远小于实际细胞数 $N$，VampPrior 构成了一种高度紧凑、可表达多峰形态且与数据紧密耦合的动态边界，消除了标准高斯先验对潜在生物信号的强行钝化作用。
3.1.2 隐空间循环一致性损失
为促使不同生物系统的同类细胞实现高精度重合，sysVI 避免使用容易混淆无关细胞的对抗判别器，转而对单细胞实施跨批次“转译”。
设有属于源系统 $i$ 的细胞 $x_i$，其隐表征为 $z_i \sim q_\phi(z | x_i, s_i)$ 。现在将其与目标系统标签 $s_j$ 配对送入解码器，生成该细胞在目标系统下的虚拟转译谱 $x'_j = \text{Decoder}(z_i, s_j)$ 。接着，将该虚拟表达谱再次送入编码器，提取目标系统下的潜表征 $z'_j \sim q_\phi(z | x'_j, s_j)$ 。
由于转译过程仅改变了技术背景，两者的生物学本体应该完全恒等。模型通过惩罚 $z_i$ 与 $z'_j$ 之间的距离来确保这一特性：
$$\mathcal{L}_{\text{CYC}} = \mathbb{E}_{x_i \sim p_{\text{data}}(x)} \left[ \left\| \frac{z_i - \mu_z}{\sigma_z} - \frac{z'_j - \mu_z}{\sigma_z} \right\|_2^2 \right]$$
通过这种两阶段的循环约束，细胞仅在不改变生物学特性的前提下进行跨系统的对齐，极大地降低了过矫正的风险。该损失可通过调节超参数 z_distance_cycle_weight 控制，推荐的初始优化范围通常在 2 至 10 之间。
3.2 基于梯度反向传播层（GRL）的领域对抗训练
另一种去批次的技术路径是通过引入对抗训练来主动消除隐空间中的批次特征。
在 scGLUE 及其变体中，网络除编码器 $q_\phi$ 外，额外配置了一个批次判别器（Domain Discriminator）$D_\psi$ 。判别器的输入为隐表征 $z$，输出为该细胞所属批次的概率分布，优化目标为最小化多分类交叉熵：
$$\mathcal{L}_{\text{ADV}}(\phi, \psi) = \mathbb{E}_{x \sim p_{\text{data}}} \left$$
编码器（特征提取器）的优化目标则是最大化该分类损失，从而彻底混淆不同批次的表征，构成极小极大（Minimax）对立博弈。
为在单次反向传播中同步更新双方，在编码器输出端和判别器输入端之间嵌入梯度反向传播层（GRL）。GRL 在前向传播时作为恒等映射，不改变数值流量；在反向传播时，截断并将判别器的梯度乘上负系数 $-\alpha$ ($\alpha > 0$) 传回编码器：
$$\text{Forward: } R_\alpha(z) = z$$
$$\text{Backward: } \frac{d R_\alpha(z)}{d z} = -\alpha \mathbf{I}$$
这种对抗对齐虽去批次极其强力，但易发生模式坍塌，混淆比例分布不同的独特细胞。
3.3 scDML 与 scCRAFT：三元组损失与拓扑保留
以锚点为导向的方法通过拉近跨批次的互近邻（MNN）细胞对来消除批次效应。
3.3.1 scDML：互近邻与度量学习
scDML 首先通过在不同批次间识别 MNN 作为对齐锚点对，同时在各批次内部执行高分辨率的层次聚类（Hierarchical Clustering）。基于细胞聚类分支和 MNN 配对关系，构建三元组 $\mathcal{T} = \{(z_A, z_P, z_N)\}$ 。在隐空间中最小化三元组损失（Triplet Loss），以拉近锚点与正样本（同类型跨批次细胞或配对细胞）的距离，推远与负样本（异类细胞）的距离：
$$\mathcal{L}_{\text{triplet}} = \sum_{(z_A, z_P, z_N) \in \mathcal{T}} \max\left(0, d(z_A, z_P)^2 - d(z_A, z_N)^2 + m\right)$$
其中 $d(u, v) = \|u - v\|_2^2$，而 $m > 0$ 表示人为设定的边界阈值（Margin）。
3.3.2 scCRAFT：双分辨率拓扑约束
与高度依赖跨批次 MNN 寻找品质的 scDML 不同，scCRAFT 提出了一种无需锚点的“双分辨率三元组损失”机制。该算法在各批次内部独立进行双分辨率聚类：
低分辨率聚类（Low-resolution Clustering）：将细胞划分为宽谱系的粗大聚类簇，若两个细胞处于不同的低分辨率簇中，表明其绝对不可能属于同一种生物学细胞类型。
高分辨率聚类（High-resolution Clustering）：精细划分局部亚型。
在构建三元组时，锚点 $A$ 与正样本 $P$ 必须来自同一高分辨率聚类簇，而负样本 $N$ 则必须取自不同的低分辨率聚类簇。最关键的是，三元组的采样与损失计算完全限制在各自的批次内部进行。结合全局的对抗性判别器去除跨批次宏观漂移，此批次内拓扑保留机制能够对冲对抗训练带来的过矫正副作用，精准锁定局部稀有细胞谱系不被混淆。
3.4 SCALEX：非对称 VAE 与领域特定批归一化（DSBN）
在构建海量单细胞图谱时，全局重构或 MNN 对检索极易引发计算复杂度的指数暴涨。SCALEX 通过解耦非对称 VAE 的编码与解码，实现了极佳的计算可扩展性。
3.4.1 非对称 VAE 架构
SCALEX 移除了传统 cVAE 中在编码器输入端串联批次标签的做法，构建了一个“批次无关”（Batch-free）的通用编码器，强迫其仅依据基因表达的协同作用提取本质的、 batch-invariant 的细胞潜表征 $z$ 。相对应地，将全部批次信息置于解码器端，并在解码器中引入领域特定批归一化（Domain-Specific Batch Normalization, DSBN）来重构技术噪声。
3.4.2 DSBN 数学机理
领域特定批归一化是无监督领域适应（Domain Adaptation）中的核心技术，旨在解决由于样本来源不同带来的 internal covariate shift 问题。
对于常规的 Batch Normalization (BN) 层，其对输入的 mini-batch 数据 $x$ 统一进行归一化：
$$\text{BN}(x_i) = \gamma \left( \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \right) + \beta$$
其中 $\mu_{\mathcal{B}}$ 与 $\sigma_{\mathcal{B}}^2$ 分别是该批数据的统计均值和方差。而在 DSBN 中，针对属于不同批次（域）的数据建立多个平行的 BN 分支，它们共享主干权重，但维护各自专属的缩放参数 $\gamma_b$ 和偏移参数 $\beta_b$ 。对于来自批次 $b$ 的细胞 $x_i$，其在前向计算时的标准化映射为：
$$\text{DSBN}(x_i; b) = \gamma_b \left( \frac{x_i - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}} \right) + \beta_b$$
其中 $\mu_b$ 和 $\sigma_b^2$ 是由属于同一批次 $b$ 的样本单独计算出的局部统计量。通过将所有特定批次的技术偏好隔离在 DSBN 专属分支的 $\gamma_b$ 和 $\beta_b$ 中，网络实现了技术差异的精准解耦，使得中间特征空间保持高度的域不敏感性。
4. 核心算法在 PyTorch 中的工程实现与源码级逻辑
本节提供核心算法算子的工业级 PyTorch 源码实现逻辑，深入解析工程落地中的技术细节。
4.1 梯度反向传播层（GRL）的自定义 Autograd 实现
在 PyTorch 中，实现 GRL 必须通过继承 torch.autograd.Function 来重写反向传播梯度流。
Python
import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 缓存当前的对抗调节系数 alpha
        ctx.alpha = alpha
        # 前向传播为恒等映射，为了使 PyTorch 的 Autograd 引擎注册该操作，
        # 采用 view_as() 保证张量在不拷贝物理内存的前提下被引擎记录
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 提取梯度并乘以负对抗因子 -alpha 传回编码器
        grad_input = None
        if ctx.needs_input_grad:
            grad_input = -ctx.alpha * grad_output
        # 因为前向有两个输入 (x, alpha)，反向需要对应返回两个梯度值，alpha 不需要梯度
        return grad_input, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


4.2 领域特定批归一化（DSBN）模块的 PyTorch 实现
该模块需要根据当前样本的批次编号，动态将张量分流到对应的归一化分支。
Python
class DomainSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_batches, eps=1e-5, momentum=0.1):
        super(DomainSpecificBatchNorm1d, self).__init__()
        self.num_batches = num_batches
        # 为每一个批次单独分配一个标准的 BatchNorm1d 算子，动态管理
        self.bns = nn.ModuleList()

    def forward(self, x, batch_indices):
        """
        x: 形如 (Batch_Size, num_features) 的隐藏层激活输出
        batch_indices: 一维张量 (Batch_Size,)，包含当前细胞对应的批次 ID
        """
        out = torch.zeros_like(x)
        for b in range(self.num_batches):
            mask = (batch_indices == b)
            if mask.sum() > 0:
                # 对当前批次的细胞分片送入专属 BN
                out[mask] = self.bns[b](x[mask])
        return out


4.3 VampPrior 损失计算组件的 PyTorch 实现
本组件实现可学习伪输入的管理，并使用 Log-Sum-Exp 技巧稳定计算混合后验分布。
Python
class VampPriorLoss(nn.Module):
    def __init__(self, latent_dim, num_pseudo_inputs, input_dim):
        super(VampPriorLoss, self).__init__()
        self.latent_dim = latent_dim
        self.num_pseudo_inputs = num_pseudo_inputs
        
        # 注册可训练的虚拟伪输入，维度与编码器输入的基因数相同
        self.pseudo_inputs = nn.Parameter(
            torch.randn(num_pseudo_inputs, input_dim) * 0.01
        )

    def forward(self, encoder, z, q_mu, q_logvar):
        """
        encoder: 编码器网络，接受表达谱返回其均值与方差
        z: 隐空间采样值，形状 (batch_size, latent_dim)
        q_mu, q_logvar: 变分后验均值与方差，形状 (batch_size, latent_dim)
        """
        batch_size = z.size(0)
        
        # 1. 计算伪输入的变分后验参数，用于构建 VampPrior 的混合组分
        p_mu, p_logvar = encoder(self.pseudo_inputs)  # (num_pseudo_inputs, latent_dim)
        
        # 2. 计算混合概率对数 log p_Vamp(z)
        # 采用维度扩展以利用广播机制计算两两之间的平方距离
        z_expand = z.unsqueeze(1)                     # (batch_size, 1, latent_dim)
        p_mu_expand = p_mu.unsqueeze(0)               # (1, num_pseudo_inputs, latent_dim)
        p_logvar_expand = p_logvar.unsqueeze(0)       # (1, num_pseudo_inputs, latent_dim)
        
        # 多维高斯密度对数：-0.5 * [log(2*pi) + logvar + (z - mu)^2 / var]
        log_gaussian = -0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=z.device)) + 
            p_logvar_expand + 
            (z_expand - p_mu_expand) ** 2 / (torch.exp(p_logvar_expand) + 1e-8)
        ) # (batch_size, num_pseudo_inputs, latent_dim)
        
        log_components = log_gaussian.sum(dim=-1)     # (batch_size, num_pseudo_inputs)
        
        # 使用 logsumexp 将混合概率对数进行稳定求和
        log_p_z = torch.logsumexp(log_components, dim=1) - torch.log(
            torch.tensor(self.num_pseudo_inputs, dtype=torch.float, device=z.device)
        ) # (batch_size,)
        
        # 3. 计算后验熵 H[q_\phi(z|x)]
        entropy_q = 0.5 * (1.0 + q_logvar + torch.log(torch.tensor(2 * torch.pi, device=z.device))).sum(dim=-1)
        
        # 4. 变分下界正则化项：DKL = - H(q) - E_q[log p(z)]
        kl_div = -entropy_q - log_p_z
        return kl_div.mean()


4.4 隐空间循环一致性损失（Cycle-Consistency Loss）工作流实现
工程上需精细管理“编码 $\to$ 跨条件转译 $\to$ 再编码”的整个信息环路。
Python
class CycleConsistencyWorkflow(nn.Module):
    def __init__(self, encoder, decoder):
        super(CycleConsistencyWorkflow, self).__init__()
        self.encoder = encoder  # 输入 (x, batch_id) -> (z, mu, logvar)
        self.decoder = decoder  # 输入 (z, batch_id) -> 重构的 x_mean

    def forward(self, x_i, s_i, s_j):
        """
        x_i: 源批次真实的单细胞表达量，形状 (B, input_dim)
        s_i: 源批次 One-Hot 标签, 形状 (B, num_batches)
        s_j: 目标对齐批次 One-Hot 标签, 形状 (B, num_batches)
        """
        # Step 1: 提取源系统细胞在隐空间的均值 z_i (代表其生物本体特征)
        z_i, _, _ = self.encoder(x_i, s_i)
        
        # Step 2: 改变批次标签，在目标批次下转译生成虚拟谱 x_prime_j
        x_prime_j = self.decoder(z_i, s_j)
        
        # Step 3: 对该虚拟细胞在目标批次下重新提取其低维特征 z_prime_j
        z_prime_j, _, _ = self.encoder(x_prime_j, s_j)
        
        # Step 4: 执行 Z-score 标准化以消除不同隐藏层维度的数值偏置，随后计算 MSE 损失
        mean_zi = z_i.mean(dim=0, keepdim=True)
        std_zi = z_i.std(dim=0, keepdim=True) + 1e-8
        
        z_i_norm = (z_i - mean_zi) / std_zi
        z_prime_j_norm = (z_prime_j - mean_zi) / std_zi
        
        cyc_loss = torch.mean((z_i_norm - z_prime_j_norm) ** 2)
        return cyc_loss


5. 算法集成性能评估与 scIB 基准测试度量体系
对不同的批次效应矫正方法进行基准测试，必须在同一个坐标框架下量化去批次能力与生物多样性保护。本节系统性地归纳 scIB 评价体系的核心数学度量指标。
5.1 scIB 评价体系指标定义
下表展示了 scIB 框架中用于横向对比的五大核心定量指标：
指标名称
分类维度
数学定义与机理
理想分值与物理含义
iLISI (integration Local Inverse Simpson's Index)
批次消除
评估局部邻域的批次混合度。对每个细胞 $i$ 统计其 $k$ 邻域内的批次概率分布 $p_i(b)$，计算局部辛普森指数倒数：$\text{LISI}_i = (\sum_{b} p_i(b)^2)^{-1}$。完美的批次混合对应分值为 $B$（批次总数），未混合的值接近 1，最后整体放缩至 $$。
越接近 1 越好。表明任意细胞的邻域内其邻居均来自于不同批次，技术偏差消除干净。
kBET (k-nearest-neighbor Batch Effect Test)
批次消除
基于 $\chi^2$ 独立性检验。在大图上采样成百上千个局部邻域，通过卡方检验测试各邻域的批次组成比率是否偏离全局真实批次比率。统计接受无显著差异（即混合极好）的测试通过率，按细胞类型加权平均。
越接近 1 越好。表明数据分布在全图所有局部区间都达到了理想的统计一致性。
PCR (Principal Component Regression)
批次消除
衡量技术标签对整体方差的解释度。计算矫正前后，数据前 $M$ 个主成分与批次分类协变量进行线性回归得到的决定系数 $R^2$。总贡献分值求和后做 $1 - R^2$ 的变换，以保证大分值代表小影响。
越接近 1 越好。说明潜空间中的前几个特征轴完全承载生物变异，技术方差被彻底压缩。
Silhouette Batch
批次消除
轮廓系数在批次控制上的延伸。在各细胞类型内部独立计算基于批次标签的平均轮廓宽度（Silhouette Width）。最终将该分值经过 $1 - \|S_i$ 线性变换至 $$ 区间。
越接近 1 越好（轮廓系数越接近0）。越接近 0 表明相同细胞类型的不同批次完全重合无法分离。
Graph Connectivity
生物学保留
检验谱系内连接。计算整合后的 kNN 图。针对特定的细胞类型，计算其所包含细胞在图上的最大连通分量占比。最后在全部已知细胞类型中计算其平均值。
越接近 1 越好。表明属于同一种细胞类型的细胞在大图上彼此相连，未被过矫正算法强行撕裂。

5.2 核心算法特性横向基准测试对比
除了经典的概率深度学习和锚点检索外，单细胞基础模型（如 scGPT）也在整合领域占据了一席之地。基于多项国际权威基准测试以及强批次应用数据，本报告对典型算法的深度矫正性能进行了横向提炼与总结：
算法路径
代表性工具
核心优势与特点
局限性与风险
适用场景与规模
经典 cVAE
scVI , totalVI , liam
严格的概率密度推断，能精准拟合 UMI 的 NB/ZINB 计数分布；生物保留极佳，对弱/中度批次去除极为高效且稳定。
在处理跨物种、类器官对比等强系统差异时，存在欠矫正，无法彻底对齐。
标测 CITE-seq，或百万级以上中等批次效应的多样本 RNA-seq 整合任务。
强正则 cVAE
sysVI
结合 VampPrior 与跨批次循环转译，极大避免了对高维生物特异性的粗暴破坏，批次去除和生物保留的帕累托前沿极好。
模型训练增加了转译和再编码流程，对内存、显卡及超参调节（如 z_distance_cycle_weight）有更高的技术要求。
跨物种比较（如人类-小鼠胰腺）、单细胞核 snRNA 与单细胞 scRNA 整合等强技术系统差异。
领域对抗 (DANN)
scGLUE
对抗训练对批次信息的清除极具破坏力，能强制模糊各批次的边界。
对判别器超参极其敏感，当各批次细胞组成比例极度不均时，极易合并完全无关的邻近细胞类型（过矫正严重）。
跨模态无配对组学对齐（例如，在不同细胞中测得的 RNA-seq 与 ATAC-seq 特征对齐）。
深度度量学习
scDML , scCRAFT
scDML 通过局部 MNN 锚点结合全局层次聚类拉开细胞距离；scCRAFT 利用各批次内双分辨率拓扑采样的三元组保护极其稀有的边缘细胞状态不丢失。
scDML 极度依赖初筛 MNN 的品质；对于低共享细胞类型的极限情境，锚点缺失会导致矫正失效。
样本间存在独有细胞亚型，或不完全重叠的马赛克整合。
域特定批归一化
SCALEX
异步 VAE 架构，通过在解码器中利用 DSBN 分支路由解耦批次方差，编码器完全 Batch-free 。
局部微弱的发育轨迹演变特征，容易在 DSBN 分支的强标准化对齐过程中产生一定程度的平滑损失。
超千万级单细胞巨型图谱构建；支持无需重训的在线、增量式 query 细胞投影。
单细胞大语言模型
scGPT
基于 Transformer 架构和海量多组织数据预训练。在批次矫正微调中展现了卓越的表征泛化能力。
微调过程严重依赖超参数：使用 cross-entropy 进行表达量重构优于 MSE；梯度反转 approach（如仅使用 GRL）效果最差，需要 GEPC 等多损失联合约束。
依赖预训练基础知识，在有充足算力的多源跨组织复杂病理切片及全景整合任务中表现极佳。

6. 结论与单细胞多组学多批次整合选型建议
在实际的单细胞多样本、多批次多组学数据分析中，算法的选择应该基于生物学背景、样本的异质性分布以及计算资源。
6.1 弱至中等批次下的常规转录组/多组学整合
对于来自同一实验室、使用相同或相似测序平台（如 10X Chromiumv3 对比 v2）测得的常规多样本 scRNA-seq 或 CITE-seq 整合，应当首先采用基于标准似然的概率自编码器模型，如 scVI 或 totalVI 。此类模型的概率似然对 UMI 计数的物理背景有着深厚的机理刻画，且其变分下界的正则化程度相对温和，能够在确保基本批次消除的同时，近乎无损地保留极其细微的细胞状态、疾病激活态演变以及发育轨迹连续性，且计算开销较小，不易引入过矫正伪影。
6.2 存在极端异质性背景的“强批次”跨系统整合
若多样本数据之间存在极其严重的技术或物理学硬跨度（例如：老鼠器官与人体组织的跨物种演变对比、完全配对的类器官 organoid 模型与真实成人原位组织对比、同一患者的 scRNA-seq 细胞悬液与 snRNA-seq 冰冻单细胞核悬液对比），此时常规的 cVAE 与线性对齐会发生欠矫正崩溃。在此类场景下，强烈建议选用结合了可表达多峰后验的 VampPrior 先验与隐空间跨系统转译的 sysVI 模型。VampPrior 有效避免了多细胞群在潜空间的合并坍塌，而隐空间循环一致性（Cycle-Consistency）能够在不同物种/系统的同类型细胞间建立起高度平滑、有生物学语义保护的微观对齐，是强批次效应分析的黄金工具。
6.3 跨多源实验室的多中心马赛克多模态整合
当需要对来自不同研究中心、测序平台不一、且各数据集所测抗体面板（Antibody Panels）不尽相同（包含缺失模态的马赛克整合，如部分数据集仅有 RNA，部分有 CITE-seq 配对）的数据进行整合时，推荐引入具有高度定制化似然的 liam 或 MultiVI 模型。此类模型能够灵活定义并分别计算 RNA、蛋白质和 ATAC 各组学模态的似然损失（如 ATAC 模态的负多项似然），在保障多模态特征独立重建的前提下，通过跨模态对抗分类或隐空间 MNN 锚点机制约束整体低维空间的一致对齐，具有极佳的多模态马赛克兼容能力。
6.4 跨巨量规模单细胞图谱（Atlas）构建与持续在线扩充
当面临千万级细胞规模的超级单细胞图谱集成任务，或需要在未来频繁、持续地将新产生的、多中心的 query 样本数据集在线扩展投影至已构建好的参考图谱（Reference Atlas）中时，SCALEX 的解耦非对称 VAE 机制提供了最优的工程解。SCALEX 彻底移除了编码器端对批次标签的依赖，其提取的 latent 纯粹保留了生物自相关的共性变异。得益于其领域特定批归一化（DSBN）对技术偏差的完美解耦，新流入的追加数据集只需在解码器侧快速构建或动态分配一组专属的 DSBN 缩放参数，即可在无须将历史全部数据集拿出来进行耗时耗力的整体重新训练、也无须重构庞大 MNN 图的工程条件下，实现真正的“在线、即时、无缝”投影对齐，满足巨型单细胞云端图谱的基础设施建设诉求。

