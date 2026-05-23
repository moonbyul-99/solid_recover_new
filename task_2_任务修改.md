# 多样本整合策略迭代优化计划（修订版）
> **图片评估能力声明**: 我无法从图片中视觉判断UMAP整合效果的好坏。我将生成UMAP可视化图片
> 并计算所有定量指标（ARI/NMI），但最终"好不好看"需要你来判断。每次迭代我会将定量指标
> 和UMAP图片一并汇报，你结合视觉判断给出下一步方向。

## 用户补充说明

1. 评估工具这里，既然你无法进行图片的评估，除了交给我评估外，可以发送给qwen-3.6-plus进行图片评估。你需要哪些信息可以告诉我。
2. 基线这里，我说不需要训练，是因为obsm中已经包含了训练过的embedding，你直接评估就可以了。
3. 你不需要直接规划好每一轮迭代做什么修改，要根据实施结果、用户反馈来调整。
---

## Task 1: 数据降采样与准备（优先执行）
**文件涉及**: 新建 `scripts/prepare_brain_dev_subset.py`
- 加载 `train.h5mu` 和 `test.h5mu`（路径从 `configs/case_brain_dev.yaml` 读取）
- ATAC peaks降采样：对 `mdata['atac_count'].X` 统计每个peak的非零比例，保留前100k（140k→100k）
- 细胞降采样：train随机选取40k细胞保存为 `train_sub.h5mu`，test随机选取20k细胞保存为 `test_sub.h5mu`，去重确保无交集
- 保存到当前工作目录 `/home/rsun@ZHANGroup.local/solid_recover_dev/`
- 创建 `configs/case_brain_dev_sub.yaml`（拷贝自 `case_brain_dev.yaml`）：
  - `feature_num_2`: 100000
  - `train_data_path`: `train_sub.h5mu`（相对或绝对路径）
  - `test_data_path`: `test_sub.h5mu`
  - `train_steps`: 6000
  - `warmup_steps`: 500, `steady_1_steps`: 500, `cosine_anneal_steps`: 5000
  - `save_points`: 1500
  - `eval_points`: 500
  - `batch_size`: 512

---

## Task 2: 邮件报告工具（优先实现并测试）
**文件涉及**: 新建 `scripts/send_report.py`
- 实现 `generate_markdown_report(round_info, metrics, umap_paths)` — 生成Markdown格式报告
- 实现 `send_email(report_md, recipient, smtp_config)` — 通过SMTP发送邮件到 `sunrui171@mails.ucas.edu.cn`
- **先做一个最小测试**：发送一封"测试邮件"验证SMTP连通性
- 如果SMTP不通（无可用SMTP服务器、网络限制等），放弃邮件功能，改为将报告保存到本地 `reports/` 目录
- 报告内容模板：策略描述、超参数表、每个ckpt的ARI/NMI汇总表、loss曲线描述、UMAP图片引用

---

## Task 3: 实现评估工具模块
**文件涉及**: 新建 `solid_recover/evaluation/batch_metrics.py`
- `compute_ari_nmi_leiden(embedding, labels, resolutions)`:
  - 用 `scanpy.pp.neighbors` 构建KNN图
  - 对每个resolution在 [0.1, 0.2, ..., 1.5]，用 `scanpy.tl.leiden` 聚类
  - 计算每个resolution的ARI和NMI vs ground truth labels
  - 返回 max(ARI+NMI) 对应的resolution、ARI、NMI值
- `evaluate_checkpoint(model, ckpt_path, mdata, key_1, key_2, device)`:
  - 加载ckpt，提取RNA embedding（z_mu_1）
  - 计算Class/Subclass/cell_type三组标签的ARI/NMI
  - 返回 dict: {ckpt_step, class_ari, class_nmi, subclass_ari, subclass_nmi, celltype_ari, celltype_nmi}
- `generate_umap_plots(embedding, obs, output_dir, ckpt_step, min_dist=0.2)`:
  - 使用scanpy计算UMAP
  - 按 Class、Subclass、cell_type、Group、Sample_ID、Region 分别着色
  - 保存为 `{output_dir}/umap_ckpt{step}_{color_key}.png`
  - 返回图片路径列表

---

## Task 4: 建立基线（Round 0）
**文件涉及**: `configs/case_brain_dev_sub.yaml`, Task 3的评估模块
- 使用**未经修改**的原始SR模型，在降采样数据上从头训练
- 训练完成后，对每个ckpt（1500, 3000, 4500, 6000）执行完整评估
- 生成基线报告并生成UMAP图片
- 记录Round 0的定量指标作为后续迭代的对比基准

---

## Task 5: 策略1 — CVAE解码器端批次条件注入
**文件涉及**: `solid_recover/nn/vae.py`, `solid_recover/nn/encoder.py`, `solid_recover/nn/pair_vae.py`, `solid_recover/models/pair.py`, `solid_recover/data/datasets.py`, `solid_recover/losses/composite.py`

**核心修改点**:
1. **FeatureDecoder** (`nn/encoder.py`): 新增可选 `batch_embed_dim` 参数。若>0，在 `forward` 中接收 `batch_embed`，先与 z 拼接再送入 fc_blocks
2. **SRVAE** (`nn/vae.py`): `forward` 新增可选 `batch_embed` 参数，透传给 decoder
3. **SRPairVAE** (`nn/pair_vae.py`): 新增 `nn.Embedding(num_batches, batch_embed_dim)` 作为 `batch_embeddings` 层；`forward` 新增 `batch_indices` 参数
4. **PairDataset** (`data/datasets.py`): 新增 `batch_indices` 字段；`create_dataset` 从 `mdata[key_1].obs['Sample_ID']` 映射为整数标签
5. **PairScratch** (`models/pair.py`): `_process_batch` 传递 `batch_indices` 给 `self.net`
6. **VAEClipLoss** (`losses/composite.py`): **不变**（解码器端批次注入不影响loss结构）

**配置新增字段** (`case_brain_dev_sub_cvae.yaml`):
```yaml
model:
  batch_embed_dim: 8    # 批次嵌入维度，0表示禁用CVAE
```

**参考依据**: scVI/totalVI/MultiVI均采用此策略，与现有多组学架构最兼容（Claude报告路径A）

---

## Task 6: 策略2 — 对抗训练（GRL + Batch Discriminator）
**文件涉及**: 新建 `solid_recover/nn/batch_discriminator.py`, 修改 `solid_recover/losses/composite.py`

**核心修改点**:
1. **新建 `batch_discriminator.py`**:
   - `GradientReversalFunction(torch.autograd.Function)`: forward恒等, backward乘 `-alpha`
   - `GradientReversalLayer(nn.Module)`: 包装上述Function
   - `BatchDiscriminator(nn.Module)`: 2层MLP(embed_dim→128→num_batches)，输入z_mu，输出batch logits
2. **VAEClipLoss** (`losses/composite.py`):
   - 新增 `adversarial_batch_weight` 参数（默认0.0=禁用）
   - 在 `forward` 中：对 z_mu_1 过GRL→Discriminator，计算CrossEntropy作为对抗损失
   - 总loss新增项: `+ adversarial_batch_weight * adversarial_loss`

**配置新增字段**:
```yaml
loss:
  adversarial_batch_weight: 0.01   # 对抗训练权重，0表示禁用
  grl_alpha: 1.0                   # 梯度反转系数
```

**参考依据**: scFLASH/SAFAARI/ResPAN均使用对抗训练去批次效应（DS报告2.2节）

---

## Task 7: 策略3 — Harmony启发的聚类级批次对齐损失
**（替代原Harmony后处理方案，融入模型训练过程）**

**文件涉及**: 新建 `solid_recover/losses/batch_alignment.py`, 修改 `solid_recover/losses/composite.py`

**Harmony核心思想**: 对隐空间做软K-Means聚类，以批次多样性最大化为约束，对每个cluster计算批次修正向量。

**模型内实现方案**:
1. **新建 `batch_alignment.py`**:
   - `BatchAlignmentLoss(nn.Module)`:
     - 输入: z_mu（RNA embedding）, batch_labels
     - 每步用EMA更新K个聚类中心（K默认为cell数/30，即~1333）
     - 用余弦相似度计算软分配（替代Harmony的KL-based membership）
     - 对每个cluster，计算各batch的均值，损失为各cluster内batch均值偏离全局cluster均值的程度
     - `loss = mean_k( weighted_variance_of_batch_means_in_cluster_k )`
   - 损失公式: `L_batch = Σ_k Σ_b (||μ_{k,b} - μ_k||²)，其中 μ_{k,b}是cluster k中batch b细胞的平均z`
2. **VAEClipLoss** 新增 `batch_alignment_weight` 参数
3. **训练时**: 需要batch_labels传入loss

**配置新增字段**:
```yaml
loss:
  batch_alignment_weight: 0.0     # Harmony对齐损失权重
  num_alignment_clusters: 0       # 聚类数，0=自动(N_cells/30)
```

**参考依据**: Harmony（Claude报告路径B）+ scDML的cluster-level对齐思想

---

## Task 8: 策略4 — 编码器端全局注意力抑制（MultiGAI启发，可选探索）
**文件涉及**: 新建 `solid_recover/nn/global_attention.py`, 修改 `solid_recover/nn/encoder.py`

- 在编码器末端引入一个全局可训练Key-Value集合
- 每个细胞的编码在最后通过Cross-Attention查询这些全局KV
- 使编码时参考数据集全局结构，抑制批次特异信息的流入
- **这是高优先级探索项**，仅在CVAE和对抗训练效果不足时启用

---

## Task 9: 迭代训练与评估主控脚本
**文件涉及**: 新建 `scripts/run_iteration.py`
- 编排完整流程:
  1. 读取当前轮的配置文件（如 `configs/case_brain_dev_sub_cvae.yaml`）
  2. 调用 `solid-recover train --config config.yaml`
  3. 对每个ckpt调用Task 3的 `evaluate_checkpoint` + `generate_umap_plots`
  4. 调用Task 2的 `generate_markdown_report` + `send_email`（或保存到本地）
  5. 将报告和指标追加到 `reports/iteration_log.json`
- 支持 `--round N --config config.yaml` 参数化执行
- 输出目录结构:
  ```
  reports/
    round_0_baseline/
      report.md
      umap_ckpt1500_class.png
      umap_ckpt1500_subclass.png
      ...
      metrics.json
    round_1_cvae/
      ...
    iteration_log.json   # 汇总所有轮次的指标
  ```

---

## Task 10: 版本管理与Git提交
**执行时机**: 每完成一轮有效迭代修改后
```
git add .
git commit -m "feat(agent): [策略名] 多样本整合Round N - 指标改善描述"
git push origin dev
```
**注意**: 不要提交 `train_sub.h5mu` / `test_sub.h5mu` 等大文件

---

## 迭代策略优先级与执行顺序

| Round | 策略 | 修改量 | 预期收益 |
|-------|------|--------|---------|
| 0 | 基线 (无修改) | - | 建立基准 |
| 1 | CVAE解码器批次注入 | 小 | 高（scVI系方法已验证） |
| 2 | 对抗训练 GRL | 中 | 中高（但训练不稳定风险） |
| 3 | CVAE + 对抗联合 | 中 | 高（双重约束） |
| 4 | CVAE超参调优 (batch_embed_dim=4/8/16) | 小 | 中 |
| 5 | 对抗权重调优 (0.01→0.1→1.0) | 小 | 中 |
| 6 | Harmony启发聚类对齐损失 | 中 | 中高 |
| 7 | Harmony对齐 + CVAE联合 | 中 | 高 |
| 8 | 全局注意力抑制 (MultiGAI启发) | 大 | 未知 |
| 9-10 | 兜底优化（混合策略、学习率调整等） | - | - |

**执行总次数不超过10轮**，若提前达标则停止。

---

## 核心架构修改示意

```
当前 SRPairVAE (Round 0):
  x1 → Encoder1 → z1 → Decoder1 → x1_recon
  x2 → Encoder2 → z2 → Decoder2 → x2_recon
  CLIP(z1_embed, z2_embed)
  Loss = VAE_1 + VAE_2 + CLIP + cross_recon

修改后 (Round 3, CVAE + Adversarial + Harmony对齐):
  x1 → Encoder1 → z1 ─┐
                        ├→ [z1|batch_emb] → Decoder1 → x1_recon
  x2 → Encoder2 → z2 ─┘
                        └→ [z2|batch_emb] → Decoder2 → x2_recon
  CLIP(z1_embed, z2_embed)
  GRL(z1_mu) → BatchDiscriminator → adv_loss
  BatchAlignmentLoss(z1_mu, batch_labels) → align_loss
  Loss = VAE_1 + VAE_2 + CLIP + cross_recon + w_adv*adv + w_align*align
```
