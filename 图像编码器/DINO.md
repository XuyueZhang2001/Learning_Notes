# DINO：从整图自蒸馏到局部特征关系

## 1. 三代方法概览

DINO 系列从无人工标签的图像中学习通用视觉特征。以 ViT 为主干时，模型会输出一个概括整图的 `[CLS]` token，以及一组保留空间位置的 patch tokens。全局特征可以用于分类或检索，局部特征可以用于匹配、分割，也可以像 PointWorld 一样附到三维点上。

| 版本 | 预训练读取哪些特征 | 主要监督 | 论文结果和规模 |
| --- | --- | --- | --- |
| **DINO（2021）** | `[CLS]` | 同一图像的不同裁剪之间做整图自蒸馏 | 在不使用人工标注的情况下，ViT 特征和注意力图也能反映物体区域 |
| **DINOv2（2023）** | `[CLS]` 和 patch tokens | 保留整图损失，加入 iBOT 的逐 patch 遮挡预测 | 用约 1.42 亿张筛选图像训练，得到适用于全局和密集任务的通用特征 |
| **DINOv3（2025）** | `[CLS]`、patch tokens 及 patch 两两相似度 | 大模型训练延续 DINO 和 iBOT 目标，在后期加入 Gram anchoring | 网络图像数据集扩大到约 16.9 亿张，训练 70 亿参数主模型，再蒸馏到较小的模型。Gram anchoring 改善了密集特征的局部一致性 |

## 2. 原版 DINO 的 ViT 主干

**ViT 内部包含 Transformer 编码器。** 原版 DINO 的教师和学生各有一套 ViT 主干和 DINO 投影头。下面以 **ViT-S/16** 和 $`224\times224`$ 的 RGB 图像为例。

| 阶段 | 处理 | 输出尺寸 |
| --- | --- | --- |
| 切分图像 | 切为不重叠的 $`16\times16`$ 像素块，共 $`14\times14=196`$ 个 patch | 每个 patch 有 $`16\times16\times3=768`$ 个像素值 |
| Patch embedding | 展平每个 patch，用同一个可学习线性层投影，再加对应位置的编码 | $`196\times384`$ |
| 加入 `[CLS]` token | 在序列开头加入一个可学习向量，并带有对应的位置编码 | $`197\times384`$ |
| Transformer | 用多层自注意力和前馈网络更新整个序列 | 一个 $`1\times384`$ 的 `[CLS]` 和 $`196\times384`$ 的 patch tokens |

### 2.1 一个 patch 为什么从 768 变成 384 维？

设第 $`i`$ 个 patch 展平后为 $`\mathbf x_i\in\mathbb R^{768}`$。

```math
\mathbf z_i=W\mathbf x_i+\mathbf b+\mathbf p_i,
\qquad
W\in\mathbb R^{384\times768},
\qquad
\mathbf b,\mathbf p_i,\mathbf z_i\in\mathbb R^{384}.
```

$`W`$ 的每一行对 768 个输入值做一次加权求和，共得到 384 个输出。各 patch 共用 $`W`$ 和偏置 $`\mathbf b`$。位置编码 $`\mathbf p_i`$ 是 384 维，逐元素相加后维度保持不变。

> *“/16”决定 patch 的像素尺寸，“S”对应 ViT-S 的特征维度 384。ViT-B/16 切出的 RGB patch 同样有 768 个像素值，主干特征维度则是 768。*

### 2.2 两类 token 各自表示什么？

| token | 进入 Transformer 时 | 离开 Transformer 时 |
| --- | --- | --- |
| **`[CLS]` token** | 每张图共用同一个可学习初始向量 | 通过自注意力读取各位置的信息，形成随图像变化的整图特征。原版 DINO 把这一步的输出送进投影头 |
| **Patch tokens** | 每个向量来自对应小块的像素投影和位置编码 | 每个向量仍对应原来的图像位置，并通过自注意力融合了其他位置的信息。可排回 $`14\times14`$ 的特征图 |

> *例如，一小块图像只有一段白色弧线。输入 token 根据这块像素和位置计算。输出 token 可以结合周围的杯口和杯身，把这里表征为杯柄的一部分。位置和维度都保留了，向量的内容随着上下文更新。*

原版 DINO 在 `[CLS]` 上计算训练损失。由于 `[CLS]` 会通过自注意力读取 patch 信息，这个训练信号也会更新处理 patch 的主干。见 DINO 第 3.2 节、表 1 和第 4.2 节。

## 3. 原版 DINO 的投影头和训练过程

### 3.1 投影头的输入和输出

投影头读取**最后一层的 `[CLS]` token**。原版 DINO 官方实现中，以 ViT-S 的默认维度为例：

| 步骤 | 操作 | 维度 |
| --- | --- | --- |
| MLP 第 1 层 | 线性层 + GELU | $`384\rightarrow2048`$ |
| MLP 第 2 层 | 线性层 + GELU | $`2048\rightarrow2048`$ |
| MLP 第 3 层 | 线性层 | $`2048\rightarrow256`$ |
| 特征归一化 | 将 256 维向量除以自己的 $`L_2`$ 范数 | $`256\rightarrow256`$ |
| 输出层 | 带权重归一化的无偏置线性层 | $`256\rightarrow K,\quad K=65{,}536\text{（默认）}`$ |

**投影头输出 $`K`$ 个 logits，训练损失再用带温度的 softmax 把它们转成概率分布。** 这些维度是自监督训练学到的分数维度。下游任务可直接取 ViT 的 `[CLS]` 或 patch 特征。见 DINO 第 3.1 节和官方 `DINOHead` 实现。

### 3.2 教师和学生如何训练？

从同一张图生成两个大裁剪和若干小裁剪。教师只看大裁剪，学生看大裁剪和小裁剪。用同一图像中**不同裁剪**的教师和学生分布计算交叉熵。$`x_t`$ 和 $`x_s`$ 分别表示一组教师与学生裁剪，$`p_t^{(j)}`$ 和 $`p_s^{(j)}`$ 表示其投影输出分布的第 $`j`$ 维。

```math
\mathcal L_{\mathrm{DINO}}(x_t,x_s)
=
-\sum_{j=1}^{K}p_t^{(j)}(x_t)\log p_s^{(j)}(x_s).
```

教师 logits 还经过 **centering** 和 **sharpening**。前者利用教师输出的移动平均做中心化，抑制某个输出维度长期占主导。后者用较低的 softmax 温度使目标分布更有区分度。

梯度只在学生分支传播，更新**学生 ViT 主干和投影头**。随后用学生参数的指数移动平均（EMA）更新教师对应模块。

```math
\theta_t\leftarrow m\theta_t+(1-m)\theta_s.
```

> *教师相当于过去多个版本学生参数的缓慢平均。教师和学生结构相同，各自保存一套参数。*

见 DINO 图 2、第 3.1 节公式（1）–（4）及算法 1。

## 4. DINOv2 的 iBOT 头

DINOv2 保留监督整图 `[CLS]` 的 DINO 头，并对 patch tokens 加入 **iBOT 的逐位置监督**。论文实验给 DINO 头和 iBOT 头分别训练参数。iBOT 头也是 MLP 投影头，逐个读取 ViT 输出的 patch token。

```math
\mathbf z_i\in\mathbb R^D
\xrightarrow{\mathrm{iBOT\ head}}
\boldsymbol\ell_i\in\mathbb R^K
\xrightarrow{\text{归一化}}
\mathbf p_i\in\mathbb R^K .
```

$`D`$ 为主干特征维度，$`K`$ 为投影输出维度。学生用带温度的 softmax，教师端的目标分布可用中心化 softmax 或 Sinkhorn–Knopp 归一化生成。见 DINOv2 第 4 节。

| 训练目标 | 教师输入 | 学生输入 | 比较对象 |
| --- | --- | --- | --- |
| DINO | 同一张照片的一个大裁剪 | 同一张照片的另一个大裁剪或小裁剪 | 两个裁剪的 `[CLS]` 输出分布 |
| iBOT | 一个未遮挡的大裁剪 | 同一个大裁剪，部分 patch 用 mask token 替换 | 学生被遮挡位置与教师同位置的输出分布 |

设 $`\mathcal M`$ 是学生被遮住的 patch 位置集合。逐位置交叉熵为

```math
\mathcal L_{\mathrm{iBOT}}
=
-\sum_{i\in\mathcal M}\sum_{j=1}^K
p_{t,i}^{(j)}\log p_{s,i}^{(j)}.
```

> *例如第 12 个 patch 是杯柄，教师能看到该位置的内容。学生在第 12 个位置收到的是 mask token，经过 ViT 后仍在该位置输出一个带上下文的向量。两个 iBOT 头将这两个向量分别转成分布，损失只在学生被遮住的位置计算。*

**“各 patch 共享一套权重”发生在单个分支内部。** 学生的各位置共用一个学生 iBOT 头，教师的各位置共用一个教师 iBOT 头。两个头各自保存参数。DINO 损失经学生 DINO 头回传到学生 ViT，iBOT 损失经学生 iBOT 头回传到同一个 ViT。教师的 ViT、DINO 头和 iBOT 头都用各自对应的学生参数做 EMA。

## 5. DINOv3 的 Gram anchoring

DINOv3 作者发现，在大模型长时间训练时，整图任务可以继续提升，局部 patch 特征却可能失去区分不同区域的能力。**Gram anchoring 给学生的 patch 两两相似度提供一个参考。** 参考教师最初取自训练较早、局部特征质量较好的教师检查点。见 DINOv3 第 4 节和图 5–10。

### 5.1 从 patch tokens 得到 Gram 矩阵

对同一个大裁剪，取学生和 Gram 教师在**投影头之前的 ViT patch 特征**。设裁剪有 $`N`$ 个 patch，每个特征为 $`D`$ 维。先逐个做 $`L_2`$ 归一化，再按行堆成矩阵 $`X\in\mathbb R^{N\times D}`$。

```math
\begin{aligned}
\widehat{\mathbf z}_i &= \frac{\mathbf z_i}{\|\mathbf z_i\|_2},\\
G &= XX^\top\in\mathbb R^{N\times N},\\
G_{ij}&=\widehat{\mathbf z}_i^\top\widehat{\mathbf z}_j
      =\cos(\mathbf z_i,\mathbf z_j).
\end{aligned}
```

> *$`L_2`$ 归一化把向量除以自己的欧氏长度并保留方向。例如 $`(3,4)`$ 除以 5 得到 $`(0.6,0.8)`$。两个单位向量的点积等于余弦相似度。若有 196 个 patch，Gram 矩阵就是一张 $`196\times196`$ 的相似度表。*

### 5.2 匹配学生和教师的 Gram 矩阵

```math
\mathcal L_{\mathrm{Gram}}
=\left\|G_S-G_G\right\|_F^2
=\sum_{i,j}\left(G_{S,ij}-G_{G,ij}\right)^2.
```

$`G_S`$ 和 $`G_G`$ 分别是学生和 Gram 教师的相似度矩阵。Frobenius 范数平方就是把矩阵对应项的差平方后相加。**约束的是两个 patch 之间的相似关系，单个特征向量可以继续变化。**

大模型的 refinement 阶段同时优化下列目标。三个 $`w`$ 是损失权重，$`\mathcal L_{\mathrm{DKoLeo}}`$ 鼓励全局特征在表示空间中分散。

```math
\mathcal L_{\mathrm{Ref}}
=w_{\mathrm D}\mathcal L_{\mathrm{DINO}}
+\mathcal L_{\mathrm{iBOT}}
+w_{\mathrm{DK}}\mathcal L_{\mathrm{DKoLeo}}
+w_{\mathrm{Gram}}\mathcal L_{\mathrm{Gram}}.
```

Gram 教师在 refinement 中按间隔从主 EMA 教师更新。在把 7B 模型蒸馏到较小模型的阶段，论文用训练好的 7B 模型作为固定教师，沿用初始训练阶段的损失。见 DINOv3 公式（2）、（3），第 4.2–4.3 节和第 5.2 节。

> *如果教师给出的“杯柄–杯身”相似度是 0.8、“杯柄–桌面”是 0.2，而学生对应的是 0.5 和 0.6，这两项误差会推动学生提高前一对的相似度、降低后一对的相似度。*

## 6. 在 PointWorld 中使用哪一部分？

PointWorld 用训练好的 **DINOv3 ViT-L/16** 提取起始 RGB 图像的多层 **ViT patch tokens**。DINOv3 保持冻结。这些特征被附到对应的三维场景点上，再交给 PTv3 结合夹爪轨迹预测场景运动。见 [PointWorld 笔记](PointWorld_notes.md) 和原论文附录 A.4。

## 参考资料

- [原版 DINO 论文，教师–学生结构、公式和注意力图](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
- [原版 DINO 官方实现，ViT 和 DINOHead](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
- [原版 DINO 官方训练代码，投影头默认维度](https://github.com/facebookresearch/dino/blob/main/main_dino.py)
- [DINOv2 论文，整图和逐 patch 的训练目标](https://arxiv.org/html/2304.07193)
- [iBOT 论文，masked image modeling 与在线教师](https://arxiv.org/abs/2111.07832)
- [DINOv3 论文，Gram anchoring 与局部特征](https://arxiv.org/html/2508.10104v1)
- [PointWorld 论文，DINOv3 特征与三维点的对应](https://arxiv.org/html/2601.03782v1)
