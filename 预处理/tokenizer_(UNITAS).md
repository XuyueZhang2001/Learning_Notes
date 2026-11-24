# UNITAS 轨迹 Tokenizer

UNITAS 用轨迹 tokenizer 将每个 3D 点的整段运动编码为一个 256 维 token，再按查询时间恢复点的位移。

排版沿用手写笔记的层次。黑笔为正文，*蓝笔为斜体小标题与引用块中的推导*，**绿笔为粗体补充**。

## 输入与输出

Tokenizer 是一个**按物理时间条件化的自编码器**。对第 $n$ 个点，设 $t$ 为当前时刻，$f$ 为有效采样率，$H$ 为未来采样步数。编码器读取未来各时刻的 3D 位移与时间偏移，输出一个 256 维向量 $z_n$。解码器给定 $z_n$ 和查询时间，恢复该时刻的 3D 位移。

$$
\tau_k=\frac{k}{f},\qquad
\Delta \mathbf{x}_{n,k}
=\mathbf{x}_{t+\tau_k,n}-\mathbf{x}_{t,n},
\qquad k=1,\ldots,H.
$$

> *符号与例子*  
> $k$ 是采样步，$n$ 是点的索引，$\mathbf{x}_{t,n}$ 是该点当前的 3D 位置。以 10 Hz 采样时，$k=2$ 对应 0.2 秒，该时刻的位移相对于预测起点计算。$H$ 个未来位移与起始位置一起，能表示 $H+1$ 个位置。

## 编码、解码与训练

**编码器**先把每个时刻的位移映射到特征空间，再按该时刻的时间编码做 FiLM 调制，最后沿时间维度汇聚。Padding 项从求和中排除。

$$
\begin{aligned}
h_{n,k} &= \mathrm{MLP}_{\mathrm{in}}(\Delta \mathbf{x}_{n,k}),\\
z_n &= \frac{1}{\sqrt H}\sum_{k=1}^{H}
\left[(1+a_k^{\mathrm{enc}})\odot h_{n,k}+b_k^{\mathrm{enc}}\right].
\end{aligned}
$$

### *时间戳怎样变成 FiLM 参数*

先用固定公式把标量时间偏移 $\tau_k$ 编码为向量 $e_k$，再用可训练的时间 MLP 输出逐通道缩放和偏移。UNITAS 在各数据源上共用 $T_{\max}=3\,\mathrm{s}$，时间编码维度为 $d_z=256$。未来和历史分别使用正、负时间偏移。

$$
[e_k]_j
=
\operatorname{sgn}(\tau_k)
\sin\!\left[
\frac{\pi}{2}
\left|\frac{\tau_k}{T_{\max}}\right|^{j/d_z}
\right],
\qquad j=0,\ldots,d_z-1.
$$

$$
\begin{bmatrix}
a_k^{\mathrm{enc}}\\
b_k^{\mathrm{enc}}
\end{bmatrix}
=
\mathrm{MLP}^{\mathrm{enc}}_{\mathrm{time}}(e_k),
\qquad
a_k^{\mathrm{enc}},b_k^{\mathrm{enc}}\in\mathbb{R}^{256}.
$$

> *先调制，再求和*  
> 同一时刻的各个点共享 FiLM 参数，各点保留自己的位移特征。即使两个时刻都对应向上 5 cm，时间条件也会改变这两个位移特征对 $z_n$ 的贡献。

**解码器**用自己的时间 MLP 从 $e_k$ 得到 $a_k^{\mathrm{dec}},b_k^{\mathrm{dec}}$，用它们调制同一个 $z_n$，再输出所查询时刻的位移。把预测位移加回起始 3D 坐标，即可恢复未来点的位置。

$$
\widehat{\Delta \mathbf{x}}_{n,k}
=
\mathrm{MLP}_{\mathrm{out}}
\left[(1+a_k^{\mathrm{dec}})\odot z_n+b_k^{\mathrm{dec}}\right],
\qquad
\widehat{\mathbf{x}}_{t+\tau_k,n}
=
\mathbf{x}_{t,n}+\widehat{\Delta \mathbf{x}}_{n,k}.
$$

**训练**在有效的点和时刻上最小化平方欧氏重建误差。历史和未来动作流的重建损失取平均，再加上可用的场景流重建损失。训练结束后冻结 tokenizer，用训练集统计量逐通道标准化 latent，再交给主模型训练。

$$
\mathcal{L}_{\mathrm{rec}}
=
\frac{1}{|\mathcal V|}
\sum_{(n,k)\in\mathcal V}
\left\|\widehat{\Delta \mathbf{x}}_{n,k}
-\Delta \mathbf{x}_{n,k}\right\|_2^2.
$$

这里 $\mathcal V$ 是某一类轨迹中有效的点与时刻对。

**夹爪点、人手关键点和场景点共用这一套 tokenizer。主模型每个点只处理一个轨迹 token，并额外用该点的初始 3D 位置作为空间锚点。**

## 附录　训练数据对齐

### 样本如何组织

UNITAS 按**同一条示范中的当前时刻 $t$**切出训练窗口。该窗口包含当前观测、任务指令、历史动作流，以及已有的未来动作流、场景流和控制标签。观测和点轨迹共享窗口内固定的米制 3D 坐标系。机器人数据通常用机器人基座或本体共享坐标系，第一人称人手数据用窗口锚点的相机坐标系。

### 空间对齐

| 数据 | 怎样映射到训练窗口 |
| --- | --- |
| **视觉观测** | 对有深度的像素，用相机内参和深度反投影，再用外参变换到共享坐标系。把对应的 3D 位置编码加到视觉 token 上。 |
| **动作流** | 对每条机械臂，在夹爪的 URDF 几何上选 20 个固定点，保存它们的连杆局部坐标。用每帧的关节状态做正向运动学，把同一批点转到共享坐标系。人手数据用每只可见手的 21 个 MANO 关键点，并用相机位姿把轨迹转到窗口锚点坐标系。 |
| **场景流** | 用 URDF 和当前关节状态遮去机器人像素，再从剩余 RGB-D 像素反投影并采样场景查询点。仿真状态提供同一批场景点在未来时刻的对应位置。 |
| **控制标签** | 把同一窗口内的末端相对平移、旋转和夹爪开度作为控制分支的标签，与同段示范的夹爪点轨迹配对。 |

#### *像素怎样进入共享坐标系*

设 $d$ 是像素 $(u,v)$ 的深度，$K$ 为相机内参，$R,\mathbf t$ 为相机到共享坐标系的刚体变换。

$$
\mathbf p_{\mathrm{cam}}=dK^{-1}[u,v,1]^\top,\qquad
\mathbf p_{\mathrm{world}}=R\mathbf p_{\mathrm{cam}}+\mathbf t.
$$

> *纯 RGB 的处理*  
> 模型在已标定的相机射线上，从 0.1 到 2.0 米采样 64 个候选深度，用 patch 视觉特征给候选点的位置编码加权。有深度时，真实深度派生的 Fourier 位置特征提供监督。具体的场景流查询点由深度或已有几何给出初始 3D 坐标。

### 时间对齐与监督

动作流和场景流共用当前时刻 $t$，各步都用物理时间偏移 $\tau_k=k/f$ 表示。RoboTwin 的 **15 Hz、20 步**覆盖约 **1.33 秒**，LIBERO 的 **20 Hz、20 步**覆盖 **1 秒**。Tokenizer 用以秒为单位的时间条件处理两种轨迹。

例如，在一段插花示范中，场景分支读取接下来这段时间的**夹爪 3D 轨迹**，用同一段时间里**花上的点实际怎样移动**作为监督。点的初始位置、未来位置和夹爪轨迹都在同一坐标系中。

**场景流监督来自仿真。不同来源可以提供不同的标注，缺失的监督项会被掩码跳过。这样，混合训练可以同时利用仿真、真机和人手示范。**

---

对应图 2、第 3.3–3.4 节和附录 A.1–A.2。[UNITAS 论文](https://arxiv.org/abs/2610.12099)
