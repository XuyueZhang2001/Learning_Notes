# UNITAS Rectified Flow

UNITAS 用 Rectified Flow 训练 DiT，从噪声生成动作流、场景流的轨迹 latent 或可执行的机器人控制指令。

排版沿用手写笔记的层次。黑笔为正文，*蓝笔为斜体小标题与引用块中的推导*，**绿笔为粗体补充**。

## 输入与输出

设 $y$ 为干净目标，$\epsilon\sim\mathcal N(0,I)$ 为同形状的高斯噪声，$\gamma\in[0,1]$ 为生成进度，$c$ 为该分支的条件。DiT 根据当前带噪状态 $y_\gamma$、$\gamma$ 和 $c$ 预测速度向量 $v_\theta$，其中 $\theta$ 是模型的可训练参数。

$$
y_\gamma=(1-\gamma)\epsilon+\gamma y,\qquad
\mathcal L_{\mathrm{flow}}
=
\left\|v_\theta(y_\gamma,\gamma,c)-(y-\epsilon)\right\|_2^2.
$$

### *为什么训练目标是 $y-\epsilon$*

对一对固定的噪声与干净目标，沿两者之间的直线路径对 $\gamma$ 求导，得到该训练样本的监督速度。

$$
\frac{d y_\gamma}{d\gamma}
=
\frac{d}{d\gamma}\big[(1-\gamma)\epsilon+\gamma y\big]
=
y-\epsilon.
$$

> *两个时间变量*  
> $\gamma$ 表示生成进行到哪一步，$v_\theta$ 是目标表示随 $\gamma$ 的变化率。$\tau_k$ 表示距当前观测时刻的物理时间偏移。

## 与 Flow Matching 的关系

就训练目标而言，**Rectified Flow 是 Flow Matching 采用线性插值时的一个特例**。Flow Matching 可以使用不同的条件概率路径，包括扩散型路径。在常见的高斯条件路径下，可以用 $\alpha(\gamma)$ 和 $\sigma(\gamma)$ 决定干净样本和噪声在各时刻的权重。

| 方法 | 训练路径 $y_\gamma$ | 速度监督 $dy_\gamma/d\gamma$ |
| --- | --- | --- |
| **Flow Matching 的一种常见形式** | $\alpha(\gamma)y+\sigma(\gamma)\epsilon$ | $\alpha'(\gamma)y+\sigma'(\gamma)\epsilon$ |
| **UNITAS 使用的 Rectified Flow** | $\gamma y+(1-\gamma)\epsilon$ | $y-\epsilon$ |

取 $\alpha(\gamma)=\gamma$、$\sigma(\gamma)=1-\gamma$，就得到 UNITAS 的公式。在标准高斯噪声与数据样本独立配对、终点残余噪声 $\sigma_{\min}=0$ 的设定下，这也对应 Lipman 等人论文中线性条件路径的训练目标。

> *训练路径是直线，推理轨迹为什么还能弯曲？*  
> 对同一对 $(\epsilon,y)$，监督速度 $y-\epsilon$ 沿线恒定。不同训练配对却可能在同一生成时刻经过同一位置，提供不同的速度目标。模型只能在该位置给出一个速度，因此在均方误差下学习的是给定当前位置、生成进度和条件后的平均速度。对这个共享速度场做积分，得到的推理轨迹仍可能弯曲。

### *Reflow 如何让推理轨迹更直*

先用已训练的模型从噪声 $\epsilon$ 积分到生成结果 $\hat y$，再用模型产生的新配对 $(\epsilon,\hat y)$ 做一轮线性插值和速度回归。每轮使用同样的速度回归形式，改变的是噪声与端点的配对。这样可以使推理轨迹更直，便于用更少的积分步数采样。

**UNITAS 论文描述了线性插值和速度回归，未报告额外的 reflow 阶段。**

原始论文：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) · [Flow Straight and Fast](https://arxiv.org/abs/2209.03003)

## 三个训练目标与推理

UNITAS 用两个逐层耦合的 DiT。Point Dynamics DiT 负责动作流和场景流，Action DiT 通过共享的观测与语言 K/V 生成机器人控制指令。

| 分支 | 干净目标 $y$ | 预测条件 $c$ |
| --- | --- | --- |
| **Action flow** | 夹爪或人手关键点的未来轨迹 latent | 当前观测、语言指令、历史动作流及动作点的当前 3D 位置 |
| **Scene flow** | 场景点的未来轨迹 latent | 同一交互的观测、语言和历史，另加场景查询点的初始 3D 位置和示范中的干净未来动作流 |
| **Robot commands** | 未来各步的末端相对平移、旋转和夹爪开度 | 共享观测与语言特征、当前末端状态 |

训练时，同一段示范动作轨迹编码得到 $A_{\mathrm{clean}}$。它直接为场景流提供条件，也经加噪得到 $A_\gamma$，供动作流分支学习生成轨迹。三项损失的权重各为 1，另加权重为 0.1 的 3D 位置编码损失。

推理时，从高斯噪声出发，用 DiT 预测速度并做 Euler 更新，得到轨迹 latent 或控制指令。生成或外部给定的动作流可以继续用于场景预测。只执行机器人控制时，可以直接生成控制指令。

**Rectified Flow 定义训练目标与采样方法，DiT 实现速度场 $v_\theta$。一般也可以用 U-Net 或 MLP 实现速度场。UNITAS 用 Transformer 在观测、语言和轨迹 token 之间做注意力交互。**

---

对应图 2、第 3.5 节和附录 A.3。[UNITAS 论文](https://arxiv.org/abs/2610.12099)
