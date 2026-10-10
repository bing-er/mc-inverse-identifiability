# BASELINE_SPEC — P1 / P2 配置差异逐项对照

**建立：** 2026-09-30
**用途：** 合并稿的唯一 production configuration 之依据。写稿前必须先把本表收敛成单一列。
**来源：** 全部从代码读出，非引自稿件叙述。

| 项 | P1 | P2 |
|---|---|---|
| 主源文件 | `jax-pad/scripts/inverse/p1_jacobian_spectrum.py` | `jax-pad-fisher/scripts/fisher/fim_analyze.py` |
| 前向常量 | `jax-pad/scripts/archive/inverse/multi_wl_v12_colab.py` | `jax-pad-fisher/scripts/fisher/fim_crb_prototype.py` |

---

## 1. 参数向量 — **最深的一处差异**

| | P1 | P2 |
|---|---|---|
| 坐标 | $\varphi=\log\mathbf c$，$\mathbf c=[B_US_U,\;B_U(1-S_U),\;B_DS_D,\;B_D(1-S_D),\;\mathrm{DermR},\;\mathrm{SubQ}]$ | $\psi=[S_U,\;B_U,\;S_D,\;B_D,\;\mathrm{DermR},\;\mathrm{SubQ}]$ |
| 性质 | 六个**对数浓度**（每层氧合/脱氧分开） | 六个**直接生理量**（饱和度与血容量分开） |
| 代码 | `theta_log()`：`[bu*su, bu*(1-su), bd*sd, bd*(1-sd), dermr, subq]` 取 log | `PSI_LABELS = ["StO2_U","B_U","StO2_D","B_D","DermR","SubQ"]` |

**这两套坐标之间是非线性重参数化，不是对角缩放。** 因此统一公式里的 $D$ **不是**对角尺度矩阵，而是完整的重参数化 Jacobian

$$T=\frac{\partial\varphi}{\partial\psi},\qquad J_\psi = J_\varphi\,T$$

$T$ 为分块结构（两个 $2\times2$ 块 + 两个对角元）：

$$
\frac{\partial\varphi_1}{\partial S_U}=\frac{1}{S_U},\quad
\frac{\partial\varphi_1}{\partial B_U}=\frac{1}{B_U},\quad
\frac{\partial\varphi_2}{\partial S_U}=\frac{-1}{1-S_U},\quad
\frac{\partial\varphi_2}{\partial B_U}=\frac{1}{B_U}
$$

（$\varphi_3,\varphi_4$ 对 $S_D,B_D$ 同构；$\varphi_5,\varphi_6$ 为 $1/\mathrm{DermR}$、$1/\mathrm{SubQ}$。）

**交叉验证：** P1 自己的方向助手 `direction("sto2", s, offset)` 返回 `e[offset]=1/s, e[offset+1]=-1/(1-s)`——这正是上式 $T$ 中 $\partial\varphi/\partial S$ 那一列。所以 P1 的「对数坐标下的 StO₂ 方向」就是 P2 的 StO₂ 单位向量在 $T$ 下的前推。两套谱**可以**统一，但必须显式过 $T$。

## 2. 标称工作点 — **一致**

两边同为 $S_U=0.95$、$S_D=0.65$、$B_U=0.30$、$B_D=0.10$、$\mathrm{DermR}=0.49$、$\mathrm{SubQ}=0.45$。

⚠️ $B_U=0.30$、$B_D=0.10$ 是 **legacy blood coordinate**（参照 28 g/L，非物理 BVF）。P1 的消光向量
`EPS_HbO2_WL=[68.8448, 64.5420, 8.6056, 3.8333]`、`EPS_HHb_WL=[55.8055, 61.3861, 11.1611, 7.1667]`
与 P5 记录的 `EPS_*_cm-1_per_legacy_blood_coordinate` **完全相同**。合并稿必须显式声明用的是哪一套；
换算到物理 BVF 需 $\times 5.355813953$（同时两个血容量坐标 $\div$ 同一因子以保持 $\mu_a$ 不变）。

## 3. 径向观测定义 — **不一致**

| | P1 | P2 |
|---|---|---|
| 探测器外沿 | `DETECTOR_R_MAX = 2.0` cm | `R_MAX = 1.5` cm |
| 分箱 | `linspace(0, 2.0, 51)` → 50 箱 | 30 箱 |
| 保留判据 | 箱心 $\le$ `MAX_R_CUTOFF = 1.5` → **约 38 箱** | 全部 30 箱（外沿即 1.5） |
| 观测向量长度 | $4\times38=152$ | $4\times30=120$ |
| 低计数处理 | 无 | **丢弃** $N A_j R_j <$ `MIN_COUNT = 5.0` 的箱 |

P2 源码注释明确写道网格取自 pooled FIM（Table 1），「not the prototype 50/2.0 grid」——即 P2 自己也有两套网格，
合并稿只能留一套。

## 4. 噪声模型与权重 — **不一致**

| | P1 | P2 |
|---|---|---|
| 噪声模型 | **无**。docstring 明写「deliberately does not form a Fisher information matrix, assume a noise model, compute a Cramer-Rao bound」 | 对角 Poisson 散粒噪声 |
| 权重 | 无（原始 $J$） | $W_{jj}=N A_j/R_j$，即 $C_{jj}=R_j/(NA_j)$ |
| 分解对象 | `np.linalg.svd(J)` | `eigh(I)`，$I=\sum_j W_{jj}u_ju_j^{\mathsf T}$ |

## 5. 特征分解前的尺度 — **不一致**

| | P1 | P2 |
|---|---|---|
| 尺度 | 无（对数坐标本身即相对灵敏度，docstring 称「no arbitrary rescaling is needed」） | `scale = [1.0, B_U0, 1.0, B_D0, DERMR0, SUBQ0]`，对 `FIM * outer(scale,scale)` 取 `eigh` |

## 6. 光子预算与终止 — **部分不一致**

| | P1 | P2 |
|---|---|---|
| 光子数 | `--n-photons` 默认 **30 000** | `PROD_N = 100 000` |
| step cap | `--max-steps` 默认 **2000** | **2000**（原型脚本默认 500，生产为 2000） |
| 权重截断 | 未在谱脚本中出现 | `wmin = 1e-4`，无 Russian roulette |

## 7. 波长 — **一致**

两边 `N_WL = 4`，540 / 580 / 633 / 800 nm。

---

## 已冻结的 production 决定（2026-09-30）

机器可读形式见 `spec/baseline_spec.json`。**脚本只能从那里读取，不得携带隐式默认值。**

| 项 | 决定 |
|---|---|
| 参数坐标 | $\psi=[S_U,\,f_U,\,S_D,\,f_D,\,\mu_{a,\mathrm{DermR}},\,\mu_{a,\mathrm{SubQ}}]$（生理坐标） |
| 谱分析尺度 | 无量纲归一化；StO₂ 尺度为 1，其余用标称值 |
| 径向网格 | 0–1.5 cm，**30 个固定环形箱** → 120 个观测 |
| 低计数箱 | 主分析**不使用** `MIN_COUNT=5` 删除规则，保留全部 30 箱；`MIN_COUNT=5` 仅作敏感性 profile |
| 测量光子预算 | $N_{\mathrm{meas}}=10^5$/波长，四波长共 $4\times10^5$ |
| MC 数值预算 | 与测量预算**分开**；生产目标每波长 32 streams × 200 000 photons |
| step cap | 主配置 **4000**；2000 仅作成本/收敛对照 |
| blood coordinate | 对外报告物理 BVF：$f_U=0.0560139$、$f_D=0.0186713$；代码内部可换算到 legacy |
| 似然模型 | **固定发射数 multinomial** 为主；diagonal Poisson 作复现与敏感性比较 |

### 两个光子数必须分开命名

这是本仓库最容易出错、也最容易悄悄出错的地方。P1 与 P2 都只有一个 `N`，含义却不同。

| 名称 | 含义 |
|---|---|
| `n_transport_photons` | Monte Carlo 估计 $R,J$ 的**数值精度** |
| `n_measurement_photons` | Fisher 似然中假设的**实验光子预算** |

**任何情况下不得共用一个 `N`。**

### 三个 profile

| profile | 用途 |
|---|---|
| `p1_legacy` | 复现原始 P1 raw-$J$ SVD（$\varphi$ 坐标、152 观测、无噪声模型） |
| `p2_legacy` | 复现原始 P2 diagonal-Poisson FIM（$\psi$ 坐标、120 观测、`MIN_COUNT=5`、legacy blood） |
| `production` | 新稿唯一主配置 |

先用前两个复现各自的原始结果，确认迁移无误，再运行 `production`。
