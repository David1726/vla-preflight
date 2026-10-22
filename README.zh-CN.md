# VLA Preflight 中文说明

把机器人策略接到环境前，重放同一份输入，对照 reference 与 candidate，找出
**第一个有记录证据的处理边界差异**。适用场景包括相机映射、状态预处理、动作反归一化、
chunk 消费和 episode reset。核心实现只依赖 Python 与 NumPy。

当前版本为 `0.1.0rc1` 发布候选，GitHub 仓库为 `David1726/vla-preflight`。
尚未发布 PyPI；安装请用源码，不能假定同名包已在 PyPI 发布。

## 一键体验

先获取源码，再在项目目录执行：

```sh
git clone https://github.com/David1726/vla-preflight.git
cd vla-preflight
```

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-cache-dir --index-url https://pypi.org/simple .
.venv/Scripts/python.exe -m vla_preflight demo --out demo-output
Start-Process demo-output/index.html
```

每次 demo 使用新的输出目录，避免覆盖已有 trace。安装需要下载依赖；安装后离线运行，
不需要机器人、GPU、模型权重、服务器或 API key。Linux/macOS 命令见英文 README。
本机成功复验共 **33.4 秒**，包括新环境、源码安装和全部 demo 报告；pip 缓存已禁用，
此前尝试可能使系统/网络缓存变暖。之前因全局缓存不可写而失败的尝试也已记录，没有改系统权限。

打开报告后依次查看：健康对照 → camera mapping → missing reset → fixed。
每个故障都明确标为 **seeded fault（人为注入）**，没有声称发现上游 bug。
10 个场景包括 5 个故障、健康与修复 2 个通过对照，以及缺单位、缺 hook、未固定随机性
3 个 UNKNOWN 对照。5 个故障均在预期阶段检出；这个小型合成集合不能代表真实检出率。

## 用于自己的 PR

1. 用 `Capture` 在原始输入、模型输入、模型输出、控制命令等需要的位置插入 hook。
2. 显式声明语义、单位、布局，按 session/episode/step 对齐，记录软件/模型版本。
3. 从 reference 的 `Trace.replay()` 读取原始输入，送入 candidate；随机策略需要固定随机性。
4. 执行下面的命令，把 HTML/JSON 作为 PR 的回归证据。

```sh
vla-preflight compare traces/reference traces/candidate --out parity-report
```

退出码：0 为 PASS，1 为 FAIL，2 为 UNKNOWN，3 为输入/命令无效。
普通 CI 会把 UNKNOWN 当成未通过。报告显示相机对照、动作曲线、首个数值偏差、
episode/阶段时间线、容差和版本。曲线只预览平铺后的通道 0；数值比较覆盖全部元素。

[完整捕获教程](docs/capture.md) · [真实接入教程](docs/integrations.md) ·
[GitHub Actions 示例](examples/github-actions.yml)

## 已运行的真实接入

- **OpenPI 原版 ActionChunkBroker**：固定源码 SHA，原版 infer/reset 和 dm-tree 实际执行，
  模型为确定性轻量替身。健康/修复通过；调用方多消费动作、漏传 reset 的人为故障失败。
  broker 未声明多 session 支持，因此不运行交错 session 要求。
- **LeRobot 原版处理器**：固定源码 SHA，真实 MEAN_STD normalizer、unnormalizer、
  converters、基础 pipeline 和类型代码实际执行。隔离包初始化，避开与本测试无关的
  dataset/video/robot 依赖，没有替换待测数值代码。健康/修复通过；调用方重复 inverse
  在 command 阶段失败。检查了显式期望数值和原始输入不被修改。

这不等于验证全部上游版本、完整安装、模型推理、仿真或真机。固定版本与精确加载范围
见 [兼容矩阵及教程](docs/integrations.md)，原始结果见 `docs/evidence/`。

## 结果边界

PASS 只证明已记录边界上满足声明的对照条件；reference 本身需要调用方确认。
两个同样接错的 pipeline 仍可能通过。不能把 PASS 称为模型成功率或机器人安全。
缺证据返回 UNKNOWN；未插入 hook 的内部步骤不可见，只有输入输出时不能推断中间哪行代码错。
不自动猜相机、坐标系或单位，不自动做单位换算与转置。

trace 使用 JSON 和数值 NPY，禁用 pickle，读取时校验路径与摘要。报告可能包含部署图像、
状态和配置，分享前请匿名化。同步捕获会增加运行时开销；当前不是实时控制记录器。

报告的本地 file URL 自动浏览被本次工具安全策略阻止，未绕过。交互 HTML 的浏览器视觉验证
仍需本机人工执行；首屏 GIF 是由已运行 demo 的数据生成的演示图，不冒充浏览器截图。
具体完成项和限制见 [验证说明](docs/verification.md)。

源码 MIT；可选上游原始文件保留 Apache-2.0 许可和归属。StarVLA 历史 issue/PR 的贡献
保留给原贡献者，本项目没有把它算作新发现或已实现桥接。
欢迎按 [贡献教程](CONTRIBUTING.md) 增加有健康对照、固定版本证据的真实 adapter。

GitHub 当前提供可复制的 CI 示例 `examples/github-actions.yml`。自动启用 Actions 所需的
workflow 授权仍待确认，因此尚未运行远程 CI；本地验证结果单独记录。
