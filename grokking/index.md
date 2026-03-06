---
layout: default
title: Grokking
---

<div class="page-nav">
  <a href="{{ site.baseurl }}/" class="btn btn-secondary">← Home</a>
  <a href="{{ site.baseurl }}/loss-plateau/" class="btn btn-secondary">Loss Plateau →</a>
</div>

<div class="hero">
  <h1>Grokking: Shortening the Delay</h1>
  <p>How we reduced a 332,000-step generalization delay down to just 1,050 steps.</p>
</div>

<div class="main-content">

  <div class="section">
    <h2 class="section-title">What is Grokking?</h2>
    <p class="section-intro">
      Grokking is a two-phase learning phenomenon first described by Power et al. (2022).
      A model reaches near-perfect training accuracy early, then stalls for a very long time
      before validation accuracy suddenly jumps.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Phase 1 — Memorization</h3>
        <p>Training accuracy hits ~100% quickly. The model finds a lookup-table shortcut
        but learns nothing generalizable. Validation accuracy stays near 0%.</p>
      </div>
      <div class="card">
        <h3>Phase 2 — Generalization</h3>
        <p>Under continued weight-decay pressure, internal representations slowly reorganize
        into compact, algorithmic structures. Validation accuracy then jumps sharply.</p>
      </div>
    </div>
    <div class="callout">
      <strong>Our baseline:</strong> A 2-layer decoder Transformer on modular division (Z₉₇)
      memorized training data in ~2,000 steps but took <strong>334,000 total steps</strong>
      to generalize — a grokking delay of ~332,000 steps.
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/grokking_plot.png"
           alt="Baseline grokking at 334k steps">
      <figcaption>Figure 1 — Baseline (AdamW, Division only). Training accuracy saturates by ~2,000
      steps while validation accuracy stays near 0% until the abrupt transition at 334k steps.</figcaption>
    </figure>
  </div>

  <div class="section">
    <h2 class="section-title">Experimental Setup</h2>
    <p class="section-intro">
      We trained a small decoder-only Transformer on modular arithmetic tasks — a controlled
      setting where grokking is well-documented and easy to measure precisely.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Model</h3>
        <p>2-layer decoder Transformer<br>
        d_model=128, 4 heads<br>
        ~422,627 parameters</p>
      </div>
      <div class="card">
        <h3>Default Optimizer</h3>
        <p>AdamW<br>
        LR=1e-3, weight decay=1e-3<br>
        Batch size=512, 50% train split</p>
      </div>
      <div class="card">
        <h3>Tasks (mod 97)</h3>
        <p>Division: x·y⁻¹ mod p (hardest)<br>
        Addition: (x+y) mod p<br>
        Subtraction: (x−y) mod p<br>
        Multiplication: (x·y) mod p</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">Experiment 1: Task Diversity</h2>
    <p class="section-intro">
      Samuel investigated whether training on multiple arithmetic operations simultaneously
      would force shared representations and accelerate grokking on Division.
    </p>
    <div class="callout">
      <strong>Key insight:</strong> Algorithmic complexity determines grokking speed.
      Multiplication groks in ~40k steps; Division takes ~334k — over 8× longer.
    </div>
    <div class="card-grid">
      <div class="card">
        <h3>2-Task (Div + Add) — Failed</h3>
        <p>Validation stuck at <strong>5.09%</strong> after 250k steps. Division and Addition
        are too similar — no meaningful synergy forms.</p>
      </div>
      <div class="card">
        <h3>3-Task (Div + Add + Sub) — Promising</h3>
        <p>Validation climbed to <strong>64.60%</strong> by 250k steps and still rising.
        Strong synergy building but grokking not yet complete.</p>
      </div>
      <div class="card">
        <h3>4-Task Balanced — Breakthrough</h3>
        <p>With strict balanced batching (256 examples per task), Division grokked at
        <strong>~80,000 steps</strong> — a 4× speedup over baseline.</p>
      </div>
    </div>
    <div class="callout">
      <strong>Critical finding:</strong> Random multi-task sampling causes catastrophic
      interference (stalls at 2.16%). Samuel's strict balanced batching — 256 examples
      per task per batch — was the essential unlock.
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/50_division_50_addition.png"
           alt="2-task: 50% division 50% addition">
      <figcaption>Figure 2 — 2-Task (50% Division / 50% Addition). Validation flatlines at 5.09%.</figcaption>
    </figure>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/33_addition_33_subtraction_34_division.png"
           alt="3-task split">
      <figcaption>Figure 3 — 3-Task (34% Div / 33% Add / 33% Sub). Validation reaches 64.60% and is still climbing at 250k steps.</figcaption>
    </figure>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/4_task_split.png"
           alt="4-task split">
      <figcaption>Figure 4 — 4-Task randomized (250k steps). Only 16.05% — but with balanced batching extended to 400k steps, Division grokked at ~80,000 steps.</figcaption>
    </figure>
  </div>

  <div class="section">
    <h2 class="section-title">Experiment 2: Optimizer Noise</h2>
    <p class="section-intro">
      Tommy investigated whether replacing AdamW with SGD — which introduces more gradient
      noise — could push the model out of its memorization basin faster.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>SGD LR=0.01 — Unstable</h3>
        <p>Grokked at step <strong>16,950</strong> (delay: 14,900 steps, ~22× faster)
        but catastrophically collapsed immediately after. Not usable.</p>
      </div>
      <div class="card">
        <h3>SGD LR=0.005 — Stable</h3>
        <p>Grokked at step <strong>44,900</strong> (delay: 41,500 steps, ~8× faster).
        Final validation accuracy: <strong>86.32%</strong>. Genuine stable speedup.</p>
      </div>
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/sgd_result.png"
           alt="SGD LR=0.01: fast but unstable">
      <figcaption>Figure 5 — SGD LR=0.01. Grokking at 16,950 steps but catastrophic collapse shortly after.</figcaption>
    </figure>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/sgd_final.png"
           alt="SGD LR=0.005: stable grokking">
      <figcaption>Figure 6 — SGD LR=0.005. Stable grokking at 44,900 steps, final validation 86.32%.</figcaption>
    </figure>
  </div>

  <div class="section">
    <h2 class="section-title">Experiment 3: Initialization</h2>
    <p class="section-intro">
      Tommy attacked the root cause: standard initialization gives the model excess capacity,
      letting it find a memorization shortcut. What if we removed that option entirely?
    </p>
    <div class="callout">
      <strong>Hypothesis:</strong> Constraining initial weight magnitudes forces the model
      to find a generalizing solution from the start — because there is no room for a
      bloated memorization circuit.
    </div>
    <div class="card-grid">
      <div class="card">
        <h3>Sparse Init (sparsity=0.9)</h3>
        <p>90% of weights zeroed at start.<br>
        Grokking delay: <strong>1,050 steps</strong><br>
        Final validation: <strong>99.68%</strong></p>
      </div>
      <div class="card">
        <h3>Small Init (scale=0.01)</h3>
        <p>Weights initialized at tiny uniform scale.<br>
        Grokking delay: <strong>1,050 steps</strong><br>
        Final validation: <strong>73.75%</strong></p>
      </div>
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/init_sparse.png"
           alt="Sparse initialization result">
      <figcaption>Figure 7 — Sparse Init (sparsity=0.9). Grokking delay just 1,050 steps. Final validation 99.68%.</figcaption>
    </figure>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/init_small.png"
           alt="Small initialization result">
      <figcaption>Figure 8 — Small Init (scale=0.01). Grokking delay just 1,050 steps. Final validation 73.75%.</figcaption>
    </figure>
    <div class="callout">
      <strong>Result:</strong> Both strategies reduced the grokking delay from ~332,000 steps
      to just <strong>1,050 steps</strong> — an approximately <strong>316× speedup</strong>,
      essentially eliminating the plateau entirely.
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">Summary</h2>
    <div class="card-grid">
      <div class="card">
        <h3>Task Diversity</h3>
        <p>4-task balanced training<br>
        334,000 → ~80,000 steps<br>
        <strong>~4× speedup</strong></p>
      </div>
      <div class="card">
        <h3>SGD Optimizer</h3>
        <p>LR=0.005, stable run<br>
        332,000 → 41,500 step delay<br>
        <strong>~8× speedup</strong></p>
      </div>
      <div class="card">
        <h3>Initialization</h3>
        <p>Sparse or small init<br>
        332,000 → 1,050 step delay<br>
        <strong>~316× speedup</strong></p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">References</h2>
    <p class="section-intro">
      Power et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.<br><br>
      Lee et al. (2024). Grokfast: Accelerated Grokking by Amplifying Slow Gradients. arXiv:2405.20233.<br><br>
      Lyu & Li (2024). Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking. ICLR 2024.
    </p>
  </div>

</div>