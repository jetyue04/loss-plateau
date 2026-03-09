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
  <!-- WHAT IS GROKKING -->
  <div class="section">
    <h2 class="section-title">What is Grokking?</h2>
    <p class="section-intro">
      Imagine a student who studies for a test by memorizing every answer in a textbook. At first, the student would fail
      the test as the test does not have the same problems as the textbook that the student memorized. After reviewing the material long enough, however, the student is able to truly <em>understand</em> the underlying concepts and pass any version of the test. Grokking is the neural network equivalent of this experience. First described by Power et al. (2022), it refers to a learning phenomenon where a model achieves near-perfect training accuracy early on, then appears stuck for a very long time — before validation accuracy suddenly jumps to near-perfect as well.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Phase 1 — Memorization</h3>
        <p>Training accuracy hits ~100% quickly. The model finds a lookup-table shortcut
        but learns nothing generalizable. Validation accuracy stays near 0%.</p>
      </div>
      <div class="card">
        <h3>Phase 2 — Generalization</h3>
        <p>Under continued weight-decay pressure, the model's internal understanding slowly
        reorganizes into something more principled and rule-based. Validation accuracy then suddenly increases.</p>
      </div>
    </div>
    <p class="section-intro">
      Understanding what causes this delay, — and how to shorten it — has real practical value.
      Training neural networks is expensive. If grokking can be accelerated, models could generalize
      in a fraction of the compute time, making AI development faster and cheaper.
    </p>
    <div class="callout">
      <strong>Our baseline:</strong> A small 2-layer decoder Transformer trained on modular division (mod 97)
      memorized the training data in ~0.2% training progress but required <strong>~83.5%</strong> training progress, or 334,000 total steps, to truly generalize — a grokking delay of ~83.3% training progress.
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/grokking_plot.png"
           alt="Baseline grokking at 0.2% training progress">
      <figcaption>Figure 1 — Baseline (AdamW, Division only). Training accuracy (orange dashed) reaches near perfect accuracy at around 0.2% training progress while validation accuracy (red) stays flat until the sudden transition at 83.5% training progress. The x-axis is on a log scale to make the gap visible.</figcaption>
    </figure>
  </div>

  <!-- WHY DOES IT HAPPEN -->
  <div class="section">
    <h2 class="section-title">Why Does Grokking Happen?</h2>
    <p class="section-intro">
      Grokking isn't random — there's a mathematical reason for it. Recent theoretical work (Lyu et al., 2024)
      proved that it arises from a <strong>dichotomy of implicit biases</strong> during training.
      Early in training, the optimizer is biased toward simple memorization — it finds the easiest
      solution that fits the training data, even if that solution doesn't generalize.
      But as training continues under weight decay, the optimizer is slowly pushed toward discovering
      more structured, efficient solutions — the kind that actually generalize. The transition between
      these two regimes is what produces the sudden jump in validation accuracy we call grokking.
    </p>
    <p class="section-intro">
      In short: <strong>weight decay is the driver of grokking</strong>. Without it, models memorize
      and stay memorized. With it, the model is eventually forced to find a better answer —
      it just takes a very long time by default.
    </p>
  </div>

  <!-- EXPERIMENTAL SETUP -->
  <div class="section">
    <h2 class="section-title">Our Approach</h2>
    <p class="section-intro">
      We used modular arithmetic for our experiments for a controlled setting where grokking is well-documented,
      fully controlled, and easy to measure exactly. The task: given two numbers <em>a</em> and
      <em>b</em>, predict the result of an arithmetic operation modulo 97 (e.g., 45 ÷ 23 mod 97).
      We then systematically tested three strategies to accelerate the transition from memorization
      to generalization.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Model</h3>
        <p>2-layer decoder Transformer<br>
        d_model=128, 4 attention heads<br>
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
  
  <!-- EXPERIMENT 1: TASK DIVERSITY -->
  <div class="section">
    <h2 class="section-title">Experiment 1: Task Diversity</h2>
    <p class="section-intro">
      Our first strategy: train on multiple arithmetic operations at the same time.
      The intuition is that all four operations share the same underlying modular structure —
      so a model trained on several tasks simultaneously is nudged toward learning
      that shared structure rather than memorizing task-specific shortcuts.
      This promotes the kind of general, algorithmic representations that support generalization.
    </p>
    <div class="callout">
      <strong>Key observation from prior work:</strong> Not all tasks are equally hard to grok.
      In single-task runs, addition groks at ~14% of training progress, multiplication at ~16%,
      subtraction at ~37% — but division takes until <strong>83.5%</strong>. Division requires
      computing modular inverses, making it the most structurally complex of the four.
    </div>
    <p class="section-intro">
      When we trained all four tasks together (for 1,600,000 steps, scaled proportionally for fairness),
      every task grokked dramatically faster than the baseline:
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>1st — Multiplication</h3>
        <p>Grokked at <strong>4.3%</strong> training progress<br>(68,250 steps)</p>
      </div>
      <div class="card">
        <h3>2nd — Division</h3>
        <p>Grokked at <strong>4.7%</strong> training progress<br>(75,750 steps)</p>
      </div>
      <div class="card">
        <h3>3rd — Addition</h3>
        <p>Grokked at <strong>6.5%</strong> training progress<br>(104,150 steps)</p>
      </div>
      <div class="card">
        <h3>4th — Subtraction</h3>
        <p>Grokked at <strong>7.8%</strong> training progress<br>(124,650 steps)</p>
      </div>
    </div>
    <figure class="figure">
<<<<<<< HEAD
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/task_diversity_plots/div_add_sub_mult.png"
=======
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/tommy_plots/4_task_split.png"
>>>>>>> tommy-web
           alt="4-task training result">
      <figcaption>Figure 2 — All four tasks trained simultaneously. The black curve shows the baseline (division only). All four tasks grok well before the baseline, with multiplication and division leading the way.</figcaption>
    </figure>
    <p class="section-intro">
      Not every task combination accelerates grokking equally, however. Examining two-task combinations,
      we found a striking pattern: pairing division with multiplication produced the
      <strong>fastest grokking across all two-task experiments</strong> — with both tasks grokking
      at just ~0.7% of training progress. By contrast, pairing division with addition or subtraction
      actually <em>slowed</em> grokking compared to the baseline. We hypothesize this is because
      division and multiplication share a deep multiplicative group structure that addition and
      subtraction do not — making their joint training especially synergistic.
    </p>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/task_diversity_plots/div_mult.png"
           alt="Div + Mult task combination">
      <figcaption>Figure 3 — Division + Multiplication (2-task). Both tasks grok at just ~0.7% training progress, the fastest result across all task combination experiments.</figcaption>
    </figure>
  </div>

  <!-- EXPERIMENT 2: OPTIMIZER -->
  <div class="section">
    <h2 class="section-title">Experiment 2: SGD as a Generalization Catalyst</h2>
    <p class="section-intro">
      Our second strategy: replace AdamW with SGD. SGD introduces more gradient noise —
      essentially making the optimizer less smooth and more likely to escape the trap of memorization
      it gets stuck in. Think of it as shaking the model loose from a local trap.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>SGD LR=0.01 — Unstable</h3>
        <p>Grokked at step <strong>16,950</strong> (~22× faster than baseline)
        but catastrophically collapsed immediately after. Not usable in practice.</p>
      </div>
      <div class="card">
        <h3>SGD LR=0.005 — Stable ✓</h3>
        <p>Grokked at step <strong>44,900</strong> (~8× faster than baseline).
        Final validation accuracy: <strong>86.32%</strong>. A genuine, stable speedup.</p>
      </div>
    </div>
    <figure class="figure">
      <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/tommy_plots/sgd_final.png"
           alt="SGD LR=0.005: stable grokking">
      <figcaption>Figure 4 — SGD (LR=0.005). Stable grokking at 44,900 steps with final validation accuracy of 86.32% — roughly 8× faster than the AdamW baseline.</figcaption>
    </figure>
  </div>

  <!-- EXPERIMENT 3: INITIALIZATION -->
  <div class="section">
    <h2 class="section-title">Experiment 3: Constrained Initialization</h2>
    <p class="section-intro">
      Our third — and most dramatic — strategy targets the root cause of grokking directly.
      Standard initialization gives the model abundant capacity right from the start,
      making it easy to memorize training data without learning anything generalizable.
      What if we took that option away?
    </p>
    <div class="callout">
      <strong>Hypothesis:</strong> If the model starts with very limited capacity — sparse or tiny weights —
      it never has room to build a memorization shortcut. Instead, it is forced to find a
      generalizing solution from the very beginning.
    </div>
    <div class="card-grid">
      <div class="card">
        <h3>Sparse Init (90% zeros)</h3>
        <p>90% of weights zeroed at initialization.<br>
        Grokking delay: <strong>1,050 steps</strong><br>
        Final validation: <strong>99.68%</strong></p>
      </div>
      <div class="card">
        <h3>Small Init (scale=0.01)</h3>
        <p>All weights initialized at a tiny uniform scale.<br>
        Grokking delay: <strong>1,050 steps</strong><br>
        Final validation: <strong>73.75%</strong></p>
      </div>
    </div>
    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
      <figure class="figure" style="flex: 1; min-width: 280px;">
        <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/tommy_plots/init_sparse.png"
             alt="Sparse initialization result" style="max-width: 100%;">
        <figcaption>Figure 5 — Sparse Init (sparsity=0.9). Grokking delay reduced to just 1,050 steps with 99.68% final validation accuracy.</figcaption>
      </figure>
      <figure class="figure" style="flex: 1; min-width: 280px;">
        <img src="https://raw.githubusercontent.com/jetyue04/loss-plateau/main/grokking/tommy_plots/init_small.png"
             alt="Small initialization result" style="max-width: 100%;">
        <figcaption>Figure 6 — Small Init (scale=0.01). Also grokked at 1,050 steps, achieving 73.75% final validation accuracy.</figcaption>
      </figure>
    </div>
    <div class="callout">
      <strong>Result:</strong> Both strategies reduced the grokking delay from ~332,000 steps
      to just <strong>1,050 steps</strong> — a <strong>~316× speedup</strong> that essentially
      eliminates the memorization plateau entirely.
    </div>
  </div>
  
  <!-- SUMMARY -->
  <div class="section">
    <h2 class="section-title">Summary & Takeaways</h2>
    <p class="section-intro">
      Across three experiments, we demonstrated that the grokking delay is not a fixed property
      of the problem — it can be dramatically shortened through targeted interventions.
      A common thread runs through all three strategies: grokking is accelerated whenever
      the model is pushed away from high-capacity memorization and toward structured,
      generalizable representations.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Task Diversity</h3>
        <p>4-task training (all operations)<br>
        83.5% → 4.7% training progress for Division<br>
        <strong>~18× speedup</strong></p>
      </div>
      <div class="card">
        <h3>SGD Optimizer</h3>
        <p>LR=0.005, stable run<br>
        332,000 → 41,500 step delay<br>
        <strong>~8× speedup</strong></p>
      </div>
      <div class="card">
        <h3>Initialization</h3>
        <p>Sparse or small-scale init<br>
        332,000 → 1,050 step delay<br>
        <strong>~316× speedup</strong></p>
      </div>
    </div>
    <p class="section-intro">
      There are open questions worth exploring further. Our experiments used a fixed model size
      and a single prime modulus (97) — it remains unclear whether these results hold for larger
      models or different settings. Our hypothesis about shared algebraic structure driving
      multi-task acceleration is also still qualitative; future work applying interpretability
      tools (or tools that let researchers inspect what a model has actually learned internally) could provide direct
      mechanistic evidence. Finally, exploring task diversity beyond modular arithmetic could
      reveal whether these acceleration effects are specific to our setting or more broadly applicable.
    </p>
  </div>

  <!-- REFERENCES -->
  <div class="section">
    <h2 class="section-title">References</h2>
    <p class="section-intro">
      Power et al. (2022). <a href="https://arxiv.org/abs/2201.02177" target="_blank">Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.</a> <em>arXiv:2201.02177</em>.<br><br>
Lyu, Jin, Li, Du, Lee &amp; Hu (2024). <a href="https://arxiv.org/abs/2311.02058" target="_blank">Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking.</a> <em>ICLR 2024</em>.<br><br>
Kim et al. (2025). <a href="https://arxiv.org/abs/2501.19512" target="_blank">Task Diversity Shortens the ICL Plateau.</a> <em>arXiv preprint</em>.<br><br>
Lee et al. (2024). <a href="https://arxiv.org/abs/2405.20233" target="_blank">Grokfast: Accelerated Grokking by Amplifying Slow Gradients.</a> <em>arXiv:2405.20233</em>.
    </p>
  </div>
</div>
