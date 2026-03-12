---
layout: default
title: Home
---

<div class="hero">
  <h1>Shortening the Loss Plateau</h1>
  <p class="hero-subtitle">
    Transformer models waste enormous compute stuck in two training stalls —
    loss plateaus and grokking. We identify what causes them and show targeted
    interventions that cut grokking delay by up to <strong>316× through sparse initialization</strong>.
  </p>
  <div class="hero-links">
    <a class="btn btn-secondary" href="{{ site.baseurl }}/grokking/">Grokking →</a>
    <a class="btn btn-secondary" href="{{ site.baseurl }}/loss-plateau/">Loss Plateau →</a>
    <a class="btn btn-primary" href="{{ site.baseurl }}/assets/report.pdf" target="_blank">📄 Read the Report</a>
    <a class="btn btn-primary" href="https://github.com/jetyue04/loss-plateau" target="_blank">GitHub Repository</a>
  </div>
</div>

<div class="main-content">

  <div class="section">
    <h2 class="section-title">Core Question</h2>
    <p class="section-intro">
      Transformer models (neural networks that power modern AI systems like GPT and BERT) often spend enormous amounts of compute stuck in inefficient
      training regimes. These stalls appear in two well-known forms:
      <b>training-loss plateaus</b> and <b>grokking (generalization plateau)</b>.
    </p>
    <p>
      In this project we investigate whether these phenomena arise from similar
      optimization dynamics, and whether the same interventions can shorten
      both forms of stalled learning.
    </p>
  </div>

  <div class="section">
  <h2 class="section-title">Two Forms of Training Stall</h2>
  <div class="card-grid">
    <div class="card">
      <h3>Training-Loss Plateau</h3>
      <p>
        Early in training, loss stalls due to representation collapse and
        repetition bias in token embeddings. Attention structure forms
        slowly, delaying useful learning.
      </p>
    </div>

    <div class="card">
      <h3>Grokking (Generalization Plateau)</h3>
      <p>
        Models memorize training data almost instantly but take hundreds of
        thousands of steps before generalizing. We investigate what drives
        this delay and how to eliminate it.
      </p>
    </div>
  </div>
</div>

<div class="section">
  <h2 class="section-title">Our Approach</h2>
  <p class="section-intro">
    We test whether shared interventions can accelerate both phenomena.
  </p>

  <div class="card-grid">
    <div class="card">
      <h3>Task Diversity</h3>
      <p>
        Training on multiple arithmetic tasks simultaneously to break
        memorization dynamics.
      </p>
    </div>
    <div class="card">
      <h3>Optimizer Noise</h3>
      <p>
        Comparing SGD and AdamW to study how gradient noise influences
        escape from memorization basins.
      </p>
    </div>
    <div class="card">
      <h3>Initialization Constraints</h3>
      <p>
        Testing sparse and small-weight initialization to restrict early
        representational capacity.
      </p>
    </div>
  </div>
</div>

<div class="section">
  <h2 class="section-title">Dataset</h2>
  <p class="section-intro">
    All experiments use synthetically generated modular arithmetic data —
    expressions of the form <code>a OP b (mod p)</code> where OP is addition,
    subtraction, or division, and p = 97. This setup is standard in grokking
    research because the task is simple enough to train quickly yet complex
    enough to exhibit both memorization and generalization phases. Data is
    generated programmatically; there is no external dataset, no train/test
    leakage, and no sensitive or private information involved.
  </p>
</div>

<div class="section">
  <h2 class="section-title">Key Findings</h2>
  <p class="section-intro">
  Our experiments reveal that training stalls arise from inefficient early
  representation learning. Several interventions significantly accelerate
  training dynamics across both loss plateaus and grokking.
  </p>
  <h3 class="subsection-title">Loss Plateau Insights</h3>

  <div class="card-grid">
    <div class="card">
      <h3>Slow Representation and Attention Formation</h3>
      <p>
        The loss plateau is driven by slow development of meaningful internal
        representations. Early in training, attention maps and token embeddings
        change very slowly, limiting gradient signal quality and delaying useful
        learning.
      </p>
    </div>
    <div class="card">
      <h3>Task Diversity Accelerates Learning</h3>
      <p>
        Training on multiple arithmetic tasks simultaneously shortens the loss
        plateau. By distributing training across tasks, the model requires fewer
        samples per task while learning more generalizable representations,
        allowing loss to converge faster.
      </p>
    </div>
  </div>
  <h3 class="subsection-title" style="margin-top:40px;"> Grokking Acceleration</h3>
  <div class="card-grid">
    <div class="card">
      <h3>4× Faster via Task Diversity</h3>
      <p>Training on 4 arithmetic tasks simultaneously reduced Division grokking
      from 334,000 → ~80,000 steps.</p>
    </div>
    <div class="card">
      <h3>8× Faster via SGD</h3>
      <p>Replacing AdamW with SGD introduced gradient noise that escaped the
      memorization basin, achieving stable grokking at 44,900 steps.</p>
    </div>
    <div class="card">
      <h3>316× Faster via Initialization</h3>
      <p>Sparse or small weight initialization reduced the delay from
      ~332,000 steps to just 1,050 steps.</p>
    </div>
  </div>
</div>

<div class="section">
  <h2 class="section-title">Discussion & Implications</h2>
  <p class="section-intro">
    Our results suggest that both training-loss plateaus and grokking share a
    common root cause: the model's early representations are too unconstrained
    to learn efficiently. Interventions that restrict or guide early
    representation learning — whether through initialization, optimizer choice,
    or task diversity — consistently accelerate convergence.
  </p>
  <p>
    These findings have practical implications for anyone training transformers
    at scale: careful initialization and optimizer selection can dramatically
    reduce the compute needed to reach generalization, potentially saving
    significant GPU hours in real workloads.
  </p>

  <h3 class="subsection-title" style="margin-top:32px;">Limitations</h3>
  <p>
    All experiments are conducted on small toy arithmetic tasks (mod 97). While
    these are standard benchmarks in grokking research, we cannot guarantee
    the same speedups will transfer to large-scale language model training or
    other domains. The interaction between our interventions (e.g., combining
    SGD with sparse initialization) was not fully explored. Future work should
    validate these findings on larger models and real NLP tasks.
  </p>

  <h3 class="subsection-title" style="margin-top:32px;">What Changed Along the Way</h3>
  <p>
    Early experiments with AdamW showed that grokking delay was extremely
    sensitive to weight decay but still required hundreds of thousands of steps.
    Switching to SGD introduced gradient noise that helped escape memorization
    basins, but the biggest breakthrough came from initialization: restricting
    early representational capacity via sparse or small-weight initialization
    collapsed the grokking delay from ~332,000 steps to just 1,050 — a result
    we did not anticipate from the optimizer experiments alone.
  </p>
</div>

<div class="section">
  <h2 class="section-title">Team</h2>
  <div class="team-grid">
    <a href="https://www.linkedin.com/in/jyue04/" target="_blank" class="team-card">
      <div class="avatar">J</div>
      <h4>Jet Yue</h4>
      <p>Training-Loss Plateau</p>
    </a>
    <a href="https://www.linkedin.com/in/samuel-cho-980704299/" target="_blank" class="team-card">
      <div class="avatar">S</div>
      <h4>Samuel Cho</h4>
      <p>Task Diversity & Grokking Dynamics</p>
    </a>
    <a href="https://www.linkedin.com/in/tommy-li-76647082/" target="_blank" class="team-card">
      <div class="avatar">T</div>
      <h4>Tommy Li</h4>
      <p>Optimizer & Initialization</p>
    </a>
    </div>
  </div>  

</div>
