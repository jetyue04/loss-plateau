---
layout: default
title: Home
---

<div class="hero">
  <h1>Shortening the Loss Plateau</h1>
  <div class="hero-links">
    <a class="btn btn-primary" href="https://github.com/jetyue04/loss-plateau" target="_blank">GitHub Repository</a>
    <a class="btn btn-secondary" href="{{ site.baseurl }}/grokking/">Grokking →</a>
    <a class="btn btn-secondary" href="{{ site.baseurl }}/loss-plateau/">Loss Plateau →</a>
  </div>
</div>

<div class="main-content">

  <div class="section">
    <h2 class="section-title">Core Question</h2>
    <p class="section-intro">
      Transformer models often spend enormous amounts of compute stuck in inefficient
      training regimes. These stalls appear in two well-known forms:
      <b>training-loss plateaus</b> and <b>grokking delays</b>.
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
      <div class="card">
        <h3>Our Approach</h3>
        <p>
          We test whether shared interventions can accelerate both phenomena.
          Specifically, we study task diversity, optimizer noise (SGD vs AdamW),
          and initialization constraints in modular arithmetic transformers.
        </p>
      </div>
    </div>
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
        <h3>Representation Collapse</h3>
        <p>
        Early training exhibits embedding repetition and limited attention
        structure, causing gradients to produce minimal improvements in loss.
        </p>
      </div>
      <div class="card">
        <h3>Slow Attention Formation</h3>
        <p>
        Useful attention patterns emerge gradually, delaying the model's ability
        to exploit structure in modular arithmetic tasks.
        </p>
      </div>
    </div>
    <h3 class="subsection-title">Grokking Acceleration</h3>
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
    <h2 class="section-title">Team</h2>
    <div class="team-grid">
      <div class="team-card">
        <div class="avatar">J</div>
        <h4>Jet Yue</h4>
        <p>Training-Loss Plateau</p>
      </div>
      <div class="team-card">
        <div class="avatar">S</div>
        <h4>Samuel</h4>
        <p>Task Diversity & Grokking Dynamics</p>
      </div>
      <div class="team-card">
        <div class="avatar">T</div>
        <h4>Tommy</h4>
        <p>Optimizer & Initialization</p>
      </div>
    </div>
  </div>

</div>
