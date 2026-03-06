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
    <h2 class="section-title">Overview</h2>
    <p class="section-intro">
      Transformer models often waste enormous amounts of computation getting "stuck" during training.
      This project identifies two distinct stalling phenomena and investigates how training strategy,
      optimizer choice, and initialization can dramatically accelerate learning.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>📉 Training-Loss Plateau</h3>
        <p>Early in training, loss stalls due to representation collapse and repetition bias in token embeddings.
        Attention structure forms slowly, delaying useful learning.</p>
      </div>
      <div class="card">
        <h3>⚡ Grokking</h3>
        <p>Models memorize training data almost instantly but take hundreds of thousands of steps before
        generalizing. We investigate what drives this delay and how to eliminate it.</p>
      </div>
      <div class="card">
        <h3>🎯 Our Approach</h3>
        <p>We systematically test task diversity, optimizer noise (SGD vs AdamW), and initialization
        constraints across modular arithmetic tasks with a decoder-only Transformer.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">Key Findings</h2>
    <p class="section-intro">Across both phenomena, constraining the model's initial capacity proved to be the most powerful intervention.</p>
    <div class="card-grid">
      <div class="card">
        <h3>4× Faster via Task Diversity</h3>
        <p>Training on 4 arithmetic tasks simultaneously with strictly balanced batching reduced
        Division grokking from 334,000 → ~80,000 steps.</p>
      </div>
      <div class="card">
        <h3>8× Faster via SGD</h3>
        <p>Replacing AdamW with SGD (LR=0.005) introduced gradient noise that escaped the
        memorization basin, achieving stable grokking at 44,900 steps.</p>
      </div>
      <div class="card">
        <h3>~316× Faster via Initialization</h3>
        <p>Sparse or small weight initialization virtually eliminated the grokking plateau,
        reducing the delay from ~332,000 steps to just 1,050 steps.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">Team</h2>
    <div class="team-grid">
      <div class="team-card">
        <div class="avatar">J</div>
        <h4>Jet Zhang Yue</h4>
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