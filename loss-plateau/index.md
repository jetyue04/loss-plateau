---
layout: default
title: Loss Plateau
---

<div class="page-nav">
  <a href="{{ site.baseurl }}/" class="btn btn-secondary">← Home</a>
  <a href="{{ site.baseurl }}/grokking/" class="btn btn-secondary">← Grokking</a>
</div>

<div class="hero">
  <h1>Training-Loss Plateau</h1>
  <p>Investigating why Transformers stall early in training — and how task diversity shortens the wait.</p>
</div>

<div class="main-content">

  <div class="section">
    <h2 class="section-title">What is the Training-Loss Plateau?</h2>
    <p class="section-intro">
      Before grokking even occurs, Transformers exhibit a separate stalling phenomenon
      early in training. Loss stops decreasing for a prolonged period despite continued
      gradient updates — wasting significant computation.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Representation Collapse</h3>
        <p>Token embeddings early in training suffer from repetition bias — they fail to
        form distinct, meaningful representations, causing the loss to stall.</p>
      </div>
      <div class="card">
        <h3>Slow Attention Formation</h3>
        <p>Attention maps form very slowly during the plateau phase. The model cannot
        route information effectively until structure emerges.</p>
      </div>
      <div class="card">
        <h3>Multi-Task Shortcut</h3>
        <p>Training on diverse tasks forces the model to build shared representations
        faster, significantly shortening the plateau duration.</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2 class="section-title">Methods</h2>
    <p class="section-intro">
      This section is being updated by Jet. Check back soon.
    </p>
  </div>

  <div class="section">
    <h2 class="section-title">Results</h2>
    <p class="section-intro">
      This section is being updated by Jet. Check back soon.
    </p>
  </div>

</div>