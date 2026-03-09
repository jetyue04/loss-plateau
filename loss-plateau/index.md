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
      Early in training, transformer models often enter a long period where
      the training loss barely decreases despite continued gradient updates.
      This phenomenon is known as the <b>training-loss plateau</b>.
    </p>
    <!-- Main Plateau Figure -->
    <figure class="figure">
      <img src="{{ site.baseurl }}/assets/images/plateau_curve.png" alt="Training loss plateau">
      <figcaption class="figure-caption">
        Training loss remains nearly constant for many iterations before suddenly decreasing.
      </figcaption>
    </figure>

  <div class="section">
    <h2 class="section-title">Three Training Phenomena</h2>
    <p class="section-intro">
      During long training runs we observed three recurring behaviors inside the
      Transformer: slow formation of attention maps, representation collapse,
      and repetition bias. These phenomena reveal how internal structure evolves
      before the model fully generalizes.
    </p>
    <!-- <div class="card-grid"> -->
      <div class="card">
        <h3>Slow Formation of Attention Maps</h3>
        <p>
          Early attention heads appear diffuse and noisy. Over many training
          steps they gradually sharpen into structured patterns that implement
          algorithmic behavior.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;>
          <img src="{{ site.baseurl }}/assets/images/attn_map.jpg" alt="Attention map formation">
          <figcaption class="figure-caption">Attention Map during loss plateau: The attention map forms slowly</figcaption>
        </figure>
      </div>
      <div class="card">
        <h3>Representation Collapse</h3>
        <p>
          Hidden representations compress during training, with token embeddings
          becoming increasingly aligned. This collapse suggests the model is
          discovering a lower-dimensional structure for the task.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;>
          <img src="{{ site.baseurl }}/assets/images/representation_collapse.jpg" alt="Representation collapse">
        <figcaption class="figure-caption">The cosine similarity of the hidden state collapses to ~1 during plateau</figcaption>
        </figure>
      </div>
      <div class="card">
        <h3>Repetition Bias</h3>
        <p>
          Early in training the model tends to repeat recently seen tokens.
          This shortcut can dominate predictions until stronger algorithmic
          structure emerges.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;>
          <img src="{{ site.baseurl }}/assets/images/repetition_bias.png" alt="Repetition bias">
          <figcaption class="figure-caption">The model initially prefers repeating recent tokens during plateau</figcaption>
        </figure>
      </div>
    <!-- </div> -->
</div>





</div>
  <!-- <div class="section">
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
  </div> -->

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