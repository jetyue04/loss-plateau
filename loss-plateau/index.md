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
    <h2 class="section-title">Training Phenomena during the plateau</h2>
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
          During the early phase of training, attention heads are largely diffuse and lack structure. 
          Over many iterations, the heads gradually learn to focus on meaningful tokens, forming 
          algorithmically relevant patterns. This slow emergence indicates that the model first 
          organizes its internal computations before accurate long-range dependencies can be learned.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;">
          <img src="{{ site.baseurl }}/assets/images/attn_map.jpg" alt="Attention map formation">
          <figcaption class="figure-caption">
            Attention Map during loss plateau: The attention head patterns sharpen gradually as training progresses.
          </figcaption>
        </figure>
      </div>
      <div class="card">
        <h3>Representation Collapse</h3>
        <p>
          As training continues, hidden representations of tokens increasingly compress into a 
          lower-dimensional subspace. Token embeddings become highly correlated, effectively 
          collapsing the representational space. This phenomenon often precedes generalization, 
          suggesting that the model has discovered a compact encoding of the underlying rules.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;">
          <img src="{{ site.baseurl }}/assets/images/representation_collapse.jpg" alt="Representation collapse">
          <figcaption class="figure-caption">
            Cosine similarity of hidden states approaches 1 during plateau, reflecting compression of the representation space.
          </figcaption>
        </figure>
      </div>
      <div class="card">
        <h3>Repetition Bias</h3>
        <p>
          Early in training, the model exhibits a tendency to repeat recently seen tokens 
          rather than performing the correct transformation. This repetition bias serves 
          as a short-term shortcut that dominates predictions until the model learns more 
          robust algorithmic structure. Monitoring this bias can reveal when the model transitions 
          toward generalization.
        </p>
        <figure class="figure" style="max-width:600px; margin:0 auto;">
          <img src="{{ site.baseurl }}/assets/images/repetition_bias.png" alt="Repetition bias">
          <figcaption class="figure-caption">
            The model initially prefers repeating recent tokens during the plateau phase, before learning the true sequence pattern.
          </figcaption>
        </figure>
      </div>
    <!-- </div> -->
    <div class="section">
      <h2 class="section-title">Experimental Setup</h2>
      <p class="section-intro">
        In this study, we trained a shallow Transformer on modular arithmetic tasks to investigate the grokking phenomenon. 
        We describe the model architecture, training procedure, and tasks used below.
      </p>
      <h3>Model Architecture</h3>
      <p>
        We use a single-layer Transformer with causal attention and a two-layer feedforward MLP. 
        Tokens are embedded and summed with absolute positional embeddings. Pre-LayerNorm is applied before each attention and MLP block, and residual connections surround both blocks. The MLP uses GELU activations with an intermediate dimensionality four times the hidden size. A final linear layer maps hidden states to vocabulary logits for next-token prediction. 
        Sequence generation uses greedy decoding, selecting the highest-logit token at each step.
      </p>
      <p>
        Formally, for a sequence of tokens \(s_1, \dots, s_L\), the model computes:
      </p>
      <p style="text-align:center;">
        \[
        \text{TF}_\theta(s_1, \dots, s_L) = 
        \text{LM} \circ (\text{Id} + \text{MLP}) \circ (\text{Id} + \text{Attn}) \circ \text{Embed}(s_1, \dots, s_L),
        \]
      </p>
      <p>
        where <code>Embed</code> produces token plus positional embeddings, <code>Attn</code> is the causal linear attention operation, and <code>LM</code> is the output projection to the vocabulary.
      </p>
      <h3>Training Procedure</h3>
      <p>
        Models are trained online with batches of 256 sequences drawn freshly at each step. 
        The objective is next-token cross-entropy loss, and accuracy is measured on the generated output. 
        For multi-task experiments, batches contain sequences from multiple tasks, with samples evenly distributed across tasks. To ensure fair comparisons, batch size, vocabulary, model architecture, and the number of examples per task are kept constant across experiments.
      </p>
      <h3>Algorithmic Tasks</h3>
      <p>
        We evaluated several deterministic sequence-to-sequence tasks, including:
      </p>
      <ul>
        <li><strong>Moving Window Sum (MWS):</strong> \(y_i = x_1\) if \(i=1\), else \((x_{i-1}+x_i) \bmod p\).</li>
        <li><strong>Moving Window Product (MWP):</strong> \(y_i = x_1\) if \(i=1\), else \((x_{i-1} \cdot x_i) \bmod p\).</li>
        <li><strong>Moving Window Difference (MWD):</strong> \(y_i = x_1\) if \(i=1\), else \((x_i - x_{i-1}) \bmod p\).</li>
        <li><strong>Prefix Sum (PS):</strong> \(y_i = \sum_{j=1}^{i} x_j \bmod p\).</li>
      </ul>
      <p>
        Sequence length is \(n=16\) and modulus \(p=17\) for initial tasks, with task-specific separator tokens to distinguish sequences in multi-task batches. 
        For experiments focused on grokking, we expand to modular division, addition, subtraction, and multiplication with modulus \(p=97\).
      </p>
    </div>

  <div class="section">
    <h2 class="section-title">Results</h2>
    <p class="section-intro">
      This section is being updated by Jet. Check back soon.
    </p>
  </div>
</div>