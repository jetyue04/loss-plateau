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
      In this study, we trained a shallow Transformer on modular arithmetic tasks to investigate the training-loss plateau phenomenon.
      We describe the model architecture, training procedure, and tasks used below.
    </p>
    <h3 style="margin-top:2em;">Model Architecture</h3>
    <p>
      We use a 1-layer, 1-head Transformer with causal masking and linear attention. This architecture is able to solve simple algorithmic tasks such as MWS to perfect accuracy.
    </p>
    <p style="text-align:center; margin:1em 0;">
      <code>
        TF<sub>θ</sub>(s<sub>1</sub>, …, s<sub>L</sub>) = LM ∘ (Id + MLP) ∘ (Id + Attn) ∘ Embed(s<sub>1</sub>, …, s<sub>L</sub>)
      </code>
    </p>
    <p>
      where <code>Embed</code> outputs the sum of token and absolute positional embeddings <code>h<sub>i</sub> ∈ ℝ<sup>d</sup></code>, and <code>Attn</code> denotes the causal linear attention operation:
    </p>
    <p style="text-align:center; margin:1em 0;">
      <code>
        [Attn(h<sub>1</sub>, …, h<sub>L</sub>)]<sub>i</sub> = W<sub>O</sub> ( ∑<sub>j=1</sub><sup>i</sup> (h<sub>j</sub><sup>T</sup> W<sub>K</sub><sup>T</sup> W<sub>Q</sub> h<sub>i</sub>) W<sub>V</sub> h<sub>j</sub> ), &nbsp; W<sub>O</sub>, W<sub>K</sub>, W<sub>Q</sub>, W<sub>V</sub> ∈ ℝ<sup>d×d</sup>
      </code>
    </p>
    <p>
      The <code>MLP</code> is a 2-layer feedforward network, <code>h<sub>i</sub> ↦ W<sub>2</sub>(σ(W<sub>1</sub> h<sub>i</sub>))</code>, with <code>W<sub>1</sub> ∈ ℝ<sup>4d×d</sup></code>, <code>W<sub>2</sub> ∈ ℝ<sup>d×4d</sup></code>, and σ the GELU activation. <code>LM</code> is a linear layer mapping hidden states <code>h<sub>i</sub> ∈ ℝ<sup>d</sup></code> to logits <code>v<sub>i</sub> ∈ ℝ<sup>|V|</sup></code>. All linear maps include bias terms, and pre-LayerNorm is applied before Attn, MLP, and LM. For sequence generation, we use greedy decoding, selecting the token with the maximum logit at each step.
    </p>
    <h3 style="margin-top:2em;">Training Procedure</h3>
    <p>
      Models are trained online with batches of 256 sequences drawn freshly at each step. The objective is next-token cross-entropy loss, and accuracy is measured on the generated output. For multi-task experiments, batches contain sequences from multiple tasks, with samples evenly distributed across tasks. To ensure fair comparisons, batch size, vocabulary, model architecture, and the number of examples per task are kept constant across experiments.
    </p>
    <h3 style="margin-top:2em;">Algorithmic Tasks</h3>
    <p>We evaluated several deterministic sequence-to-sequence tasks, including:</p>
    <ul style="margin-left:2em;">
      <li><strong>Moving Window Sum (MWS):</strong> y<sub>i</sub> = x<sub>1</sub> if i=1, else (x<sub>i-1</sub> + x<sub>i</sub>) mod p</li>
      <li><strong>Moving Window Product (MWP):</strong> y<sub>i</sub> = x<sub>1</sub> if i=1, else (x<sub>i-1</sub> × x<sub>i</sub>) mod p</li>
      <li><strong>Moving Window Difference (MWD):</strong> y<sub>i</sub> = x<sub>1</sub> if i=1, else (x<sub>i</sub> - x<sub>i-1</sub>) mod p</li>
      <li><strong>Prefix Sum (PS):</strong> y<sub>i</sub> = Σ<sub>j=1…i</sub> x<sub>j</sub> mod p</li>
    </ul>
    <p>
        Sequence length is n=16 and modulus p=17 for initial tasks, with task-specific separator tokens to distinguish sequences in multi-task batches. Each task is chosen such that the model can solve it to 100% accuracy.
    </p>
  </div>

  <div class="section">
  <h2 class="section-title">Experiments</h2>
  <p class="section-intro">
    We conducted three main experiments to investigate the effects of task diversity, baseline performance, and transfer learning in our Transformer setup.
  </p>

  <!-- Experiment 1 -->
  <div class="section">
    <h3>Experiment 1 — Baseline with MWS</h3>
    <p>
      The model is trained on a single algorithmic task, Moving Window Sum (MWS), using sequence length n=16 and modulus p=17. 
    </p>
    <div class="callout">
      <strong>Key insight:</strong> We observed the expected characteristics associated to the loss plateau: Representation collapse, Repetition Bias and slow formation of the optimal attention map.
    </div>
  </div>

  <!-- Experiment 2 -->
  <div class="section">
    <h3>Experiment 2 — Task Diversity</h3>
    <p>
      We train the model on multiple algorithmic tasks simultaneously (e.g., MWS, MWP, MWD, PS) using balanced batches. This setup examines whether shared representations across tasks can accelerate the loss plateau on algorithmic tasks. 
      To ensure <strong>fairness</strong>, the overall training batch size is kept constant across experiments. When multiple tasks are included in a batch, the number of training samples is divided evenly among all tasks, and each task is uniquely identified by a separate separator token. 
      Notably, the loss plateau can emerge multiple times: the model often learns one task first, then others sequentially. We measure the plateau in terms of the total number of training samples observed until each task is fully learned.
    </p>
    <div class="callout">
      <strong>Observation:</strong> Multi-task training allows the model to learn each task using fewer training samples and reduces the training loss plateau. The optimal attention map is able to form with less training samples seen. The loss also spikes as the model tries to learn several tasks.
    </div>
    <div class="card">
      <h3>Moving Window Sum (MWS)</h3>
      <figure class="figure" style="max-width:600px; margin:0 auto;">
        <img src="{{ site.baseurl }}/assets/images/MWS_diversity.png" alt="MWS Task Plateau">
        <figcaption class="figure-caption">
          The model shows moderate acceleration for MWS. A small plateau is observed initially, and the task is learned before other tasks.
        </figcaption>
      </figure>
    </div>
    <div class="card">
      <h3>Moving Window Product (MWP)</h3>
      <figure class="figure" style="max-width:600px; margin:0 auto;">
        <img src="{{ site.baseurl }}/assets/images/MWP_diversity.png" alt="MWP Task Plateau">
        <figcaption class="figure-caption">
          MWP shows the largest reduction in plateau duration. The task is learned more quickly thanks to shared multi-task representations.
        </figcaption>
      </figure>
    </div>
    <div class="card">
      <h3>Moving Window Division (MWD)</h3>
      <figure class="figure" style="max-width:600px; margin:0 auto;">
        <img src="{{ site.baseurl }}/assets/images/MWD_diversity.png" alt="MWD Task Plateau">
        <figcaption class="figure-caption">
          MWD experiences a longer plateau initially, but acceleration is significant with task diversity.
        </figcaption>
      </figure>
    </div>
    <!-- Summary Table -->
    <h4>Plateau Reduction Summary</h4>
    <p>
      The table below shows, for each task and multi-task configuration, the number of training examples
      the model sees until the plateau in loss ends. The last column reports the average % speedup 
      relative to the single-task baseline, computed as: <code>1 - (Current / Baseline)</code>.
    </p>
    <table style="width:100%; border-collapse: collapse; text-align:center;">
      <thead>
        <tr style="background-color:#f2f2f2;">
          <th style="padding:8px; border:1px solid #ccc;">#Tasks</th>
          <th style="padding:8px; border:1px solid #ccc;">MWS</th>
          <th style="padding:8px; border:1px solid #ccc;">MWP</th>
          <th style="padding:8px; border:1px solid #ccc;">MWD</th>
          <th style="padding:8px; border:1px solid #ccc;">AVG % Speedup</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">1</td>
          <td style="padding:8px; border:1px solid #ccc;">16,256</td>
          <td style="padding:8px; border:1px solid #ccc;">93,568</td>
          <td style="padding:8px; border:1px solid #ccc;">34,560</td>
          <td style="padding:8px; border:1px solid #ccc;">N/A</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">2</td>
          <td style="padding:8px; border:1px solid #ccc;">13,600</td>
          <td style="padding:8px; border:1px solid #ccc;">24,096</td>
          <td style="padding:8px; border:1px solid #ccc;">19,680</td>
          <td style="padding:8px; border:1px solid #ccc;">44.55%</td>
        </tr>
        <tr>
          <td style="padding:8px; border:1px solid #ccc;">3</td>
          <td style="padding:8px; border:1px solid #ccc;">10,542</td>
          <td style="padding:8px; border:1px solid #ccc;">13,288</td>
          <td style="padding:8px; border:1px solid #ccc;">10,794</td>
          <td style="padding:8px; border:1px solid #ccc;">63.24%</td>
        </tr>
      </tbody>
      </table>
  </div>
    <!-- Experiment 3 -->
    <div class="section">
      <h3>Experiment 3 — Transfer Learning</h3>
      <p>
        Since incorporating task diversity into training shortens the loss plateau, and allows the model to learn the task on less training samples, we examine this phenoenon by conducting transfer learning. We pretrian the model on a taask before fine-tuning on a more difficult task.
      </p>
      <div class="callout">
        <strong>Observation:</strong> Pretraining tasks shortens the loss plateau significantly. It is observed that the attention (K, Q, V) matrices remain stable while their respective bias term shifts signifcantly.
      </div>
      <h3>Same Target Function: Prefix Sum -> MWS</h3>
      <p>
        When pretraining on prefix sum and then transfering the model on MWS, the loss plateau disappears. Here, the attention map is different while the target functino is shared between the two tasks.
      </p>
      <figure class="figure" style="max-width:600px; margin:0 auto;">
        <img src="{{ site.baseurl }}/assets/images/transfer_mws.png">
      </figure>
      <div class="card-grid">
        <div class="card">
          <h4>The initial attention map for initialization</h4>
          <figure class="figure" style="max-width:600px; margin:0 auto;">
            <img src="{{ site.baseurl }}/assets/images/transfer_mws_init_attn.png">
          </figure>
        </div>
          <div class="card">
          <h4>The final attention map for initialization</h4>
          <figure class="figure" style="max-width:600px; margin:0 auto;">
            <img src="{{ site.baseurl }}/assets/images/transfer_mws_fin_attn.png">
          </figure>
        </div>
      </div>
      <p>
        Upon examination, the attention matrices weights remain stable while the bias terms shift signifanctly. The MLP weights also remain stable. This is verified by rerunning the experiment freezing the MLP layer and the same phenomena occured.
      </p>
      <h3>Same Attention Map: MWS -> MWP</h3>
      <p>
        When pretraining on MWS and finetuning on MWP, the loss plateau shortens significantly. Her, they share the same attention map but has different target functions.
      </p>
      <figure class="figure" style="max-width:600px; margin:0 auto;">
        <img src="{{ site.baseurl }}/assets/images/transfer_mws_mwp.png">
        <figcaption class="figure-caption">
          Model shows significant acceleration and shortening of the loss plateau when pretrained on MWS.
        </figcaption>
      </figure>
      <p>
        Upon examination, the attention matrices weights and weights remain stable while MLP layers shifted significantly.
      </p>
    </div>
    </div>
  </div>
</div>