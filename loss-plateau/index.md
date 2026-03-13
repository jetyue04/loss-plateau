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
    We study training-loss plateaus using a small Transformer trained on simple modular arithmetic tasks.
  </p>

  <div class="card-grid">
    <div class="card">
      <h3>Model</h3>
      <p>
        A small Transformer with 1 layer and 1 attention head.
        Despite its simplicity, the model can solve several
        algorithmic sequence tasks to perfect accuracy.
      </p>
    </div>
    <div class="card">
      <h3>Training</h3>
      <p>
        The model is trained to predict the next token in a sequence
        using cross-entropy loss. Training batches are generated online,
        and multi-task experiments mix examples from several tasks.
      </p>
    </div>
    <div class="card">
      <h3>Tasks</h3>
      <p>  We selected small, deterministic sequence tasks based on modular arithmetic.
      Each task is simple enough that the model can achieve 100% accuracy,
      making it ideal to study training dynamics and the appearance of
      loss plateaus without complications from task difficulty.</p>
    </div>
  </div>
</div>
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
      Since incorporating task diversity shortens the loss plateau and allows the model to learn tasks using fewer training samples, we investigate this phenomenon via transfer learning. 
      We pretrain the model on a simpler task before fine-tuning on a more difficult task.
    </p>
    <div class="callout">
      <strong>Observation:</strong> Pretraining significantly shortens the loss plateau. 
      The attention (K, Q, V) matrices remain largely stable, while their respective bias terms shift significantly.
    </div>
    <h4>Same Target Function: Prefix Sum → MWS</h4>
    <p>
      Pretraining on Prefix Sum and then fine-tuning on Moving Window Sum (MWS) eliminates the loss plateau entirely. 
      Here, the attention map is reorganized while the target function is shared between the two tasks.
    </p>
    <figure class="figure" style="max-width:600px; margin:0 auto;">
      <img src="{{ site.baseurl }}/assets/images/transfer_mws.png" alt="Transfer learning: Prefix Sum to MWS">
      <figcaption class="figure-caption">
        Loss plateau disappears when pretraining on Prefix Sum before fine-tuning on MWS.
      </figcaption>
    </figure>
    <div class="card-grid">
      <div class="card">
        <h4>Initial Attention Map</h4>
        <figure class="figure" style="max-width:600px; margin:0 auto;">
          <img src="{{ site.baseurl }}/assets/images/transfer_mws_init_attn.png" alt="Initial attention map">
        </figure>
      </div>
      <div class="card">
        <h4>Final Attention Map</h4>
        <figure class="figure" style="max-width:600px; margin:0 auto;">
          <img src="{{ site.baseurl }}/assets/images/transfer_mws_fin_attn.png" alt="Final attention map">
        </figure>
      </div>
    </div>
    <p>
      Examination shows that attention matrix weights remain stable while bias terms shift significantly. 
      The MLP weights also remain stable. Freezing the MLP layer and rerunning the experiment reproduces the same phenomenon.
    </p>
    <div style="margin-top:40px;"></div> 
    <h4>Same Attention Map: MWS → MWP</h4>
    <p>
      Pretraining on MWS and fine-tuning on Moving Window Product (MWP) significantly shortens the loss plateau. 
      Here, the attention map remains largely unchanged, while the target functions differ between tasks.
    </p>
    <figure class="figure" style="max-width:600px; margin:0 auto;">
      <img src="{{ site.baseurl }}/assets/images/transfer_mws_mwp.png" alt="Transfer learning: MWS to MWP">
      <figcaption class="figure-caption">
        Loss plateau shortens significantly when pretrained on MWS before fine-tuning on MWP.
      </figcaption>
    </figure>
    <p>
      Analysis indicates that attention matrix weights remain stable, while the MLP layer weights shift significantly to accommodate the new target function.
    </p>
  </div>

  <div class="section">
    <h2 class="section-title">Key Findings & Takeaways</h2>
    <div class="card-grid">
      <div class="card">
        <h4>Single-Task Baseline</h4>
        <p>Training on one task (e.g., MWS) shows a long loss plateau. Attention maps form slowly, representations collapse, and repetition bias occurs early.</p>
      </div>
      <div class="card">
        <h4>Task Diversity</h4>
        <p>Training on multiple tasks simultaneously reduces the number of examples needed to exit plateaus. Tasks are often learned sequentially, and shared representations accelerate learning.</p>
      </div>
      <div class="card">
        <h4>Transfer Learning</h4>
        <p>Pretraining on one task before fine-tuning on another significantly shortens or eliminates the plateau. Attention maps remain stable if target functions differ; they reorganize if target functions are shared.</p>
      </div>
    </div>
  </div>

  <div class="section">
  <h2 class="section-title">References</h2>
  <p class="section-intro">
    Jaeyeon Kim, Sehyun Kwon, Joo Young Choi, Jongho Park, Jaewoong Cho, Jason D. Lee, Ernest K. Ryu (2025). 
    <em>Task Diversity Shortens the ICL Plateau</em>. arXiv:2410.05448. 
    <a href="https://arxiv.org/abs/2410.05448" target="_blank">https://arxiv.org/abs/2410.05448</a>.<br><br>
    Pulkit Gopalani, Wei Hu (2025). 
    <em>What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers</em>. arXiv:2506.13688. 
    <a href="https://arxiv.org/abs/2506.13688" target="_blank">https://arxiv.org/abs/2506.13688</a>.<br><br>
    Jianliang He, Xintian Pan, Siyu Chen, Zhuoran Yang (2025). 
    <em>In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention</em>. arXiv:2503.12734. 
    <a href="https://arxiv.org/abs/2503.12734" target="_blank">https://arxiv.org/abs/2503.12734</a>.<br><br>
  </p>
</div>
</div>
