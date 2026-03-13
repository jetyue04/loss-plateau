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

  <!-- ELEVATOR PITCH -->
  <div class="section">
    <p>
      AI models sometimes memorize training data perfectly but still can't answer new questions for a very long time.
      This phenomenon is called <em>grokking</em>. We tested three strategies to shorten this delay,
      achieving speedups ranging from <strong>8x to 316x</strong>.
    </p>
    <hr style="margin: 1.5em 0; border: none; height: 1px; background: linear-gradient(to right, transparent, rgba(0,0,0,0.2), transparent);">
    <p>
      All experiments use a small model on controlled arithmetic tasks. We cannot guarantee these results
      transfer to large-scale language models; generalization to other domains is left for future work.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>What we did</h3>
        <p>Trained a small AI model on simple math problems and tested three strategies to help it generalize faster: training on more varied problems, switching training algorithms, and changing how the model starts.</p>
      </div>
      <div class="card">
        <h3>What we didn't do</h3>
        <p>We did not test large models, real-world tasks, or practical applications. Results are specific to small models on a controlled arithmetic setting used here as a research tool.</p>
      </div>
      <div class="card">
        <h3>Our contributions</h3>
        <p>We built the full training pipeline from scratch and ran original experiments on task diversity. Other experiments extend prior research (Power et al., 2022; Lyu et al., 2024) with new results.</p>
      </div>
    </div>
    <p style="text-align:center; margin-top: 1rem;">
      <a href="https://github.com/jetyue04/loss-plateau" class="btn btn-secondary" target="_blank">GitHub Repository</a>
      &nbsp;
      <a href="{{ site.baseurl }}/assets/report.pdf" class="btn btn-secondary" target="_blank">Full Report</a>
    </p>
  </div>

  <!-- WHAT IS GROKKING -->
  <div class="section">
    <h2 class="section-title">What is Grokking?</h2>
    <p>
      Imagine a student who studies for a test by memorizing every answer in a textbook. At first, they would fail
      any test that asks new questions, because they memorized answers, not concepts. But after reviewing the material
      long enough, something clicks: they truly <em>understand</em> the underlying ideas and can answer any version of the test.
    </p>
    <p>
      Grokking is the AI equivalent of this experience. First described by Power et al. (2022), it refers to a
      two-phase learning pattern where a model nails the training examples almost immediately, then appears completely
      stuck before suddenly generalizing perfectly to new examples.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Phase 1: Memorization</h3>
        <p>The model quickly learns to answer every training question correctly, but only by remembering them like a lookup table. Ask it anything new and it fails. This happens very early in training.</p>
      </div>
      <div class="card">
        <h3>Phase 2: Generalization</h3>
        <p>Much later, something shifts internally. The model stops relying on memorized answers and starts understanding the underlying pattern. Accuracy on new questions then jumps sharply.</p>
      </div>
    </div>
    <p>
      The gap between Phase 1 and Phase 2 can be enormous, sometimes hundreds of thousands of additional training steps.
      Training AI is expensive, so shortening this gap has real practical value.
    </p>
    <div class="callout">
      <strong>Our baseline:</strong> Our model memorized the training data almost immediately, but needed
      <strong>334,000 total training steps</strong> before it could correctly answer new questions.
    </div>
    <figure class="figure">
      <img src="{{ site.baseurl }}/assets/images/grokking_plot.png"
           alt="Line chart showing training accuracy reaching near 100% almost immediately, while accuracy on new questions stays near 0% for a very long time before suddenly jumping to 100% near the end of training.">
      <figcaption>Figure 1 - Baseline results (division only). The orange dashed line shows how well the model does on training questions: it memorizes them almost instantly. The red line shows accuracy on new questions, which stays near 0% for the vast majority of training before the sudden jump. The x-axis is stretched so the enormous gap is visible; on a regular scale, the transition would appear as a tiny sliver at the far right.</figcaption>
    </figure>
  </div>

  <!-- WHY DOES IT HAPPEN -->
  <div class="section">
    <h2 class="section-title">Why Does Grokking Happen?</h2>
    <p>
      When an AI model trains, it is constantly trying to reduce its mistakes. Early on, the easiest way to do this
      is to memorize: remember exactly what answer goes with each training question. This works perfectly for the
      training set, but teaches the model nothing useful about the underlying pattern.
    </p>
    <p>
      Over time, a training technique called <strong>weight decay</strong>, which gently discourages the model from
      growing overly complicated, slowly pushes it toward a simpler, more general solution. When the model finally
      finds that simpler solution, generalization happens suddenly and dramatically.
    </p>
    <p>
      In short: the model takes the easy route first (memorization), and only later, under pressure, finds the right
      route (understanding). Our goal was to find ways to make that transition happen faster.
    </p>
  </div>

  <!-- EXPERIMENTAL SETUP -->
  <div class="section">
    <h2 class="section-title">Our Approach</h2>
    <p>
      We used <strong>clock (modular) arithmetic</strong> as our testing ground: a type of math where numbers wrap around after
      reaching a limit, just like a clock that resets after 12. For example, 10 + 5 on a 12-hour clock is 3, not 15.
      Our version wraps around at 97. The model's job is to predict the correct wrap-around result given two numbers
      and an operation.
    </p>
    <hr style="margin: 1.5em 0; border: none; height: 1px; background: linear-gradient(to right, transparent, rgba(0,0,0,0.2), transparent);">
    <p>
      All results are reported as <strong>training progress %</strong>, how far through training the model was when it
      generalized, rather than raw step counts. This makes it easy to compare experiments that ran for different lengths.
      We define "generalized" as when the model first answers at least 95% of new questions correctly, following the
      standard used in prior work. Small differences below ~1% training progress should be interpreted cautiously.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>The Model</h3>
        <p>A small Transformer (the same family of architecture behind ChatGPT) with around 400,000 internal connections, compared to billions in production systems.</p>
      </div>
      <div class="card">
        <h3>The Training Algorithm</h3>
        <p>By default we used AdamW, a popular and stable training algorithm. One experiment swapped this for SGD (Stochastic Gradient Descent), a simpler and noisier alternative.</p>
      </div>
      <div class="card">
        <h3>The Tasks</h3>
        <p>Four arithmetic operations using wrap-around math:<br>
        Division (hardest to learn)<br>
        Multiplication<br>
        Addition<br>
        Subtraction</p>
      </div>
    </div>
  </div>

  <!-- EXPERIMENT 1: TASK DIVERSITY -->
  <div class="section">
    <h2 class="section-title">Experiment 1: Task Diversity</h2>
    <p>
      Our first strategy: train on multiple types of arithmetic at the same time rather than just one. A model trained
      on all operations at once cannot rely on memorizing shortcuts for any single task. It has to find the deeper
      pattern they all share.
    </p>
    <div class="callout">
      <strong>Key observation:</strong> Not all tasks are equally hard to generalize. When trained alone, addition generalizes at ~14% training progress, multiplication at ~16%, subtraction at ~37%, but division takes until <strong>83.5%</strong>. Division is the hardest because it requires computing a mathematical inverse, a more complex operation than the others.
    </div>
    <h3 class="subsection-title">Training all four tasks together</h3>
    <p>
      We scaled total training time so each task received the same amount of training examples as in a single-task run.
      Every task generalized dramatically faster:
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>1st: Multiplication</h3>
        <p>Generalized at <strong>4.3%</strong> training progress (68,250 steps)</p>
      </div>
      <div class="card">
        <h3>2nd: Division</h3>
        <p>Generalized at <strong>4.7%</strong> training progress (75,750 steps)</p>
      </div>
      <div class="card">
        <h3>3rd: Addition</h3>
        <p>Generalized at <strong>6.5%</strong> training progress (104,150 steps)</p>
      </div>
      <div class="card">
        <h3>4th: Subtraction</h3>
        <p>Generalized at <strong>7.8%</strong> training progress (124,650 steps)</p>
      </div>
    </div>
    <figure class="figure">
      <img src="{{ site.baseurl }}/assets/images/div_add_sub_mult.png"
           alt="Line chart showing four colored accuracy curves all reaching 95% before 10% training progress, compared to a black baseline curve that does not reach 95% until 83.5%.">
      <figcaption>Figure 2 - All four tasks trained together. The black curve is the single-task baseline (division only). Every colored curve crosses the 95% threshold well before 10% training progress, compared to the baseline's 83.5%. Training on variety pushed the model to understand rather than memorize.</figcaption>
    </figure>
    <h3 class="subsection-title">Two-task combinations</h3>
    <p>
      Not every pairing helped equally. Pairing division with multiplication was by far the best result: both tasks
      generalized at just ~0.7% training progress, a <strong>~119x speedup</strong>. Pairing division with addition
      or subtraction actually made things <em>worse</em> than training on division alone.
    </p>
    <p>
      We think this is because division and multiplication are mathematically similar (both involve a kind of inverse
      operation) while addition and subtraction work differently. When two tasks are similar enough, the model finds
      shared patterns that help both. When they are too different, they may pull the model in conflicting directions.
    </p>
    <figure class="figure">
      <img src="{{ site.baseurl }}/assets/images/div_mult.png"
           alt="Line chart showing division and multiplication both reaching 95% accuracy at around 0.7% training progress, compared to a black baseline reaching 95% at 83.5%.">
      <figcaption>Figure 3 - Division and Multiplication trained together. Both generalize at just ~0.7% training progress, roughly 119x faster than division alone. This was the fastest result across all task combination experiments.</figcaption>
    </figure>
    <h3 class="subsection-title">What we learned along the way</h3>
    <p>
      In early runs, randomly mixing tasks in each training batch caused one task to dominate and hurt the others.
      We fixed this by scaling total training time with the number of tasks, ensuring each task always received
      equal representation.
    </p>
  </div>

  <!-- EXPERIMENT 2: OPTIMIZER -->
  <div class="section">
    <h2 class="section-title">Experiment 2: Introducing Noise</h2>
    <p>
      Our second strategy: swap the default training algorithm for a noisier one. AdamW carefully smooths each
      update to keep training stable. SGD is less careful: its updates are rougher and less predictable. That
      roughness can help shake the model out of the memorization trap, much like jostling a stuck drawer can
      suddenly free it.
    </p>
    <div class="card-grid">
      <div class="card">
        <h3>Too much noise: Unstable</h3>
        <p>The model generalized very quickly but then immediately fell apart. Too much noise is destabilizing and not usable in practice.</p>
      </div>
      <div class="card">
        <h3>Moderate noise: Stable</h3>
        <p>Generalized at step <strong>44,900</strong>, roughly <strong>8x faster</strong> than the baseline. Final accuracy: <strong>86%</strong>. A real speedup, but with a tradeoff.</p>
      </div>
    </div>
    <p>
      <strong>The tradeoff:</strong> The stable run was 8x faster, but final accuracy capped at ~86% rather than ~100%.
      The same roughness that helped escape memorization also prevented the model from fully converging later on.
      Whether this tradeoff is worth it depends on whether speed or accuracy matters more for a given use case.
    </p>
    <figure class="figure">
      <img src="{{ site.baseurl }}/assets/images/sgd_final.png"
           alt="Line chart showing the noisy training algorithm reaching 95% accuracy at step 44,900 but leveling off around 86% final accuracy, compared to the default algorithm reaching 95% at step 334,000.">
      <figcaption>Figure 4 - Moderate noise (SGD, LR=0.005). Generalization happens 8x faster than the baseline, but final accuracy levels off at ~86% rather than ~100%. Notice the validation curve plateaus rather than continuing to climb.</figcaption>
    </figure>
  </div>

  <!-- EXPERIMENT 3: INITIALIZATION -->
  <div class="section">
    <h2 class="section-title">Experiment 3: Starting Small</h2>
    <p>
      Our third strategy: limit the model's capacity right from the start. Normally, a model begins training with
      all its internal connections active, giving it plenty of room to build a memorization circuit. What if we
      dramatically reduced that starting capacity?
    </p>
    <div class="callout">
      <strong>The idea:</strong> If the model starts with very few active connections, it cannot afford to memorize. It has to find the most efficient, general solution right away. Think of it like giving a student a tiny notecard instead of a full
      textbook: they are forced to write down only the key concepts, not every answer.
    </div>
    <div class="card-grid">
      <div class="card">
        <h3>Sparse start (90% inactive)</h3>
        <p>90% of connections set to zero at the start.<br>
        Generalization delay: <strong>1,050 steps</strong><br>
        Final accuracy: <strong>99.68%</strong></p>
      </div>
      <div class="card">
        <h3>Tiny-scale start</h3>
        <p>All connections initialized at a very small value.<br>
        Generalization delay: <strong>1,050 steps</strong><br>
        Final accuracy: <strong>73.75%</strong></p>
      </div>
    </div>
    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
      <figure class="figure" style="flex: 1; min-width: 280px;">
        <img src="{{ site.baseurl }}/assets/images/init_sparse.png"
             alt="Line chart showing both training and validation accuracy reaching near 100% within 1,050 steps, with almost no gap between them." style="max-width: 100%;">
        <figcaption>Figure 5 - Sparse start (90% inactive). Generalizes at just 1,050 steps with 99.68% final accuracy. Training and validation curves are nearly identical: the memorization phase has essentially vanished.</figcaption>
      </figure>
      <figure class="figure" style="flex: 1; min-width: 280px;">
        <img src="{{ site.baseurl }}/assets/images/init_small.png"
             alt="Line chart showing accuracy on new questions reaching 95% at 1,050 steps but leveling off around 73.75% final accuracy." style="max-width: 100%;">
        <figcaption>Figure 6 - Tiny-scale start. Also generalizes at 1,050 steps, but final accuracy levels off at 73.75%, suggesting that starting too small may limit how much the model can ultimately learn.</figcaption>
      </figure>
    </div>
    <div class="callout">
      Both approaches reduced the generalization delay from ~332,000 steps to just <strong>1,050 steps</strong>,
      a <strong>~316x speedup</strong>. The sparse start also maintained near-perfect final accuracy, making it
      the strongest overall result we found.
    </div>
    <p>
      <strong>Open Question:</strong> It is unclear whether starting sparse would still work for larger, more powerful models. Those models may genuinely need their full capacity to learn at all. Testing this is an important direction for future work.
    </p>
  </div>

  <!-- SUMMARY -->
  <div class="section">
    <h2 class="section-title">Summary &amp; Takeaways</h2>
    <p>
      Across three experiments, we showed that the delay between memorization and generalization is not a fixed
      feature of how AI learns. It can be dramatically shortened. All three strategies share a common thread:
      they work by preventing the model from over-committing to memorization in the first place.
    </p>
    <div class="card-grid"> 
      <div class="card">
        <h3>Task Diversity</h3>
        <p>Train on all four operations at once.<br>
        83.5% to 4.7% progress for Division.<br>
        <strong>~18x speedup.</strong> Full accuracy maintained.</p>
      </div>
      <div class="card">
        <h3>Noisier Training</h3>
        <p>Swap training algorithm to SGD.<br>
        332,000 to 41,500 step delay.<br>
        <strong>~8x speedup.</strong> Final accuracy capped at ~86%.</p>
      </div>
      <div class="card">
        <h3>Sparse Start</h3>
        <p>Begin with 90% of connections inactive.<br>
        332,000 to 1,050 step delay.<br>
        <strong>~316x speedup.</strong> Full accuracy maintained.</p>
      </div>
    </div>
    <h3 class="subsection-title" style="margin-top: 2rem;">Why does this matter?</h3>
    <p>
      Training AI models is slow and expensive. Anything that helps models generalize faster without sacrificing
      accuracy has direct practical value. Our results point to two particularly promising levers: training on
      diverse, related tasks and starting with a constrained model. Both are simple to apply and produce large
      speedups in our setting. Whether these benefits carry over to larger, real-world models is the key open question.
    </p>
    <h3 class="subsection-title">Limitations</h3>
    <p>
      Our experiments used a small, controlled research setting with a fixed model size and a single type of
      wrap-around arithmetic. We cannot say whether the same strategies would work for large AI models used in
      real-world products. Our explanation for why task diversity helps is a hypothesis we did not directly verify
      by examining the model's internals. And our 95% accuracy threshold for defining "generalized" is a
      convention: small differences in timing between experiments should not be over-interpreted.
    </p>
  </div>

  <!-- REFERENCES -->
  <div class="section">
    <h2 class="section-title">References</h2>
    <p>
      Power et al. (2022). <a href="https://arxiv.org/abs/2201.02177" target="_blank">Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.</a> <em>arXiv:2201.02177</em>.<br><br>
      Lyu, Jin, Li, Du, Lee &amp; Hu (2024). <a href="https://arxiv.org/abs/2311.18817" target="_blank">Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking.</a> <em>ICLR 2024</em>.<br><br>
      Kim et al. (2025). <a href="https://arxiv.org/abs/2410.05448" target="_blank">Task Diversity Shortens the ICL Plateau.</a> <em>arXiv preprint</em>.<br><br>
      Lee et al. (2024). <a href="https://arxiv.org/abs/2405.20233" target="_blank">Grokfast: Accelerated Grokking by Amplifying Slow Gradients.</a> <em>arXiv:2405.20233</em>.
    </p>
  </div>

</div>