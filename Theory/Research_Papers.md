# From "Following a DL Course" to "Hardcore AI Engineer" — A Paper-Driven Roadmap

**This doc gives structure and a reading list.**

---


## 1. Math/optimization you should be able to derive, not just use

- Chain rule → backprop (you have this)
- Why plain SGD struggles → **Adam** (Kingma & Ba, 2014, *"Adam: A Method for Stochastic Optimization"*) — implement Adam's update rule by hand once, don't just call `torch.optim.Adam`
- **Dropout** (Srivastava et al., 2014) and **Batch Norm** (Ioffe & Szegedy, 2015) — you've studied these conceptually; read the actual papers to see *why* they were proposed (internal covariate shift argument, later debated) — good critical-thinking exercise
- **Layer Norm** (Ba et al., 2016) — matters because it's what transformers use instead of batch norm, and knowing *why* is a common interview question

---

## 2. Classic vision architectures (read + reimplement the block, not the whole net)

Read in this order, since each solves a problem in the previous one:

1. **AlexNet** (Krizhevsky et al., 2012) — the ImageNet moment, why deep+GPU mattered
2. **VGG** (Simonyan & Zisserman, 2014) — depth via small filters
3. **ResNet** (He et al., 2015) — **the single most important idea to internalize deeply.** Skip connections show up everywhere in modern DL, including transformers. Implement a residual block from scratch and explain *why* it fixes vanishing gradients, not just that it does.

Skip full reads of LeNet/GoogLeNet — know the ideas (multi-scale conv, historical CNN), don't burn time on them.

---

## 3. Sequence models → the path to attention

This sequence is the actual intellectual history of why transformers exist — reading it in order makes the Transformer paper click instead of feeling arbitrary:

1. **LSTM** (Hochreiter & Schmidhuber, 1997) — vanishing gradient fix for RNNs
2. **Seq2Seq** (Sutskever et al., 2014) — encoder-decoder framing
3. **Bahdanau Attention** (2014) — attention *before* transformers; understand this was invented to fix a seq2seq bottleneck, not born from nowhere
4. **"Attention Is All You Need"** (Vaswani et al., 2017) — read this one multiple times. Then build a transformer from scratch — this is literally what Karpathy's **nanoGPT** walks you through. Do this next after finishing Zero to Hero; it's the highest-leverage single project you can do right now.

---

## 4. The modern LLM era — this is what actually gets you noticed in ML research/industry right now

- **BERT** (Devlin et al., 2018) — encoder-only, masked LM pretraining
- **GPT-1/2/3** (Radford et al.; Brown et al., 2020) — decoder-only, in-context learning, emergent scaling behavior
- **Scaling Laws** (Kaplan et al., 2020) and **Chinchilla** (Hoffmann et al., 2022) — compute-optimal training. This is genuinely math-heavy (power laws, loss curves as functions of params/data/compute) — should be very approachable given your math background, and it's a differentiator most undergrads skip
- **InstructGPT / RLHF** (Ouyang et al., 2022) — why raw pretrained LLMs need alignment/finetuning
- **LoRA** (Hu et al., 2021) — parameter-efficient finetuning. **Highly recommend reproducing this one** — it's implementable on modest/free-tier compute (Colab), directly relevant to "efficient finetuning" which is a hot practical skill, and pairs naturally with your hardware-constrained-systems instincts from Fire-Volt Green
- **Mixture of Experts** (Shazeer et al., 2017; Switch Transformer, 2021) — sparsity, conditional compute
- **Mamba / state space models** (Gu & Dao, 2023–24) — the current "is attention all we need, actually?" debate. Don't need to master it, but know it exists and roughly why it's interesting (linear-time sequence modeling vs. quadratic attention)

---

## 5. Vision beyond CNNs

- **ViT** (Dosovitskiy et al., 2020) — transformers applied to images (patches as tokens)
- **CLIP** (Radford et al., 2021) — contrastive vision-language pretraining, foundational to most modern multimodal systems

## 6. Generative models (lighter priority, but know the landscape)

- **VAE** (Kingma & Welling, 2013)
- **GAN** (Goodfellow et al., 2014)
- **DDPM / Diffusion** (Ho et al., 2020) — the basis for Stable Diffusion-style systems

---

## 7. How to actually read a paper (this is a skill, not automatic)

1. **Pass 1** — abstract, figures, conclusion only. Get the "what and why" in 10 minutes.
2. **Pass 2** — method section, understand the approach without chasing every proof.
3. **Pass 3** — full read with pen and paper, work through the math yourself, especially any equation you'd have to explain in an interview.
4. **Then reimplement the core idea** in under ~100 lines, even at toy scale. This step is what separates "I read the paper" from "I understand the paper," and it's exactly the habit you already have from your from-scratch projects — just point it at papers now instead of only architectures you invent for coursework.

Use **Papers With Code** to find the canonical paper + strongest follow-ups for any topic, and arXiv directly once you know what to search for.

---

## 8. Engineering skills to build alongside the papers (the "engineer," not just "reader")

- PyTorch internals: autograd mechanics, writing custom `nn.Module`s, hooks
- Efficient training: mixed precision, gradient accumulation, `DistributedDataParallel` basics
- Rough CUDA/Triton awareness — not mastery, but know what's happening under `.cuda()`
- Experiment tracking (Weights & Biases) — you're already deployment-savvy (CineMatch's FastAPI/Docker/Railway pipeline); this is the missing piece for research-style iteration

---

## 9. Concrete next 3 moves, given where you are right now

1. **Finish Zero to Hero → build nanoGPT.** This single project operationalizes sections 3 and 4 at once and is a genuinely strong portfolio piece for your professor outreach.
2. **Reimplement a ResNet block from scratch** on a small dataset (CIFAR-10 is fine) — cheap to run, cements section 2.
3. **Pick LoRA as your first "real paper reproduction."** It's practical, cheap to run, and directly demonstrates you can go from paper → working code → written explanation — exactly the signal a professor doing ML/embedded work would want to see alongside Fire-Volt Green.

Each of these, written up with a short README explaining *why* the paper's idea matters (not just that you ran the code), becomes outreach material — the same way you built out Fire-Volt Green's Technical_Overview.md.
