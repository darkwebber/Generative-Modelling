# Curriculum — Season 1 to Season 9

The whole series, one question per season. Each season follows the life of a modern AI model: build it, scale it,
teach it, make it think, serve it, give it senses, let it act, then look inside it.

Rules for every season:

1. One question per season. By its end, the viewer can answer it from first principles.
2. One companion model per season, trained by hand in this repo and grown episode by episode, with a playground lab
   for every episode.
3. Honest scale. When the real thing needs a GPU cluster, we show published numbers, clearly labelled, next to our own
   small-scale runs.
4. Every season opens by quoting the previous season's last line.

---

## Season 1 · Generative modelling (done, 11 episodes)

The question: *how do you turn noise into data?*

- It ends at Episode 11 (guidance) with the hook: "we handed it a number; real prompts are sentences. To draw what we
  say, a machine must first read."

## Season 2 · Machines that read: sequences and the transformer

The question: *how does a network turn text into meaning and predict the next word?*
Companion model: a tiny GPT in numpy that grows from characters to real words.

1. Tokens: characters, words, BPE; why tokenization is a compression problem
2. Embeddings: word2vec, meaning as geometry, analogies as arithmetic
3. Memory is hard: RNNs, vanishing gradients, why a fixed-size state forgets
4. Attention: a soft dictionary lookup, and softmax as a weighted average
5. Where am I? Positional encodings, sinusoids and RoPE
6. The transformer block: residual stream, LayerNorm, the MLP as key–value memory
7. Training a GPT: cross-entropy, AdamW, warmup, initialisation, the loss curve
8. Encoders, decoders, both: BERT vs GPT vs T5, masking, why decoder-only won
9. Inside the head: induction heads, attention patterns, a first look at superposition
10. Beyond quadratic attention: sparse and sliding windows, linear attention, state-space models (Mamba)
11. Hook: our GPT writes plausible nonsense. What happens with 10⁶× more of everything?

## Season 3 · Scale: pretraining

The question: *why does "bigger and more data" work, and how is it done in practice?*
Companion model: a family of our GPTs at five sizes, so we can fit our own scaling law.

1. Scaling laws: power laws in loss, compute-optimal training (Chinchilla)
2. Data: crawling, deduplication, quality filters, mixtures, tokenizer effects
3. Counting compute: FLOPs, memory, and the 6ND rule of thumb
4. Numbers that fit: fp32, bf16, fp8, and loss scaling
5. Many GPUs I: data parallelism and ZeRO
6. Many GPUs II: tensor, pipeline, and sequence parallelism
7. Mixture of experts: routing, load balancing, sparse compute
8. Keeping training stable: loss spikes, μP, learning-rate schedules, checkpoints
9. Long context: RoPE scaling, needle-in-a-haystack tests
10. What emerges: in-context learning, and whether "emergence" is a measurement artefact
11. Judging a base model: perplexity, benchmarks, contamination
12. Hook: it can complete any text, but ask it a question and it writes more questions.

## Season 4 · Teaching it to help: post-training

The question: *how does a text predictor become an assistant that follows instructions?*
Companion model: our base GPT, fine-tuned three ways.

1. Supervised fine-tuning: instruction data, chat templates, masking the prompt
2. Cheap fine-tuning: LoRA and adapters, and the low-rank intuition
3. Learning what people prefer: pairwise preferences, Bradley–Terry, reward models (Ep 4's judges return)
4. Policy gradients from scratch: REINFORCE, baselines, variance
5. RLHF: PPO and the KL leash (Ep 5's judge that teaches, grown up)
6. DPO: deriving it in one line from the RLHF objective; IPO and KTO
7. AI feedback: constitutions, self-critique, synthetic data
8. Reward hacking: Goodhart's law, sycophancy, overoptimisation curves
9. Evaluating assistants: win rates, LLM-as-judge and its biases
10. Hook: rewarding correct answers made it start writing its reasoning out.

## Season 5 · Thinking: reasoning and test-time compute

The question: *can a model get smarter by thinking longer instead of being bigger?*
Companion model: a small reasoner on arithmetic and puzzles with checkable answers.

1. Chain of thought: why writing the steps out helps, with an information argument
2. Sampling many: self-consistency and majority vote
3. Verifiers: outcome vs process reward models
4. Search: beam search, tree search, MCTS over thoughts
5. RL with verifiable rewards: GRPO from scratch
6. Test-time scaling laws: trading training compute for inference compute
7. Distilling reasoning: teaching small models from long traces
8. Failure modes: overthinking, unfaithful reasoning, reward hacking in chains of thought
9. Hook: thinking made every answer a thousand tokens long. Who pays for that?

## Season 6 · Serving it: inference engineering

The question: *why is running a model hard, and how do we make it fast and cheap?*
Companion model: our GPT served in the browser, timed step by step.

1. Anatomy of a forward pass: prefill vs decode
2. The KV cache and the memory it costs
3. The roofline: compute-bound vs memory-bound, arithmetic intensity
4. FlashAttention: IO-aware tiling
5. Batching: static, continuous, and paged attention (vLLM)
6. Quantization: int8, int4, GPTQ/AWQ, and outliers (our labs' 8-bit models, explained)
7. Speculative decoding: draft and verify, and why the output is provably unchanged
8. Distillation and pruning
9. Serving at scale: routing, caching, latency vs throughput, cost per token
10. Hook: it's fast and cheap now, but it has only ever seen text.

## Season 7 · Seeing and hearing: multimodal models

The question: *how do models understand and generate images, audio and video as fluently as text?*
Companion model: mini-CLIP and a mini-VLM on MNIST with captions; this finally closes Season 1's "ask for a 7".

1. Images as tokens: CNNs recap, ViT patches
2. Contrastive learning: CLIP and the InfoNCE loss (Ep 7's energies return)
3. Vision-language models: connectors, LLaVA-style training, interleaved inputs
4. Text-to-image, properly: text encoders plus Season 1's guidance, finally with real sentences
5. Discrete image tokens: VQ-VAE (Ep 3 returns) and autoregressive image generation
6. Diffusion Transformers: Season 1's flows on Season 2's architecture
7. Sound: spectrograms, neural codecs, speech tokens
8. Speech in and out: ASR, TTS, full-duplex voice
9. Video: space-time patches, consistency, world models
10. Unified models: one transformer, every modality
11. Hook: it can see, hear and talk. Can it do things?

## Season 8 · Acting: agents

The question: *how does a model use tools, remember, plan and act in the world?*
Companion model: a small agent in a sandboxed toy world.

1. Tool use: function calling as generation
2. Retrieval: embedding search, vector indices, RAG
3. Memory: context windows, summaries, external stores
4. Planning: ReAct, decomposition, reflection
5. Code as action: writing, running and debugging code
6. Computer use: acting in GUIs and browsers
7. Multi-agent systems: debate, delegation, orchestration
8. Training agents: RL in environments, credit assignment over long horizons
9. Evaluating agents: benchmarks, reliability, cost
10. Hook: it acts on its own. Do we actually know what it's thinking?

## Season 9 · Looking inside: interpretability and alignment (series finale)

The question: *what is really happening inside these models, and how do we make sure they do what we mean?*
Companion model: every companion model from earlier seasons, opened up.

1. Probing: linear representations, and what a direction means
2. Superposition: more features than neurons
3. Sparse autoencoders: dictionary learning (Ep 3 returns one last time)
4. Circuits: tracing a behaviour end to end
5. Steering: activation editing, and guidance's cousin
6. Specification: goals, proxies, and Goodhart at scale
7. Robustness: jailbreaks, adversarial prompts, defences
8. Honesty and deception: can we tell what a model believes?
9. Scalable oversight: debate, AI-assisted evaluation
10. Finale: the whole map, from p(x) to minds we can inspect
