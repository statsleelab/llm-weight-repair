# Blind Weight Repair

**Can a frontier LLM repair a damaged language model by editing its weights directly, without ever seeing the original?**

Report page (English / 한국어): https://statsleelab.github.io/llm-weight-repair/

This is a follow-up to [statsleelab/weight_space_programming](https://github.com/statsleelab/weight_space_programming). There, an LLM read and wrote the weights of small networks. Here the target is a real language model, and the task is repair.

## The experiment

1. **Damage.** Take GPT-2 (124M). Corrupt 1.0% of the entries of every 2-D weight matrix with a fixed seed. Four conditions:

   | Case | Pattern | Value | Held-out perplexity after damage (clean: 28.9) |
   |---|---|---|---|
   | A | scattered random entries | set to 0 | ~10³⁰ |
   | B | scattered random entries | + N(0, (10σ)²) noise | ~10¹⁶² |
   | C | one rectangle per matrix (10% rows × 10% cols) | set to 0 | 395.7 |
   | D | one rectangle per matrix | overwritten with N(0, σ²) | 43.9 |

2. **Blind repair.** Four Claude models (Fable, Opus, Sonnet, Haiku) each get only the damaged checkpoint, a small GPT-2 implementation, and 60 kB of development text. They are told that about 1% of the matrix entries are wrong, and nothing else. The rules ([`agent_kit/TASK.md`](agent_kit/TASK.md)): no original or pretrained weights, no network, no gradient training or loss-driven optimisation. Statistics, linear algebra and forward passes are allowed.

3. **Held-out scoring.** The clean weights, the damage masks and all test data were encrypted while the agents ran. Afterwards every repaired checkpoint was scored on WikiText-2 test perplexity, BLiMP (67 paradigms × 50 pairs) and an 80-item capital-city recall test, and compared entry by entry with the clean weights.

## Results

Held-out WikiText-2 perplexity (lower is better; undamaged model 28.93). One run per cell.

| Repair by | A · scattered erasure | B · scattered noise | C · block erasure | D · block noise |
|---|---|---|---|---|
| **Fable** | 30.4 | **81.0** | **30.8** | **30.5** |
| **Opus** | **30.2** | 1,507 | 30.9 | 38.4 |
| **Sonnet** | 36.5 | ~10⁴⁷ | 31.0 | 43.7 |
| **Haiku** | 51.1 | 29,839 | 30.9 | 43.9 (unchanged) |
| Script: zeros → column mean (added after the runs) | 31.4 | no effect | 31.7 | no effect |
| Script: pre-registered outlier + zero heuristic | 10,201 | 74,839 | 15,024 | 19,497 |
| Reference: told the damage mask, column-mean fill | 31.4 | 30.7 | 31.7 | 30.9 |
| Damaged, no repair | ~10³⁰ | ~10¹⁶² | 395.7 | 43.9 |

All metrics, the scripted baselines and the weight-space comparison are on the report page and in [`results/results.json`](results/results.json).

What we take from it:

- **Partly yes.** In three of four conditions the best model restored perplexity to within about 6% of the undamaged model. Scattered noise (B) stayed 2.8× too high at best.
- **The models' contribution was diagnosis.** Given the damage mask, a five-line column-mean fill reaches 30.7–31.7 everywhere; the best models beat it by about one point. Erasure (A, C) does not need a frontier model, because zeros give the damage away. Fable's blind repair of D matched the mask-informed reference. In B no model came close to it.
- **Repair is detection plus neutralisation, not recovery.** In those three conditions, outside the two embedding tables, 65–97% of the weight error remained after the best repair. The function returned anyway.
- **Capability separated the models only where the damage had to be found.** With zeroed blocks (C) all four tied. With noise blocks (D) Fable located the block in all 50 matrices, Opus in 25, Sonnet and Haiku in none.
- **GPT-2's tied token embedding is the weak point.** A few of its columns are near-constant offsets with a large gain, so one wrong entry shifts a token's logit by tens of nats.

## Limits

- One run per model per condition, one seed. Differences of a few percent are not established.
- GPT-2, not Qwen or Gemma: the run environment could not reach the hosts those models are published on.
- Isolation was by instruction and audit, not by sandbox. The answer key was encrypted during the runs, and the agents' saved scripts contain no network calls or gradient code, but the agents shared one machine with network access.
- Agents could compare a handful of candidate repairs on development perplexity. Reported scores use separate test data.
- The time budget (about 40 minutes) was a request, not a limit; one run took 78 minutes.
- Haiku's case A report was saved by the orchestrator from the agent's final message, because the agent's own file write was refused.

## Layout

```
agent_kit/     what each repair agent was given (TASK.md, gpt2_lib.py, dev.txt, tokenizer)
src/           gpt2_lib.py, damage.py, benchmarks.py, prepare.py, evaluate_runs.py,
               heuristic_baseline.py, build_page.py
runs/case_X/<model>/   every script the agent ran, its logs and its REPORT.md
results/       results.json (all scores), run_usage.json (tokens, tool calls, minutes)
page/, docs/   report page source and the built GitHub Pages site
```

Case letters map to conditions as A = `scatter_zero`, B = `scatter_noise`, C = `block_zero`, D = `block_noise`. Agents saw only the letter.

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/fetch_assets.sh
python src/prepare.py assets secret
```

`prepare.py` writes the clean checkpoint, the four damaged checkpoints and the masks to `secret/`. Move `secret/damaged_<condition>.safetensors` to `stage/case_<letter>.safetensors`, give a copy to an agent together with `agent_kit/` as `damaged.safetensors`, and keep `secret/` and `assets/` out of its reach. Put each agent's output at `runs/case_<letter>/<model>/repaired.safetensors`, then:

```bash
python src/evaluate_runs.py assets secret runs stage results/results.json
```

```bash
python src/build_page.py --author "Your Name" --affiliation "Your Affiliation"
```

## 한국어 요약

GPT-2(124M) 가중치의 1%를 네 가지 방식(산발 소거, 산발 노이즈, 블록 소거, 블록 노이즈)으로 손상시킨 뒤, 손상된 체크포인트만 Fable·Opus·Sonnet·Haiku에게 주고 원본·다운로드·그래디언트 학습 없이 가중치를 직접 고치게 했습니다. 네 조건 중 세 조건에서 가장 좋은 모델은 held-out perplexity를 손상 전의 약 6% 이내(28.9 대비 30.2–30.8)로 되돌렸고, 산발 노이즈 조건에서는 최선이 81이었습니다. 복구는 잃어버린 값을 되찾는 것이 아니라 손상을 찾아 무력화하는 방식으로 작동했습니다. 모델·조건당 1회 실행이라는 점에 유의해 주세요. 자세한 내용은 보고서 페이지에 있습니다.
