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

## Follow-up: dose–response and stronger scripts

Two questions from the first round: is 1% a special dose, and were the scripted baselines too weak? We wrote stronger blind scripts ([`src/strong_baselines.py`](src/strong_baselines.py): ridge imputation of zeros, mixture-model outlier detection, a rectangle scan for unpredictable blocks, and selection among them on development perplexity), ran them at 0.1%, 1%, 3%, 10% and 30% damage, and then ran the four Claude models in the three cells where the scripts fell furthest short.

Held-out perplexity on a reduced suite (8,192 WikiText-2 test tokens; undamaged model 32.2). "Ceiling" is ridge imputation when the script is told the damage mask. One run per cell.

| Repair by | Scattered noise 0.1% | Scattered noise 1% | Scattered noise 3% | Block noise 1% | Block noise 3% |
|---|---|---|---|---|---|
| **Fable** | **53.6** | **86.7** | **1,426** | **34.2** | **40.3** |
| **Opus** | 494 | 2,019 | 14,507 | 43.0 | 43.4 |
| **Sonnet** | 215 | ~10⁴⁷ | 14,381 | 48.8 | 48.1 |
| **Haiku** | ~10¹¹ | 30,953 | ~10⁶⁸ | 48.8 (unchanged) | 259.6 (unchanged) |
| Best blind script | collapsed | collapsed | collapsed | 40.5 | 67.3 |
| Ceiling (mask known) | 32.6 | 33.6 | 39.6 | 34.5 | 40.0 |
| Damaged, no repair | collapsed | collapsed | collapsed | 48.8 | 259.6 |

- **Erasure never needed a model.** At every dose the blind script equals the ceiling. The ceiling itself degrades: about 39–41 at 3%, 180–280 at 10%, unusable at 30%.
- **Block noise: Fable stays on the ceiling** at 1% and 3%, ahead of the best script. At 3% Opus and Sonnet also located almost all blocks; Haiku returned the checkpoint unchanged both times.
- **Scattered noise: every script fails at every dose**, even 0.1%. Fable gets part of the way and its gap to the ceiling grows with dose. The other models are far behind.
- **Caveat.** The scripts were written in an afternoon by the model family that ran the experiment, after reading the first-round reports. A script that hard-codes Fable's handling of the high-gain embedding columns would probably do much better. Models were told the damaged share; scripts estimated it.

Full grid: [`results/dose_response.json`](results/dose_response.json); model scores on the reduced suite: [`results/llm_lite.json`](results/llm_lite.json).

## Limits

- One run per model per cell (28 runs in total), one seed. Differences of a few percent are not established.
- GPT-2, not Qwen or Gemma: the run environment could not reach the hosts those models are published on.
- Isolation was by instruction and audit, not by sandbox. The answer key was encrypted during the runs, and the agents' saved scripts contain no network calls or gradient code, but the agents shared one machine with network access.
- Agents could compare a handful of candidate repairs on development perplexity. Reported scores use separate test data.
- The time budget (about 40 minutes) was a request, not a limit; one run took 78 minutes.
- Haiku's case A report was saved by the orchestrator from the agent's final message, because the agent's own file write was refused.

## Layout

```
agent_kit/     what each repair agent was given (TASK.md, gpt2_lib.py, dev.txt, tokenizer)
src/           gpt2_lib.py, damage.py, benchmarks.py, prepare.py, evaluate_runs.py,
               heuristic_baseline.py, build_page.py,
               strong_baselines.py, dose_response.py, make_cells.py, score_llm_lite.py
runs/case_X/<model>/   every script the agent ran, its logs and its REPORT.md
results/       results.json (1% study), dose_response.json (scripted dose grid),
               llm_lite.json (all model repairs on the reduced suite), run_usage.json
page/, docs/   report page source and the built GitHub Pages site
```

Case letters map to conditions as A = `scatter_zero`, B = `scatter_noise`, C = `block_zero`, D = `block_noise`. The follow-up cells are E = `scatter_noise` 0.1%, F = `scatter_noise` 3%, G = `block_noise` 3% (briefs in `agent_kit/variants/`). Agents saw only the letter.

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

Follow-up:

```bash
python src/dose_response.py assets secret/clean.safetensors results/dose_response.json
python src/make_cells.py secret/clean.safetensors stage secret
python src/score_llm_lite.py assets secret runs results/llm_lite.json
```

```bash
python src/build_page.py --author "Your Name" --affiliation "Your Affiliation"
```

## 한국어 요약

GPT-2(124M) 가중치의 1%를 네 가지 방식(산발 소거, 산발 노이즈, 블록 소거, 블록 노이즈)으로 손상시킨 뒤, 손상된 체크포인트만 Fable·Opus·Sonnet·Haiku에게 주고 원본·다운로드·그래디언트 학습 없이 가중치를 직접 고치게 했습니다. 네 조건 중 세 조건에서 가장 좋은 모델은 held-out perplexity를 손상 전의 약 6% 이내(28.9 대비 30.2–30.8)로 되돌렸고, 산발 노이즈 조건에서는 최선이 81이었습니다. 복구는 잃어버린 값을 되찾는 것이 아니라 손상을 찾아 무력화하는 방식으로 작동했습니다. 후속 실험에서는 더 강한 스크립트 기준선과 0.1–30% 용량 격자를 추가하고, 스크립트가 무너지는 세 지점(산발 노이즈 0.1%·3%, 블록 노이즈 3%)에서 네 모델을 다시 돌렸습니다. 소거는 모든 용량에서 스크립트가 상한과 같았고, 블록 노이즈에서는 Fable만 두 용량 모두 상한에 도달했으며, 산발 노이즈에서는 모든 스크립트가 실패하고 Fable이 부분적으로만 회복했습니다(0.1%에서 53.6, 3%에서 1,426. 축소 벤치마크 기준 손상 전 32.2). 모델·조건당 1회 실행이라는 점에 유의해 주세요. 자세한 내용은 보고서 페이지에 있습니다.
