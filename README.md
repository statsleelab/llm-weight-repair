# Blind Weight Repair

**Can a frontier LLM fix a damaged language model by editing its weights directly, without ever seeing the original?**

**Donghyung Lee**, Miami University · November 2026

Report page (English / 한국어): **https://statsleelab.github.io/llm-weight-repair/**

This is Part III of our weight-space programming series, after the [reproduction of Weight-Space Programming](https://github.com/statsleelab/weight_space_programming) and the [connectome weight surgery](https://statsleelab.github.io/connectome-weight-edit/) study.

## The idea

A language model is a file of numbers called *weights*. GPT-2 has 124 million of them. Suppose that file gets damaged, so that some of the numbers are wrong. Can an LLM look at the damaged file and repair it?

Think of a book in which one letter in a hundred has been smudged, and there is no second copy. A careful reader can often spot the smudges and write in something sensible. We tested whether an LLM can do the same for a neural network.

The rule that makes it hard: **the repairing model never sees the answer.** It gets only the damaged file. It cannot download the original and it cannot retrain the model. It can read the numbers, run the broken model to see how it misbehaves, do statistics, and write new numbers.

## Short answer

- **Partly yes.** With 1% of the weights damaged, the best model brought GPT-2 back to near-normal in three of four kinds of damage.
- **What the models contributed was finding the damage.** Once you know which numbers are wrong, a five-line script fixes the model almost as well as the best LLM. The hard part is working out, unaided, what is wrong and where.
- **A frontier model is needed only when the damage leaves no mark.** Erased weights are exact zeros, which any script can find. Noise leaves no such mark. Fable found hidden noise blocks as well as a script that was told the answer. Scattered noise defeated every script we wrote, and Fable recovered only part of the way.
- **Repair is not restoration.** The repaired model works again, but 65–97% of the error in the damaged weights is still there. The network tolerates "reasonable" values; it does not tolerate wildly wrong ones.

## How the experiment ran

1. **Break.** Take GPT-2 (124M). Corrupt 1% of the entries in each of its 50 weight matrices. Encrypt everything that could give the answer: the original weights, the list of damaged entries and the test data.
2. **Repair blind.** Each of four Claude models (Fable, Opus, Sonnet, Haiku) works alone in a folder with the damaged file, a small program that runs GPT-2, and a page of practice text. It is told only that "about 1% of the entries are wrong". Statistics, linear algebra and running the model are allowed. Downloads and training are not. The full brief is [`agent_kit/TASK.md`](agent_kit/TASK.md).
3. **Score.** After all runs end, decrypt the answer key and test each repaired model on text it never saw.

Two terms used below:

- **Perplexity**: how surprised the model is by ordinary text. Lower is better. Healthy GPT-2 scores 28.9; above a few hundred its writing is nonsense.
- **Collapsed**: so broken that the model repeats one word forever. Its perplexity is astronomically large.

## Four kinds of damage

| Case | Where | What happens to the numbers | Perplexity after damage |
|---|---|---|---|
| A · Scattered erasure | random entries | set to zero | collapsed |
| B · Scattered noise | random entries | large random values added | collapsed |
| C · Block erasure | one rectangle per matrix | set to zero | 395.7 |
| D · Block noise | one rectangle per matrix | overwritten with noise the same size as real weights | 43.9 |

Zeros are easy to spot, because trained weights are almost never exactly zero. Noise leaves no such mark.

## Results at 1% damage

Perplexity on unseen text (healthy model: 28.9). One run per cell.

| Repair by | A · Scattered erasure | B · Scattered noise | C · Block erasure | D · Block noise |
|---|---|---|---|---|
| **Fable** | 30.4 | **81.0** | **30.8** | **30.5** |
| **Opus** | **30.2** | 1,507 | 30.9 | 38.4 |
| **Sonnet** | 36.5 | collapsed | 31.0 | 43.7 |
| **Haiku** | 51.1 | 29,839 | 30.9 | 43.9 (file returned unchanged) |
| Simple script: fill zeros with the column average | 31.4 | no effect | 31.7 | no effect |
| Yardstick: script that is told where the damage is | 31.4 | 30.7 | 31.7 | 30.9 |
| Damaged, no repair | collapsed | collapsed | 395.7 | 43.9 |

How to read it:

- **A and C (erasure).** Every model saw that the zeros mark the damage. A five-line script does as well as the best models.
- **D (block noise).** Nothing marks the damage. Fable found the block in all 50 matrices and matched the yardstick. Opus found half. Sonnet and Haiku found none.
- **B (scattered noise).** The hardest case. Fable's 81 is far better than the others and still far from the 30.7 that is possible when the damage is known.
- **The yardstick row is not a competitor.** It cheats on purpose, to show what perfect detection plus a crude fill is worth.

One script we fixed before any model ran (fill zeros and clip unusually large weights) made everything much worse, between 10,000 and 75,000. GPT-2 really does contain a few enormous weights, and they matter.

## Follow-up: more damage, stronger scripts

Two doubts remained. Is 1% a special amount? Were our scripts too weak to be a fair comparison? We wrote stronger automatic scripts ([`src/strong_baselines.py`](src/strong_baselines.py)), ran them from 0.1% to 30% damage, and sent the four models into the three cases where the scripts fell furthest short.

Perplexity on a smaller test set (healthy model: 32.2 on this set). "Ceiling" is the same repair when the script is told where the damage is.

| Repair by | Scattered noise 0.1% | Scattered noise 1% | Scattered noise 3% | Block noise 1% | Block noise 3% |
|---|---|---|---|---|---|
| **Fable** | **53.6** | **86.7** | **1,426** | **34.2** | **40.3** |
| **Opus** | 494 | 2,019 | 14,507 | 43.0 | 43.4 |
| **Sonnet** | 215 | collapsed | 14,381 | 48.8 | 48.1 |
| **Haiku** | collapsed | 30,953 | collapsed | 48.8 (unchanged) | 259.6 (unchanged) |
| Best automatic script | collapsed | collapsed | collapsed | 40.5 | 67.3 |
| Ceiling | 32.6 | 33.6 | 39.6 | 34.5 | 40.0 |
| Damaged, no repair | collapsed | collapsed | collapsed | 48.8 | 259.6 |

- **Erasure never needed a model.** At every damage level the script equals the ceiling. The ceiling itself gets worse: about 40 at 3%, around 200 at 10%, unusable at 30%.
- **Noise blocks: Fable matches the ceiling** at both 1% and 3%. The best script is clearly behind. Bigger blocks were easier to find, so Opus and Sonnet located almost all of them at 3%.
- **Scattered noise: still unsolved.** No script produced a working model at any level, even 0.1%. Fable is the only repair that works at all, and it falls further behind as damage grows.

Why scattered noise is so stubborn: GPT-2 has a weak spot. The table that turns words into numbers is also used, in reverse, to choose the next word. A few of its columns hold almost the same value for every word and are multiplied by a large factor, so one wrong entry there makes one word win every prediction. The model stays collapsed unless nearly every wrong entry in those columns is caught. Fable identified those columns from the broken model's behaviour and handled them separately.

## Limits

- **One run per case.** 28 runs, one random seed, no repeats. Large gaps are real; gaps of a few percent may be luck.
- **One small, old model.** We used GPT-2 because the run environment could not reach the sites that host Qwen and Gemma. GPT-2's weak spot shapes these results.
- **"Scripts fail" means our scripts.** They were written in an afternoon, after the first-round reports had been read, by the same model family that ran the experiment. A script that copies Fable's handling of the weak spot would probably do much better. The claim is that the model worked this out unaided, not that no script could.
- **Isolation was by rule and audit, not by a sealed box.** The answer key was encrypted during the runs and the saved scripts contain no downloads or training code, but the models shared one networked machine.
- **Practice text was used for choosing** between a few candidate repairs. All reported scores use separate test text.
- **Time limits were requests.** One run took 78 minutes instead of 40.
- **The follow-up is partial.** Models ran only on noise damage at 0.1%, 1% and 3%, and were told the damaged share; the scripts had to estimate it.

## What is in this repository

```
agent_kit/     what each repair agent was given (TASK.md, gpt2_lib.py, dev.txt, tokenizer)
               variants/ holds the briefs for the follow-up cases
runs/case_X/<model>/   every script the agent wrote, its logs and its own REPORT.md
results/       results.json (1% study), dose_response.json (script grid across damage levels),
               llm_lite.json (all model repairs on the smaller test set), run_usage.json
src/           gpt2_lib.py, damage.py, benchmarks.py, prepare.py, evaluate_runs.py,
               heuristic_baseline.py, strong_baselines.py, dose_response.py,
               make_cells.py, score_llm_lite.py, build_page.py
page/, docs/   report page source and the built GitHub Pages site
```

Case letters: A = scattered erasure, B = scattered noise, C = block erasure, D = block noise (all at 1%); E = scattered noise 0.1%, F = scattered noise 3%, G = block noise 3%. The agents saw only the letter.

## Run it yourself

```bash
pip install -r requirements.txt
bash scripts/fetch_assets.sh
python src/prepare.py assets secret
```

`prepare.py` writes the healthy checkpoint, the four damaged checkpoints and the damage masks to `secret/`. Damage uses a fixed seed, so the same files are rebuilt exactly. Move `secret/damaged_<condition>.safetensors` to `stage/case_<letter>.safetensors`, give a copy named `damaged.safetensors` to a repair agent together with `agent_kit/`, and keep `secret/` and `assets/` out of its reach. Put each agent's output at `runs/case_<letter>/<model>/repaired.safetensors`, then score:

```bash
python src/evaluate_runs.py assets secret runs stage results/results.json
```

Follow-up (script grid, extra cases, scores on the smaller test set):

```bash
python src/dose_response.py assets secret/clean.safetensors results/dose_response.json
python src/make_cells.py secret/clean.safetensors stage secret
python src/score_llm_lite.py assets secret runs results/llm_lite.json
```

Rebuild the report page:

```bash
python src/build_page.py --author "Your Name" --affiliation "Your Affiliation"
```

## 한국어 요약

**질문.** 언어 모델은 가중치라는 숫자들이 담긴 파일입니다. 그 파일의 숫자 일부가 망가졌을 때, 프런티어 LLM이 원본을 보지 않고 숫자를 직접 고쳐서 모델을 살릴 수 있을까요? 백 글자 중 한 글자가 번졌는데 사본이 없는 책을 고치는 일과 비슷합니다.

**방법.** GPT-2(124M)의 가중치 1%를 네 가지 방식(산발 소거, 산발 노이즈, 블록 소거, 블록 노이즈)으로 망가뜨린 뒤, 손상된 파일만 Fable·Opus·Sonnet·Haiku에게 주었습니다. 원본, 다운로드, 재학습은 금지했고, 복구가 끝난 뒤에 모델이 보지 못한 글로 채점했습니다.

**결과.**

- 가장 좋은 모델은 네 가지 손상 중 세 가지에서 모델을 거의 정상으로 되돌렸습니다(정상 28.9 대비 30.2–30.8). 산발 노이즈에서는 최선이 81이었습니다.
- 모델이 기여한 것은 손상을 찾아내는 일이었습니다. 손상 위치만 알면 5줄짜리 스크립트도 최고 모델과 거의 같은 결과를 냅니다.
- 소거 손상은 0이 위치를 알려 주므로 스크립트로 충분합니다. 흔적이 없는 노이즈 블록은 Fable만 전부 찾아냈고, 정답을 받은 스크립트와 같은 수준이었습니다.
- 후속 실험(손상 0.1–30%, 더 강한 스크립트)에서도 같은 패턴이었습니다. 산발 노이즈는 모든 스크립트가 실패했고 Fable만 일부를 회복했습니다(0.1%에서 53.6, 3%에서 1,426. 이 테스트 세트에서 정상은 32.2).
- 복구는 복원이 아닙니다. 모델은 다시 작동하지만 손상된 가중치의 오차는 65–97%가 그대로 남았습니다.

**유의점.** 경우당 실행 1회, 시드 하나, 작은 모델 하나에 대한 결과입니다. 자세한 내용은 [보고서 페이지](https://statsleelab.github.io/llm-weight-repair/)에 있습니다.
