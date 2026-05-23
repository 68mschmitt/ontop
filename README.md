# ontop

`ontop` is a local terminal dashboard for watching Ollama, NVIDIA GPUs, CPU, and RAM while your machine politely pretends that running large language models is a normal hobby.

It is like `top`, if `top` had heard about local AI and immediately started asking your GPU uncomfortable questions.

```text
o  = Ollama
n  = NVIDIA
top = top, htop, nvtop, btop, atop, and the ancient terminal tradition of staring at numbers until insight happens
```

So yes, the name is `ontop` because it sits on top of Ollama and NVIDIA telemetry while tipping its tiny ASCII hat to every `top`-shaped tool that came before it.

Was this clever? Arguably.

Was it necessary? Absolutely not.

Are we keeping it? Obviously.

## What It Does

`ontop` gives you a live, local dashboard for the question every local model tinkerer eventually asks:

> Is my GPU doing wizard math, or is everything just quietly on fire?

It watches the important bits without making you juggle five terminals, three `watch` commands, and one cursed spreadsheet named `gpu_notes_final_REAL.xlsx`.

## The Vibe

- Terminal-native, because pixels are expensive and ANSI escape codes are culture.
- Local-first, because your GPU is already screaming in the room with you.
- Ollama-aware, because `ps aux | grep ollama | grep -v grep` is a cry for help.
- NVIDIA-aware, because `nvidia-smi` is useful but has the bedside manner of a tax form.
- Small enough to understand, useful enough to keep open, nerdy enough to deserve a monospace font.

## Requirements

- Go 1.21 or newer to build the little goblin.
- `nvidia-smi` from the NVIDIA driver stack for GPU metrics.
- `ollama` on `PATH` if you want the loaded model list from `ollama ps`.

No NVIDIA GPU? No Ollama daemon? `nvidia-smi` went out for milk and never came back?

`ontop` degrades gracefully. It will not throw a tantrum. It will simply report what it can and leave the rest as a mystery for future archaeologists.

## Build

```sh
go build -o ontop .
```

Congratulations. You now have a binary named `ontop`, which is both a program and a pun delivery mechanism.

## Run

```sh
./ontop
```

Need a different refresh interval because one second is either too slow, too fast, or emotionally incompatible with your workflow?

```sh
./ontop -interval=2s
./ontop -interval=500ms
```

Supported interval examples include `500ms`, `1s`, `2s`, and whatever duration makes your terminal feel alive without becoming a strobe light for statistics.

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit, flee, return to society |
| `ctrl+c` | Quit, but with Unix seasoning |
| `r` | Refresh immediately, because patience is a kernel parameter you did not set |
| `up` / `down` | Scroll like a responsible adult |
| `pgup` / `pgdn` | Scroll like you mean it |

## Metrics

`ontop` currently keeps an eye on:

- NVIDIA GPU name, index, utilization, VRAM, temperature, power, fan speed, and active compute processes via `nvidia-smi --query-* --format=csv,noheader,nounits`.
- RAM used, total, available, and percent via `gopsutil`.
- Total CPU and per-core CPU usage via `gopsutil`.
- Ollama-related local processes with PID, command, CPU, RAM, RSS, and runtime via `gopsutil`.
- Loaded or running Ollama models from `ollama ps` when available.

In short: it watches the silicon, the memory, the model goblins, and the little daemon friends that make your laptop sound like it is preparing for takeoff.

## Why Not Just Use Other Tools?

You should. Other tools are great.

Use `top` for general process vibes.

Use `htop` when you want colors and tree views.

Use `nvtop` when your GPU needs its own command center.

Use `nvidia-smi` when you want raw NVIDIA truth beamed directly into your eyeballs.

Use `ontop` when you are running Ollama on NVIDIA hardware and want the relevant local AI chaos in one place.

This is not a replacement for the classics. This is a tiny specialized gremlin standing on their shoulders with a clipboard.

## Failure Modes

`ontop` tries to be chill.

- If `nvidia-smi` is missing, GPU metrics are unavailable.
- If there is no NVIDIA GPU, GPU metrics are unavailable.
- If `ollama` is missing, `ollama ps` output is unavailable.
- If Ollama is installed but not doing anything, `ontop` will not invent drama.

That last one is important. This is a dashboard, not a horoscope.

## Development

Format the code:

```sh
gofmt -w .
```

Run tests:

```sh
go test ./...
```

Build it:

```sh
go build -o ontop .
```

Then run it, stare at the bars, nod thoughtfully, and say "interesting" even if everything is fine.

## Name Lore

The project is named `ontop` because:

- `o` is for Ollama.
- `n` is for NVIDIA.
- `top` is for the noble Unix lineage of terminal dashboards that show you what the machine is doing instead of what you hoped it was doing.

It is short, searchable-ish, mildly cursed, and just clever enough that explaining it makes the joke both better and worse.

Perfect.
