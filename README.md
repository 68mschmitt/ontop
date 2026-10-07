# ontop

`ontop` is a local terminal dashboard for watching Ollama, NVIDIA and AMD GPUs, CPU, and RAM while your machine politely pretends that running large language models is a normal hobby.

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
- `nvidia-smi` from the NVIDIA driver stack for NVIDIA GPU metrics, when present.
- `amdgpu_top` for AMD GPU metrics, when present.
- `ollama` on `PATH` if you want the loaded model list from `ollama ps`.
- Unsloth Studio is optional and discovered on its configured port or common local ports.

No NVIDIA GPU? No Ollama daemon? `nvidia-smi` went out for milk and never came back?

`ontop` degrades gracefully. It will not throw a tantrum. It will simply report what it can and leave the rest as a mystery for future archaeologists.

## Build

```sh
go build -o ontop .
```

Or, with the included `Makefile`:

```sh
make build
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

## Configuration

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `-interval` | `1s` | Refresh interval (`500ms`, `1s`, `2s`, ...). |
| `-theme` | `dark` | Color theme: `dark`, `midnight`, or `monokai`. |
| `-list-themes` | | Print the available themes and exit. |
| `-once` | `false` | Collect a single snapshot and exit. |
| `-export` | | Output target: `stdout`, `json`, `csv`, or a file path. |
| `-unsloth-port` | `8888` | Unsloth Studio API port. |
| `-unsloth-token` | | Unsloth Studio bearer token. |
| `-version` | | Print the version and exit. |

### Environment variables

| Variable | Description |
| --- | --- |
| `ONTOP_INTERVAL` | Refresh interval; takes precedence over `-interval` when set. |
| `ONTOP_THEME` | Theme name; takes precedence over `-theme` when set. |
| `UNSLOTH_STUDIO_URL` | Explicit Unsloth Studio base URL, probed before port discovery. |
| `UNSLOTH_STUDIO_TOKEN` | Unsloth Studio bearer token, used when `-unsloth-token` is not set. |

### Themes

```sh
./ontop -list-themes
./ontop -theme=midnight
```

### Non-interactive output

```sh
./ontop -once                      # one text snapshot to stdout
./ontop -once -export=json         # one snapshot as JSON
./ontop -once -export=csv          # CPU, RAM, and swap as CSV
./ontop -export=dashboard.txt      # render once and write to a file
```

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit, flee, return to society |
| `ctrl+c` | Quit, but with Unix seasoning |
| `r` | Refresh immediately, because patience is a kernel parameter you did not set |
| `up` / `down` | Scroll like a responsible adult |
| `pgup` / `pgdn` | Scroll like you mean it |
| `s` | Toggle the System (CPU) section |
| `g` | Toggle the GPU sections (NVIDIA and AMD) |
| `m` | Toggle the Memory section |
| `u` | Toggle the Unsloth section |
| `o` | Toggle the Ollama sections |
| `?` / `h` | Show or hide the help overlay |

## Metrics

`ontop` currently keeps an eye on:

- NVIDIA GPU name, index, utilization, VRAM, temperature, power, fan speed, and active compute processes via `nvidia-smi --query-* --format=csv,noheader,nounits`.
- AMD GPU name, PCI address, utilization, VRAM, temperature, power, and per-process VRAM/GTT via `amdgpu_top --json`.
- RAM used, total, available, and percent via `gopsutil`.
- Swap used and total.
- Total CPU and per-core CPU usage via `gopsutil`, with a rolling utilization sparkline.
- Thermal zones via `/sys/class/thermal`.
- Disk read/write throughput and IOPS per device via `gopsutil`.
- Network send/receive throughput per interface via `gopsutil`.
- Ollama-related local processes with PID, command, CPU, RAM, RSS, and runtime via `gopsutil`.
- Loaded Ollama models from `ollama ps`, including size, CPU/GPU placement, context, and expiry when available.
- Unsloth Studio active model, context, loading, training, throughput, and parallel-session metrics when its API exposes them.

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
- If `amdgpu_top` is missing, AMD GPU metrics are unavailable.
- If there is no NVIDIA GPU, GPU metrics are unavailable.
- If `ollama` is missing, `ollama ps` output is unavailable.
- Missing optional services are omitted from the dashboard rather than shown as empty cards.
- If Ollama is installed but not doing anything, `ontop` will not invent drama.

That last one is important. This is a dashboard, not a horoscope.

## Development

Common tasks are wrapped in the `Makefile`:

```sh
make fmt         # gofmt -w .
make fmt-check   # fail if anything is unformatted
make vet         # go vet ./...
make test        # go test ./...
make test-race   # go test -race -count=1 ./...
make bench       # run the benchmarks
make build       # go build -o ontop .
```

Build it, run it, stare at the bars, nod thoughtfully, and say "interesting" even if everything is fine.

Print the build version without starting the dashboard:

```sh
./ontop -version
```

## Name Lore

The project is named `ontop` because:

- `o` is for Ollama.
- `n` is for NVIDIA, with AMD now invited to the telemetry party too.
- `top` is for the noble Unix lineage of terminal dashboards that show you what the machine is doing instead of what you hoped it was doing.

It is short, searchable-ish, mildly cursed, and just clever enough that explaining it makes the joke both better and worse.

Perfect.
