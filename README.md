# MNIST on a fruit fly brain (PoC)

Can a simulated *Drosophila* brain, wired exactly like the real one, recognise handwritten digits?

This PoC takes the whole-brain spiking model from
[eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
(138,639 neurons, 15M connections from the FlyWire connectome; LIF model of
[Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9)),
shows MNIST digits to its eyes, and trains a small readout to read the digit
from the spikes of different brain regions.

**The brain itself is never trained.** Its wiring is fixed, as in the real fly.
Only a linear readout (logistic regression) on top is learned. This is
*reservoir computing*: the connectome is the reservoir.

## How it works

```
MNIST image ──► fly eye ──► 138k-neuron spiking brain ──► spike counts ──► logistic regression ──► digit
   28x28        (retinotopic    (fixed FlyWire wiring,       per region      (the only trained part)
                 Poisson rates)  100 ms, dt = 0.1 ms)
```

1. **Eye mapping** ([flymnist/retina.py](flymnist/retina.py)). Neuron cell types come from the
   public [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations).
   Each eye's visual neurons lie on a flat sheet in 3D, so they are flattened with PCA and
   placed on the 28x28 image. Every input neuron fires Poisson spikes at
   `pixel intensity x rate` (default 500 Hz). Both eyes see the same digit.
   By default the image enters at the **lamina** (photoreceptors R1-6/R7/R8 + lamina cells
   L1-L5, ~18k neurons); `--layer photoreceptors` uses photoreceptors only.
2. **Brain simulation** ([flymnist/brain.py](flymnist/brain.py)). Same equations and parameters
   as `code/run_pytorch.py` in fly-brain, rewritten to be event-driven (only synapses of neurons
   that spiked are processed), which is fast on a laptop CPU (~2.5 digits/s on an M3 Max).
   It reproduces the reference implementation spike for spike.
3. **Readout** ([flymnist/readout.py](flymnist/readout.py)). For each region (optic lobe,
   visual projection neurons, central brain, descending neurons, whole brain), a logistic
   regression on `log(1 + spike count)` of that region's neurons. **The stimulated input
   neurons are excluded**, so the digit can only be decoded if the information travelled
   through synapses.
4. **Controls.** Logistic regression on raw pixels (what a linear model achieves without any
   brain), and the same brain with randomly rewired connections (`--shuffle`): every neuron
   keeps its outputs (count, sign, strength) but they go to random targets.

## Run it

```bash
uv sync
uv run main.py download                          # ~140 MB: connectome, annotations, MNIST
uv run main.py all --train 3000 --test 1000      # ~40 min on CPU
uv run main.py all --train 3000 --test 1000 --shuffle   # random-wiring control
```

### Web app: draw a digit, watch the brain

```bash
uv run main.py serve        # needs the trained readouts from the step above
# open http://127.0.0.1:8000
```

Draw a digit (or press **MNIST sample**) and press **Run through brain** (or Enter).
The server crops and centres your drawing the way MNIST digits were prepared; the brain then
runs **on your own device** (100 ms of brain activity in ~0.5 s on a laptop) and the page shows:

- the predicted digit from each region's readout, next to the raw-pixel baseline;
- a 3D map of all 138,639 neurons at their real FlyWire positions (drag to rotate, scroll to
  zoom; Front/Top/Side presets, Spin). Every spike lights up, coloured by region, so you can
  watch activity spread from the eyes into the optic lobes and further in. **Total** shows every
  neuron that fired. The main neuropils (lamina, medulla, lobula, lobula plate, mushroom body,
  lateral horn, antennal lobe, central complex, SEZ) are labelled at the median position of the
  neurons that belong to them. Uses Three.js from a CDN, so the page needs internet;
- spikes per millisecond per region, and the most active cell types (Tm1, Mi9, T1, ...).

Code: [flymnist/server.py](flymnist/server.py) (FastAPI) and
[flymnist/static/index.html](flymnist/static/index.html) (one page, no build step).
If you trained with non-default `--layer/--rate/--ms`, pass the same flags to `serve`.

**Every visitor has their own brain.** The brain runs in the browser:
[flymnist/static/brainsim.js](flymnist/static/brainsim.js) is the same model as `brain.py`
(same equations, parameters and step order) in plain JavaScript, in a Web Worker
([brain-worker.js](flymnist/static/brain-worker.js)). On first visit the page downloads the
connectome once (~40 MB, packed by [flymnist/webmodel.py](flymnist/webmodel.py) and kept in
the browser's cache). Checked against the Python model on 200 MNIST test digits: 175,353 vs
175,372 spikes per digit, per-neuron correlation 0.9996, and the same readout accuracies within
noise. Speed on an M3 Max: ~0.4 s per 100 ms digit, ~0.2 s per 50 ms camera frame; expect
2–4× longer on phones. The server keeps what is cheap or shared: image preprocessing, knockout
lookup, decoder training, everyone's recordings, and one fly body per visitor.

**Classroom use.** `serve` listens on the local network by default and prints the address
(e.g. `http://10.145.119.50:8000`). The page header shows the same address with a QR code;
click it for a full-screen QR to put on a projector. Use `--host 127.0.0.1` to keep it on
this computer only. Notes:

- Each device runs its own brain, so nobody waits for anyone else, and knockouts only affect
  the device that made them (they are remembered on that device).
- The **teacher** is whoever uses the app on the server computer. Their camera recordings and
  decoder are the **class model**. Every student starts with it, can record their own gestures
  and press **Train**; their decoder then learns from the class recordings plus their own (or only
  their own: untick "Also learn from the class recordings"). Students are told apart by a cookie
  and their work is saved in `data/sessions/`, so a reload or server restart keeps it.
  On the teacher's computer, `?as=student` shows the app as a student sees it (`?as=teacher` to
  go back).
- Browsers only allow the webcam on `https://` pages or on `localhost`, so `serve` also listens
  on **https://<LAN IP>:8443** with a self-signed certificate (made with openssl in `data/tls/`,
  remade if the IP changes) and the QR code points there. Each device sees a "connection is not
  private" warning once: Safari → *Show Details* → *visit this website*; Chrome → *Advanced* →
  *Proceed*. `--https-port 0` turns HTTPS off.
- macOS may ask whether Python may accept incoming connections: allow it.
- Long jobs (training the camera decoder, testing knockouts on 200 digits) show a progress bar
  with a time estimate (`GET /api/progress`).

### Knockouts

The **Knockouts** panel (both tabs) lesions the brain:

- **Silence neurons.** Pick a target, a side (both / left / right) and an amount (10-100%, a random
  but fixed subset), then press **Knock out**. Targets: whole regions (optic lobe, central brain,
  visual projection, descending, ascending, eye input, whole brain), optic lobe neuropils (lamina,
  medulla, lobula, lobula plate: neurons with arbors there), central structures (Kenyon cells,
  MBONs, dopaminergic neurons, antennal lobe, lateral horn, central complex), sensory neurons, or
  **cell types by name** with wildcards (`LPLC2, Tm*, DN*`). Rules stack and can be removed one by one.
  A silenced neuron loses every synapse in and out, and gets no eye input, like optogenetic
  silencing in fly-brain.
- **Split brain.** Cuts all 1,784,093 connections between left-side and right-side neurons
  (14.5% of synapses).
- Silenced neurons are dark red in the 3D map and the cut is a red midline plane. The current
  digit is re-run automatically. **Test on 200 digits** compares every readout's accuracy with the
  intact brain on the same test digits.

The readouts and the camera decoder were trained on the intact brain, so a lesion tests how
well they still work. Retrain the camera decoder to see whether the lesioned brain still carries
the information. Code: [flymnist/lesion.py](flymnist/lesion.py), `FlyBrain.set_lesion`.

Check with the escape circuit (right-eye looming detectors stimulated): the left giant fiber
fires at 100 Hz intact and 0 Hz with the hemispheres cut, while the right one stays at 190 Hz.

#### Split brain: can each hemisphere recognise digits on its own?

`uv run main.py simulate --split --train 3000 --test 1000` simulates the same digits with the
same input spikes on a brain with all left↔right connections cut; `uv run main.py hemispheres
[--split]` trains left-only / right-only / both readouts (saved to `*.hemispheres.joblib/json`,
separate from the web app's readouts). Test accuracy on 1,000 digits:

| Readout | Intact: left | Intact: right | Intact: both | Split: left | Split: right | Split: both |
|---|---|---|---|---|---|---|
| Whole brain | 91.2% | 90.6% | 91.4% | 91.0% | 90.4% | 91.4% |
| Optic lobe | 91.0% | 90.2% | 91.1% | 91.2% | 90.1% | 91.2% |
| Central brain | 52.4% | 53.6% | 63.6% | 55.0% | 50.9% | 67.1% |
| Descending | 47.6% | 51.2% | 62.2% | 48.3% | 49.0% | 64.1% |

Readouts trained on the **intact** brain and then applied to the split brain do drop
(whole brain 91.4% → 78.0%, central 63.6% → 40.0%, descending 62.2% → 36.7%). That drop comes
from the activity patterns changing, not from information being lost: readouts trained on the
split brain are as good as on the intact brain, and each hemisphere alone is about as good as both.

Both eyes see the same digit, so each half of the brain gets the whole image and the link
between hemispheres isn't needed for this task. A real split-brain test would show each eye only
half the image.

### Camera → fly brain → fly body

The **Camera → fly body** tab closes the loop with a simulated body
([NeuroMechFly](https://neuromechfly.org/) via `flygym` 2.x, physics in MuJoCo):

1. **Start camera** (the browser asks for permission; it only talks to your local server).
2. **Record your own gesture for each fly action** you want, 4 s each (~30 frames). There are 10 actions:
   rest, right / left / both front legs up, wings out, flap wings, middle legs out, hind legs out,
   antennae up, wiggle antennae. Each button is the fly's action; make up any gesture for it, as
   long as it looks different from your other gestures. Record "rest" plus at least one action.
3. **Train**: every frame is shown to the fly's eyes (as local contrast, like lamina cells)
   and run through the brain on your device for 50 ms; the spike counts go to the server, where
   a logistic regression learns the gesture from the spikes of the chosen neurons (a few seconds).
   Default: only the **descending neurons** (brain → body commands). Accuracy is cross-validated
   on held-out stretches of each recording.
4. **Start live**: each new frame goes through the brain on your device (~0.2 s on a laptop), the decoder
   says which gesture it sees, and the fly does that action (an action switches on above 40%
   confidence and is fully on at 80%).

Click any action under the fly to try it without a camera (useful on students' devices).

The body is a tethered NeuroMechFly (thorax held still, legs under position control; the
model's wings and antennae have no muscles, so their angles are set directly; flapping is slowed ~70×).
Every visitor has their own fly body on the server: one shared MuJoCo model, one state per
visitor, stepped on 6 threads with a 0.5 ms physics step (5× flygym's default; same final poses).
Flies standing still are not simulated. The server streams the position of every body part
(~25 per second, server-sent events); the browser draws the fly with Three.js, so each viewer
can rotate and zoom it independently (3/4, Front, Side, Top, Spin). Load test: 30 flies all
flapping at once get the full 25 updates per second and use ~1.6 CPU cores of the M3 Max.
Code: [flymnist/camera.py](flymnist/camera.py), [flymnist/sessions.py](flymnist/sessions.py)
and [flymnist/body.py](flymnist/body.py). The class model is kept in `data/camera_session.joblib`
and students' sessions in `data/sessions/`, until **Clear recordings**.

This is a learned mapping: a real fly does not raise a leg when it sees a human hand. The
brain's wiring is untouched; only the decoder from its neurons to the leg is trained.

Options: `--layer {lamina,photoreceptors}`, `--rate` (Hz on a white pixel), `--ms`
(how long each digit is shown), `--batch`. Spike counts are saved to `data/spikes_*.npz`
and accuracies to `data/spikes_*.results.json`, so `uv run main.py train` can refit
readouts without simulating again.

## Results

3,000 training and 1,000 test digits, lamina input at 500 Hz, 100 ms per digit, M3 Max CPU
(~45 min per run). Test accuracy of a logistic-regression readout per region (chance = 10%):

| Readout from | Real fly wiring | Randomly rewired |
|---|---|---|
| Raw pixels, no brain (baseline) | 87.9% | 87.9% |
| Eye input (stimulated neurons) | 89.9% | 89.1% |
| Optic lobe | 90.9% | 90.2% |
| Visual projection neurons | 86.7% | 90.1% |
| Central brain | 63.6% | 90.8% |
| Descending neurons (brain → body) | 62.2% | 87.0% |
| **Whole brain** | **91.1%** | **90.1%** |

Cross-entropy loss for the real wiring (`uv run main.py train`; chance = ln 10 = 2.30):

| Readout from | Active neurons | Train acc | Test acc | Train loss | Test loss |
|---|---|---|---|---|---|
| Raw pixels | 784 | 96.1% | 88.4% | 0.21 | 0.42 |
| Optic lobe | 27,811 | 100% | 91.1% | 0.01 | 0.28 |
| Visual projection | 896 | 100% | 85.7% | 0.04 | 0.44 |
| Central brain | 1,220 | 87.7% | 63.6% | 0.53 | 1.10 |
| Descending | 283 | 76.2% | 62.2% | 0.84 | 1.18 |
| Whole brain | 30,305 | 100% | 91.4% | 0.01 | 0.28 |

What this shows:

- **The digit can be read from the fly brain**, even from its ~280 active descending neurons
  (62%, vs 10% chance), although those are 4+ synapses away from where the image enters.
- **The fly wiring is not a better MNIST reservoir than random wiring.** Whole-brain accuracy is
  the same (91.1% vs 90.1%, within noise for 1,000 test digits). Deeper regions do much
  *better* when rewired, because random wiring spreads activity everywhere (57k active neurons
  vs 30k), while the real brain keeps most visual activity inside the optic lobes.
- Small gain over pixels (+3 points), as expected from a big fixed nonlinear projection.
- Readouts with many neurons overfit (train loss ~0); more training digits would help.

MNIST is not something a fly needs to do, so it can't show what the wiring is good for.
Tasks with a natural behaviour (looming → escape, taste → feeding) are better tests.
Activating the looming detectors (LPLC2/LC4) already makes the escape neuron (giant fiber,
DNp01) fire at 170-190 Hz in this model, while walking and turning neurons stay silent.

## Limitations

- **Signal fades with depth.** The model has no spontaneous activity, and the lamina is only
  partly reconstructed in FlyWire. Driving photoreceptors alone barely reaches the central brain,
  which is why the default input also drives the lamina cells, at rates (500 Hz) well above
  biology. Real lamina cells use graded potentials, not spikes.
- **Approximate retinotopy.** The eye-to-image mapping uses neuron positions, not the true
  ommatidia/column map; the medulla is curved, so coverage of the image is uneven, and the
  optic chiasm's mirror flip is not modelled.
- **Static images.** Flies see motion; the optic lobe is tuned to motion and the model sees a
  frozen digit.
- **Small data.** Simulating is the bottleneck, so only a few thousand digits are used.

## Next steps

- **Train the wiring, keep the topology.** Use the surrogate gradient already in the fly-brain
  LIF (`ATan`) to learn synapse strengths while keeping which neurons connect and the signs
  fixed (connectome-constrained learning, as in
  [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3)).
- **Time-resolved readout.** Spike counts in short time bins instead of one total.
- **Behaviour as output.** Read out from a few descending neurons only, as if the fly had to
  "answer" with an action.
- **GPU.** On CUDA the repo's PyTorch/GeNN backends simulate much faster, which allows the
  full 70k-digit MNIST.

## Credits and license

The neuron model and connectome data come from
[eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (GPL-2.0-or-later),
based on Shiu et al. (MIT). FlyWire connectome v783: Dorkenwald et al. 2024, Schlegel et al. 2024.
Since [flymnist/brain.py](flymnist/brain.py) is adapted from GPL code, treat this project as
GPL-2.0-or-later if you distribute it.
