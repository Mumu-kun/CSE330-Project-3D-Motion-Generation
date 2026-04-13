# Presentation Goal

Build a redesigned 10-12 minute class presentation for classmates on the Human Motion Animation Generation project. The deck should follow a problem-to-solution story, stay visually clear and intuitive, and remain technically faithful to the intended system design without reading like a paper talk.

This document is a standalone design prompt and review spec. It is intentionally written so it can:

- support slide-by-slide review before building the final deck in any presentation tool
- reuse sections as prompts for AI-assisted slide drafting
- keep the story aligned with the latest agreed project description rather than older, stale notes

# Audience and Constraints

Audience: classmates in a machine learning course.

Assumed background:

- they understand general ML ideas like encoders, transformers, training loops, and conditioning
- they do not already understand motion representation formats or the exact architecture choices in this project
- they will respond better to intuitive visuals and a clean story than to dense derivations

Constraints:

- target length is 10-12 minutes
- target deck length is about 10 slides
- one main idea per slide
- avoid dense equations unless they are essential
- use cautious wording where implementation details are still evolving
- do not depend on final figures or demo assets already existing

# Narrative Arc

Story shape: problem to solution.

Narrative flow:

1. Start with the task and why it matters.
2. Explain why human motion generation is difficult.
3. Present the core idea as a clean architectural split.
4. Walk through the key representations and modules.
5. Show how generation works in practice.
6. End with outputs, limitations, and what comes next.

Speaker posture:

- sound like we are teaching the room, not defending a thesis
- explain design choices in plain language first
- only add implementation details when they help the audience understand why the design makes sense

# Visual Language Rules

Overall direction:

- clean, modern academic visuals with strong contrast and minimal clutter
- prefer diagrams, flow arrows, compact tables, and highlighted keywords over text walls
- use one accent color for the model pipeline, one for training, and one for results
- reserve bold color emphasis for only the most important words on a slide

Slide-level rules:

- each slide should have a single visual focal point
- cap on-slide text to short bullets or labeled fragments
- use progressive left-to-right flow for process slides
- use top-to-bottom hierarchy for concept slides
- keep captions short and readable from a classroom distance

Typography and layout cues:

- title at top with strong visual separation
- no more than 3-4 bullets on most slides
- if a slide has two columns, one column must be clearly dominant
- use icons, skeleton diagrams, or token diagrams where possible

Technical wording rules:

- do not claim unsupported results or ablations
- describe rollout-related behavior carefully, since training uses rollout-conditioned loss terms and validation currently also passes through a stochastic rollout path
- treat FK consistency as optional behavior rather than the center of the story
- prioritize the latest agreed architecture and training behavior
- if both predictor variants are mentioned, present them briefly as two design options rather than as a full ablation section

# Slide Specs

## Slide 1

**Slide title:** Human Motion Animation Generation

**Role in story:** Opening hook and framing slide.

**Audience takeaway:** This project turns text prompts into 3D human motion, and the talk will explain both the challenge and the system design in a simple way.

**On-slide content:**

- title: Human Motion Animation Generation
- subtitle: Text-conditioned motion generation with autoregressive context encoding and flow matching
- one short hook sentence: "How do we generate believable full-body motion from a text description?"
- presenter and course information

**Layout instructions:**

- use a full-width title slide
- place title and subtitle in the upper half
- reserve the lower half for a hero visual
- keep text light; this slide should feel open, not crowded

**Visual/asset placeholder:** A large hero image or stylized sequence of 3D pose silhouettes evolving left to right.

**Speaker guidance:** Spend about 20-30 seconds here. Frame the project as text-to-motion generation and preview that the solution combines temporal context encoding with next-frame motion prediction.

**Prompt block:**

```text
Design a presentation title slide for a machine learning class project called "Human Motion Animation Generation." The style should feel clean, modern, and academic. Include a subtitle that mentions autoregressive context encoding and flow matching, but keep the slide visually light. Use a large hero visual showing human pose progression or skeletal motion silhouettes. The slide should feel like an opening hook, not a dense information slide.
```

## Slide 2

**Slide title:** What Problem Are We Solving?

**Role in story:** Establish motivation and task definition.

**Audience takeaway:** The system maps a text description into a realistic sequence of full-body 3D poses, which is harder than ordinary sequence generation because the output must remain physically and structurally plausible.

**On-slide content:**

- one-sentence task definition
- example text prompts such as walking, turning, or waving
- short statement of desired output: a sequence of 22-joint poses over time
- one line on why this matters: animation, interactive systems, controllable generation

**Layout instructions:**

- use a two-column slide
- left column: short problem statement and example prompts
- right column: a single visual showing text input transforming into a pose sequence
- make the text-to-motion mapping visually obvious with arrows or labeled blocks

**Visual/asset placeholder:** Text prompt on the left, pose sequence strip on the right, connected by an arrow labeled "generate motion."

**Speaker guidance:** Emphasize that the output is not just one pose but a coherent temporal sequence. Avoid deep technical language here.

**Prompt block:**

```text
Create a slide for classmates explaining the problem of text-to-motion generation. The slide should define the task in simple terms and show example prompts such as walking forward, turning, or waving. Visually represent a text prompt turning into a sequence of 3D human poses. Keep the content intuitive and concise.
```

## Slide 3

**Slide title:** Why Is Human Motion Generation Hard?

**Role in story:** Build tension and justify the architecture.

**Audience takeaway:** Good motion requires both temporal consistency and body-structure awareness, so the model has to remember history while also predicting coordinated joint motion at the next step.

**On-slide content:**

- challenge 1: motion unfolds over time, so short-term and long-term context matter
- challenge 2: joints are coupled, so local errors create unrealistic full-body motion
- challenge 3: outputs must stay geometrically coherent across long sequences
- short summary sentence: "We need both memory and structure."

**Layout instructions:**

- use a three-panel challenge slide
- each panel should have one challenge, one icon, and one short explanation
- place the summary sentence as a bold footer

**Visual/asset placeholder:** Three cards labeled temporal context, joint coordination, and stable rollout, each with a simple icon or mini sketch.

**Speaker guidance:** This is the justification slide. The audience should leave understanding why a naive frame-by-frame or plain sequence model would struggle.

**Prompt block:**

```text
Design a challenge slide for text-to-motion generation. The slide should explain three difficulties: maintaining temporal consistency, modeling joint coordination, and keeping long motion rollouts stable. Use three visual cards or panels with short captions and simple icons. End with a bold takeaway that the model needs both memory and body structure awareness.
```

## Slide 4

**Slide title:** Core Idea: Split the Problem Into Two Jobs

**Role in story:** Introduce the main architectural insight.

**Audience takeaway:** Instead of forcing one model to do everything, the system separates temporal history encoding from spatial next-frame prediction.

**On-slide content:**

- job 1: encode motion history with text conditioning
- job 2: predict the next-frame motion state in a reduced space
- brief note that two predictor-state versions may be mentioned: a compact reduced state or a rotation-based state decoded with FK
- conversion loop: translate between compact model states and full geometric motion
- takeaway line: "Temporal context and spatial prediction are handled by different components."

**Layout instructions:**

- make this the first major architecture diagram
- use a left-to-right pipeline with 4 labeled stages: encode, predict, convert, append
- visually separate the encoder and predictor as different colored modules
- show text conditioning entering the pipeline from above

**Visual/asset placeholder:** High-level pipeline diagram with history frames, text embedding, encoder block, predictor block, position reconstruction, and updated history.

**Speaker guidance:** This should be the cleanest slide in the deck. Focus on the architecture split, not implementation detail. Mention that this reduces the burden on any single module.

**Prompt block:**

```text
Create a high-level architecture slide for a text-conditioned human motion generation system. The core message is that the problem is split into two jobs: a temporal history encoder and a next-frame flow-matching predictor, connected by a conversion loop between compact motion states and full joint geometry. Use a clean left-to-right pipeline diagram with minimal text and clear module separation.
```

## Slide 5

**Slide title:** What Representation Does the Model Use?

**Role in story:** Explain the compact state design without drowning the audience in raw dimensions.

**Audience takeaway:** The system uses different motion representations for different purposes: a richer full-frame format for history and one of two predictor-state options for next-frame prediction.

**On-slide content:**

- 271D full frame: used for history, dataset features, and reconstruction context
- 68D reduced state: used as the predictor target space
- 135D rotation-based state: alternative predictor target using root y, root vx, root vz, and 22 joints x 6D rotations
- 257D current-frame conditioning: used to tell the predictor what the latest frame looks like
- short intuition line: "Use rich features for context, and choose a prediction state that balances compactness and geometric control."

**Layout instructions:**

- use a comparison table or stacked card layout
- do not list every dimension in dense prose
- highlight only the main semantic parts:
  - root motion
  - joint-relative structure
  - rotations and velocities where relevant
- include one small callout that the 68D version is more compact, while the 135D version predicts rotations directly and reconstructs motion through FK

**Visual/asset placeholder:** A compact representation diagram comparing 271D history features, 68D and 135D predictor target options, and 257D current-frame conditioning.

**Speaker guidance:** Keep this approachable. Mention the dimension counts, but explain the design tradeoff at a high level: one predictor version is more compact, while the other predicts root motion and joint rotations more directly.

**Prompt block:**

```text
Design a slide that explains the motion representations used in a text-to-motion model. Show that the system uses a 271D full-frame representation for history, two predictor target options for next-frame prediction (a compact 68D reduced state and a 135D rotation-based state with root y, root vx, root vz, and 22 joints x 6D rotations), and a 257D conditioning vector from the current frame. The slide should emphasize intuition and purpose rather than overwhelming detail. Use a clean comparison layout.
```

## Slide 6

**Slide title:** Motion History Encoder

**Role in story:** Explain how the model remembers the past.

**Audience takeaway:** The history encoder summarizes a variable-length motion context, injects text conditioning, and can be framed in two versions: a GRU-based memory encoder or a transformer-based temporal encoder.

**On-slide content:**

- input: motion history plus pooled text embedding
- two encoder versions may be mentioned briefly: GRU-based sequence encoding and transformer-based temporal attention
- conditioning: text modulates the sequence representation
- output: joint-level context tokens for the predictor

**Layout instructions:**

- use a left-to-right mini pipeline inside the slide
- start with a stack of recent frames
- show text entering as a conditioning stream
- if both versions are shown, keep them as two compact encoder options rather than two separate full slides
- end with 22 context tokens or joint-labeled blocks
- keep equations out unless absolutely necessary

**Visual/asset placeholder:** Diagram showing history frames flowing through either a GRU-based encoder or a transformer-based temporal encoder, with text conditioning merged in, producing per-joint context tokens.

**Speaker guidance:** Explain the encoder as the memory module. It should feel intuitive: read the past, compress what matters, pass that context forward. If both versions are mentioned, describe the GRU version as a simpler recurrent memory path and the transformer version as a more expressive temporal-context path.

**Prompt block:**

```text
Create a slide explaining a motion history encoder for a text-conditioned motion generation model. Briefly allow two versions of the encoder: a GRU-based sequence encoder and a transformer-based temporal attention encoder. The slide should show recent motion frames entering the encoder, with text conditioning injected into the sequence, and joint-level context tokens coming out. Focus on the intuition that this module remembers the motion history for the predictor.
```

## Slide 7

**Slide title:** Flow Matching Predictor

**Role in story:** Explain how the next frame is predicted once context is available.

**Audience takeaway:** The predictor operates over joint tokens, uses time and text conditioning, and predicts how to denoise a compact next-frame motion state.

**On-slide content:**

- predictor sees noisy next-frame state features
- it also receives encoder context and current-frame conditioning
- transformer layers operate across joints to model body structure
- output is a flow or velocity field in the chosen next-frame state space
- brief comparison note: one version predicts a compact reduced state, while another predicts root motion plus 6D joint rotations and relies on FK for reconstruction

**Layout instructions:**

- use a center-focused block diagram
- show three input streams converging:
  - noisy next-frame state
  - encoder context
  - current-frame features
- indicate time and text conditioning as side inputs to the transformer stack
- include one small note that kinematic structure is built into joint processing

**Visual/asset placeholder:** Token diagram with root token and non-root joint tokens, plus side arrows for time and text conditioning.

**Speaker guidance:** Avoid spending too long on flow matching math. Explain it as learning how to move a noisy sample toward the correct next-frame state. If both predictor versions are mentioned, keep the distinction brief and intuitive rather than turning the slide into a detailed comparison.

**Prompt block:**

```text
Design a slide for classmates explaining a flow-matching predictor in a human motion model. Show that the predictor takes a noisy next-frame state, history context from the encoder, and current-frame features, then predicts a denoising direction for the next frame. Briefly acknowledge two predictor variants: one that predicts a compact reduced state and one that predicts root motion plus 6D joint rotations, with motion reconstructed through FK. Visually emphasize joint-token interaction, time conditioning, and text conditioning, but keep the explanation intuitive rather than mathematical.
```

## Slide 8

**Slide title:** How Generation Works

**Role in story:** Connect the architecture to actual usage.

**Audience takeaway:** At inference time, the system starts from motion history and text, predicts the next-frame state, reconstructs motion, and feeds the new frame back into history to continue generation.

**On-slide content:**

- start from current motion history plus text conditioning
- run the predictor through an ODE-style denoising process to obtain the next-frame state
- reconstruct motion, either through the compact-state conversion path or through rotation decoding plus FK
- append the generated frame back into history
- repeat autoregressively to extend the motion sequence

**Layout instructions:**

- use a single inference-focused process diagram
- prefer a left-to-right pipeline or a circular loop with one clear focal path
- emphasize the autoregressive feedback from the generated frame back into history
- use arrows and short captions instead of paragraphs

**Visual/asset placeholder:** A clean inference-loop diagram showing history plus text, next-frame prediction, motion reconstruction, history update, and repeat.

**Speaker guidance:** Keep this slide focused on how the model is used at generation time. Avoid training details here. If both predictor variants are mentioned, keep them as a brief note at the reconstruction step rather than as separate pipelines.

**Prompt block:**

```text
Create a slide for a class presentation explaining inference in a text-conditioned motion generation model. Show a clean generation loop: start from motion history plus text, predict the next-frame state through a denoising process, reconstruct motion, append the new frame back into history, and repeat autoregressively. Briefly allow for two reconstruction paths: a compact reduced-state conversion path and a rotation-plus-FK path. Keep the slide visual, simple, and process-oriented.
```

## Slide 9

**Slide title:** What Does the Model Produce?

**Role in story:** Show outputs and make the project feel real.

**Audience takeaway:** The system can generate motion sequences from text prompts, and evaluation should focus on realism, continuity, and prompt alignment.

**On-slide content:**

- one short sentence describing what is being shown
- 2-3 example prompt labels
- side-by-side placeholders for generated motion and, if available, reference motion or teacher-forced comparisons
- one line explaining what the audience should look for: smoothness, coordination, plausibility

**Layout instructions:**

- let visuals dominate this slide
- use a grid or side-by-side comparison layout
- keep text minimal and supportive
- if videos are unavailable, design this as a film-strip or frame-sequence slide

**Visual/asset placeholder:** Demo comparison panel such as generated rollout versus reference, or multiple prompts with pose strips.

**Speaker guidance:** Narrate what looks good and what still breaks. This slide should feel honest and concrete.

**Prompt block:**

```text
Design a results slide for a human motion generation class project. The slide should be visually dominant, showing generated motion examples or comparison panels. Include short prompt labels and a small note telling the audience to look for realism, continuity, and prompt alignment. If full video is not available, use pose strips or storyboard-like frame sequences.
```

## Slide 10

**Slide title:** Takeaways, Limitations, and Next Steps

**Role in story:** Close the deck with a balanced summary.

**Audience takeaway:** The project’s main contribution is the clean separation of temporal memory and next-frame structured prediction, but there is still room to improve rollout stability, evaluation, and presentation polish.

**On-slide content:**

- takeaway 1: temporal encoding and spatial prediction are separated cleanly
- takeaway 2: compact state prediction helps keep the generation problem manageable
- limitation: long-horizon realism and evaluation remain important challenges
- next steps: stronger demos, clearer metrics, more robust rollout analysis

**Layout instructions:**

- use a three-zone closing slide:
  - key takeaways
  - limitations
  - next steps
- keep it concise and readable
- end with a simple closing line or Q&A prompt at the bottom

**Visual/asset placeholder:** Minimal closing graphic such as the high-level pipeline reduced to a compact summary icon strip.

**Speaker guidance:** End confidently but honestly. Make the audience remember the architectural split and the practical path forward.

**Prompt block:**

```text
Create a closing slide for a class presentation on human motion generation. The slide should summarize the main idea, acknowledge current limitations, and suggest clear next steps. Keep it concise, balanced, and visually clean, with a small Q&A or closing prompt at the bottom.
```

# Asset Checklist

Assets to collect or create for the final deck:

- title-slide hero visual with pose silhouettes or skeleton progression
- text-to-motion mapping visual for the problem slide
- three-panel challenge icons or sketches
- high-level pipeline diagram for the core idea slide
- compact representation diagram for 271D history features, 68D and 135D predictor-state options, and 257D conditioning features
- encoder diagram with history frames and context tokens
- predictor diagram with joint tokens, conditioning streams, and a brief note on the two output-state variants
- inference-loop process diagram
- generated motion demos, pose strips, or comparison frames
- closing mini graphic or simplified pipeline summary

Nice-to-have supporting assets:

- one clean skeleton graphic with joint labels
- one color-coded legend for representations and modules
- one visual style reference so the full deck stays consistent

# Revision Notes

Version 1 focus:

- establish the new story arc before building the final slides
- review whether the 10-slide structure feels right
- decide which slides deserve actual diagrams versus lightweight placeholders
- check whether any training-language should be simplified further for classmates

Known wording cautions:

- older project notes may describe outdated architecture details and should not be used as the main source of truth
- rollout behavior should be described carefully and not oversold
- results language should remain modest until final demo assets and metrics are selected

Next review questions for iteration:

- which slide should become the visual centerpiece of the talk
- whether the results slide should be one slide or split into demo plus observations
- whether to include one tiny equation or keep the whole deck equation-free
