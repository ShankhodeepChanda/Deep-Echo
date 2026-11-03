---
name: SETI-AI Observatory
description: A dark charcoal instrument console where one radio observation is a time-frequency strip and unusual candidates are amber marks pinned to it.
colors:
  ground: "#0f1113"
  panel: "#15181b"
  panel-raised-1: "#1b1f23"
  panel-raised-2: "#23282c"
  hairline: "#2a3035"
  hairline-soft: "#21262a"
  text: "#d9dee2"
  text-strong: "#f2f5f7"
  text-muted: "#9aa4ac"
  text-faint: "#7b868f"
  teal: "#35b8a8"
  teal-bright: "#63d3c4"
  teal-on-fill: "#07201d"
  amber-mark: "#e8a33d"
  amber-text: "#f2c177"
  danger: "#e5675f"
  danger-text: "#f0a49e"
typography:
  display:
    fontFamily: "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, -apple-system, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.005em"
  headline:
    fontFamily: "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.25
  title:
    fontFamily: "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 500
    lineHeight: 1.45
  measurement:
    fontFamily: "'JetBrains Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"
    fontSize: "0.875rem"
    fontWeight: 400
    fontFeature: "tabular-nums"
rounded:
  sm: "4px"
  lg: "6px"
  pill: "999px"
spacing:
  xs: "0.25rem"
  sm: "0.5rem"
  md: "0.9rem"
  lg: "1.1rem"
  page-x: "2rem"
components:
  button-default:
    backgroundColor: "{colors.panel-raised-1}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "0.4rem 0.9rem"
    height: "2.35rem"
  button-default-hover:
    backgroundColor: "{colors.panel-raised-2}"
    textColor: "{colors.text-strong}"
  button-primary:
    backgroundColor: "{colors.teal}"
    textColor: "{colors.teal-on-fill}"
    rounded: "{rounded.sm}"
    padding: "0.4rem 0.9rem"
    height: "2.35rem"
  button-primary-hover:
    backgroundColor: "{colors.teal-bright}"
    textColor: "{colors.teal-on-fill}"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: "0.9rem 1.1rem"
  chip-status:
    backgroundColor: "{colors.panel-raised-1}"
    textColor: "{colors.text}"
    rounded: "{rounded.pill}"
    padding: "0.18rem 0.6rem"
  badge:
    backgroundColor: "{colors.panel-raised-1}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "0.12rem 0.5rem"
  nav-tab:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    padding: "0.55rem 1.05rem"
    height: "2.6rem"
  nav-tab-active:
    textColor: "{colors.text-strong}"
---

# Design System: SETI-AI Observatory

## Overview

**Creative North Star: "The Observation Strip"**

The observation is a strip and candidates are marks pinned to it. Every working page is organised around a full-band time-frequency strip with candidate marks at their measured coordinates; the rest of the interface is a quiet charcoal instrument console that exists to read, triage, annotate and export. It is dense and calm: near-black ground, one step lighter panels, hairline borders, no decorative surfaces.

Color is rationed by meaning. One teal carries selection, primary action and signal intensity. Amber is reserved for unusual-signal marks and review flags, so that amber on screen always means "look here". A researcher should never be told a high score means extraterrestrial; the surface states what was measured and what is still unknown.

The shell is the standard Operate layout (sidebar, segmented page tabs as an underlined tab row, work area). The strip is the signature, not the shell. The Streamlit theme in `.streamlit/config.toml` mirrors the same ground, panel, text, border, teal and 4px radius values.

**Key Characteristics:**
- Charcoal neutrals with hairline borders; flat surfaces, no gradients.
- One teal for selection, action and intensity; amber only for unusual marks and flags.
- One humanist sans for all prose and labels; monospace only for measurement values.
- Spectrograms use a single-hue dark-to-teal-to-pale ramp, never a rainbow.
- Small radii (4px controls, 6px containers); one popover shadow, nowhere else.

## Colors

A near-black charcoal ground with a single cool teal and a single warm amber, each bound to one meaning.

### Primary
- **Signal Teal** (#35b8a8): primary buttons, active tab underline, selected sidebar segments (as a 14% tint with a 50% border), slider handle, progress fill, retained-review state, heatmap intensity.
- **Signal Teal Bright** (#63d3c4): links, focus outline (2px, 2px offset), hover state of primary actions, text on teal tints.

### Secondary
- **Unusual-Signal Amber** (#e8a33d): candidate marks on the strip and waterfall (open diamond, 2px stroke), the coverage and highlight rectangles, flagged-review pills, the unusual badge and legend diamond. Amber text on dark uses #f2c177. Amber never decorates; it only marks.

### Tertiary
- **Alert Coral** (#e5675f): RFI badge border and error callouts (tint rgba(229,103,95,0.12)); text variant #f0a49e. Error and interference only.

### Neutral
- **Charcoal Ground** (#0f1113): page and app background, plot interior.
- **Panel** (#15181b): sidebar, panels, callouts, expanders, plot frames.
- **Raised Panel** (#1b1f23): buttons, chips, badges, inputs, step tiles.
- **Hover Raise** (#23282c): hover fill for buttons and list options.
- **Hairline** (#2a3035) and **Soft Hairline** (#21262a): borders and row dividers.
- **Text** (#d9dee2), **Strong Text** (#f2f5f7), **Muted** (#9aa4ac), **Faint** (#7b868f): body, headings and values, labels and captions, disabled and not-available.
- **Intensity ramp** (heatmap): #0e1214, #12353a, #1f7a78, #6fd3c4, #eafaf6 at 0, 0.3, 0.62, 0.85, 1.0.

### Named Rules
**The Amber Means Look Rule.** Amber appears only on unusual-signal marks and review flags. A count, a heading, a button or a decoration is never amber; if every row is amber, nothing is.
**The One Teal Rule.** Teal marks selection, the primary action and signal intensity. It is not a second decorative accent.

## Typography

**Body and Display Font:** Atkinson Hyperlegible Next (with Segoe UI, system-ui, sans-serif)
**Measurement Font:** JetBrains Mono (with ui-monospace, Cascadia Mono, Consolas)

**Character:** A highly legible humanist sans does all the work; monospace is a narrow instrument readout face, not a theme. Atkinson Hyperlegible Next renders zeros slashed in prose; this is a known, deliberate adaptation and is accepted.

### Hierarchy
- **Display** (600, 1.5rem, 1.2): page title.
- **Headline** (600, 1.25rem, 1.25): lede summary line, section headings, candidate detail title.
- **Title** (600, 1.0625rem, 1.3): panel titles, callout titles, wordmark (700).
- **Body** (400, 0.9375rem, 1.55): prose, controls, tab labels (500); callout and lede-sub text capped near 78-80ch.
- **Label** (500, 0.8125rem): widget labels, facts, captions, badges, sidebar section heads; sentence case, no letter-spacing, no uppercase.
- **Measurement** (400, 0.8125-0.875rem, tabular figures): frequency, time and score values only.

### Named Rules
**The Mono Is Measurement Rule.** JetBrains Mono appears only on measurement values (frequency, time, score) and inline code. Prose, labels and headings stay in the sans.

## Layout

Operate shell: left sidebar (observation, sensitivity, upload, advanced), then a work area capped at 1480px with 2rem side padding (1rem under 1100px). Order on working pages: page title row with status chips, underlined tab row (Overview, Candidates, Waterfall, Cadence, Compare, Export), the full-width observation strip (about 190px), one dense facts line, then the ranked candidate list or page content. Two-pane rows (list plus detail) stack below 1320px; key-value and evidence grids collapse below 900px. Spacing is rem-based and tight: 0.25 to 0.5rem inside controls, about 0.9rem between blocks, 1.1rem panel padding. Density over air.

## Elevation & Depth

Flat. Depth comes from tonal steps (ground, panel, raised panel, hover raise) and hairline borders, never from shadow. The single exception is the popover body (`box-shadow: 0 8px 24px rgba(0,0,0,0.45)`). Interactive hover is a fill step plus a border brightening, not a lift.

### Named Rules
**The Flat Surfaces Rule.** Surfaces carry a 1px hairline and a tonal step. No other shadows, no gradients.

## Shapes

Quiet rectangles: 4px on controls, badges, inputs, segment ends; 6px on panels, callouts, expanders, plot frames, tables and dropzones; full pill only on status chips and review pills. The one distinctive silhouette is the open amber diamond mark (also the legend glyph). Dashed borders mean pending, insufficient or preliminary; solid means resolved.

## Components

### Buttons
- **Shape:** 4px radius, 2.35rem min height, 0.4rem 0.9rem padding.
- **Default:** raised panel fill (#1b1f23), hairline border, text weight 500; hover steps to #23282c with a #7b868f border.
- **Primary:** teal fill (#35b8a8), dark teal text (#07201d), weight 600; hover to bright teal. One primary per view ("Review top candidate" on first view).
- **Disabled:** panel fill, faint text, soft hairline.
- **Focus:** 2px bright teal outline, 2px offset, all elements. Labels are never truncated.

### Navigation
Page tabs are a segmented control restyled as an underlined row: transparent, muted text, 2px transparent bottom border; active is strong text, weight 600, teal underline. Hover fills panel-raised. In the sidebar the same control becomes a boxed equal-width segment group with teal tint on the selected segment.

### Chips and badges
- **Status chip:** pill with a 7px dot: ok teal, warn a hollow muted ring, bad coral.
- **Badge:** 4px, hairline; RFI coral border, artefact grey, unusual amber border with amber text, insufficient and preliminary dashed.
- **Review pill:** retained teal tint, flagged amber tint, dismissed struck through muted, pending dashed.

### Panels, callouts, facts
Panel and callout: panel fill, hairline, 6px, with info (teal tint), warn (raised fill, faint border) and bad (coral tint) variants. Facts line: horizontal metadata with muted labels over strong values between two hairlines. Evidence and key-value lists: two-column grids with soft hairline row dividers.

### Inputs
Raised panel fill, hairline border, 4px radius, border brightens to faint on hover, muted 0.8125rem labels. File dropzone is dashed with a teal-tinted border on hover.

### Observation Strip (signature)
Full-width time-frequency heatmap in the dark-to-teal ramp inside a plot frame (panel fill, hairline, 6px), with candidates drawn as open amber diamonds (size 11, 2px stroke) at measured time and frequency, a time axis in elapsed seconds, and the observation's coverage footprint shown. The waterfall reuses the same ramp with amber rectangle outlines (2px, transparent fill) for selected windows. Plot text is the sans at 12px muted, gridlines rgba(255,255,255,0.06).

## Do's and Don'ts

### Do:
- **Do** keep the strip at the top of every working page with marks at their measured coordinates.
- **Do** reserve amber (#e8a33d) for marks and review flags, and tell mark from count by shape: marks are diamonds, counts are plain strong text.
- **Do** use teal for selection, the primary action and intensity only, with dark text (#07201d) on teal fills.
- **Do** separate surfaces with a tonal step and a 1px hairline; use 4px and 6px radii.
- **Do** set measurement values in JetBrains Mono with tabular figures and everything else in Atkinson Hyperlegible Next.
- **Do** use dashed borders for pending, insufficient and preliminary states, and state what is unknown.

### Don't:
- **Don't** use a KPI-tile grid or an ML-metrics tab as the front door; model internals stay behind Advanced.
- **Don't** use rainbow or multi-hue colormaps for spectrograms.
- **Don't** add shadows beyond the popover, gradients on surfaces, or radii above 6px (except pills).
- **Don't** imply a high score means extraterrestrial origin or show fabricated numbers.
- **Don't** make a heading, count, tab or decoration amber.

Not canonized (defects/drift the build carries): the Python constant COLOR_RFI (#e0a458) and the dev-lab threshold lines in ui/dev_lab.py use an amber-adjacent tone for RFI and thresholds, which contradicts the Amber Means Look Rule; the dev lab and COLORWAY also use extra hues (#8fa3b0, #c97b84, #7fb069, #b48ead) outside the palette; the Google Fonts @import is a runtime dependency with system fallbacks.
