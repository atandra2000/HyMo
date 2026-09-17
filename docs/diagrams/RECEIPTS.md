# Archify delivery evidence: HyMo

Content repair pass, 2026-09-17. Source revision `a88006236ad8a96f17f86298da7fa9afb8c844ad`, configuration `configs/hymo_750m.yaml`.

All four artifacts passed 9/9 showcase checks with zero composition errors/warnings and fresh automated browser checks. Browser coverage measures light-theme containment at four desktop sizes and captures both endpoint sizes in light/dark.

Perceptual review: skipped (image reader unavailable). The attempted image read was rejected by this model route. The proposed 12px/11px typography standard, interaction/export checks and narrow-screen guide review are not certified.

## hymo-architecture
- [hymo-architecture.html](hymo-architecture.html) (architecture)
- Spec SHA-256: `7daf37fd2c3144675e9f8e077833bbc1b032f62bfd01951fe9da6b97eabc2f09` (3012 bytes)
- Artifact SHA-256: `f9be95cc496ab4141a576fa8ca9895d7617b834d83649fc35bd9ed49da0eab37` (709335 bytes)
- [Fresh browser receipt](hymo-architecture.visual-check.json)
- Validation: 9/9 showcase, zero errors/warnings. Browser: passed. Perception: skipped.

## hymo-dataflow
- [hymo-dataflow.html](hymo-dataflow.html) (dataflow)
- Spec SHA-256: `95ee577dc636b2a2c25c5a989e354584d5428fbfbc80ab86d4a1e79901505d11` (5235 bytes)
- Artifact SHA-256: `c6716875c6a94fd74527c7e51e68fdfd5a681aa99c1dd103cf441ae7f09353f1` (713995 bytes)
- [Fresh browser receipt](hymo-dataflow.visual-check.json)
- Validation: 9/9 showcase, zero errors/warnings. Browser: passed. Perception: skipped.

## hymo-training
- [hymo-training.html](hymo-training.html) (workflow)
- Spec SHA-256: `6e18ee2692d0a46e695892259e9d05ba83e702d10655f0b5051c29525fa80bcd` (6602 bytes)
- Artifact SHA-256: `2da893beb41e01838e57735ddd5108f0c308bf1f44030bcef34d5bcbad2f27f7` (718786 bytes)
- [Fresh browser receipt](hymo-training.visual-check.json)
- Validation: 9/9 showcase, zero errors/warnings. Browser: passed. Perception: skipped.

## hymo-optimizations
- [hymo-optimizations.html](hymo-optimizations.html) (architecture)
- Spec SHA-256: `4dbdf91bded53d586f9e4a5f0ae9c2add1faf1c694c3738a5af60c2dd84d376e` (5598 bytes)
- Artifact SHA-256: `cafa21f47bab71fced609849f1c30f26563f6ac15431d549d7f17b17f073befc` (715338 bytes)
- [Fresh browser receipt](hymo-optimizations.visual-check.json)
- Validation: 9/9 showcase, zero errors/warnings. Browser: passed. Perception: skipped.

## Evidence boundaries

Topology, configured dimensions, MTP weights, data integration gaps, loss/optimizer clocks and checkpoint format were checked against source. Model code was not changed. Parameter counts, corpus inventory, completed pretraining, exact resume and GPU speedups are not established by these artifacts.
