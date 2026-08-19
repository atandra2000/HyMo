/* HyMo Documentation Portal Client-Side Behaviors
   ======================================================================
   • Hybrid Architecture State Dynamics (FIG · A0 — GDN ⇄ MLA ⇄ MoE ⇄ MTP ⇄ 3:1 Hybrid)
   • Live Full 32-Layer Training Step Pipeline Telemetry (FIG · A1)
   • Interactive Mechanism Benchmark Labs (GDN Recurrence, MoE Router, MLA Cache)
   • Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
   • Navigation Filtering & Sidebar Mobile Toggle
   • Table of Contents Scrollspy
   • Highlight.js & KaTeX bootstrap
*/

(function () {
    'use strict';

    // ------------------------------------------------------------------
    // Model constants — mirror configs/hymo_750m.yaml + src/hymo/core/config.py.
    // Single source of truth for every numeric label/math expression below.
    // ------------------------------------------------------------------
    var MODEL = {
        dim: 896,
        n_layers: 32,
        n_heads: 16,
        n_kv_groups: 4,
        head_dim: 128,
        kv_lora_rank: 128,
        qk_rope_head_dim: 32,
        qk_nope_head_dim: 96,
        v_head_dim: 128,
        gdn_d_state: 32,
        gdn_d_inner: 1280,
        gdn_headdim: 32,
        gdn_d_conv: 4,
        moe_inter_dim: 2304,
        n_routed_experts: 16,
        n_shared_experts: 1,
        n_activated_experts: 2,
        mtp_depth: 2,
        mtp_loss_weights: [0.3, 0.1],
        vocab_size: 64256,
        max_seq_len: 4096,
        micro_batch_size: 4,
        logit_softcap: 15.0
    };
    MODEL.n_mla_layers = 8;   // n_layers // 4 (positions 0,4,8,12,16,20,24,28)
    MODEL.n_gdn_layers = MODEL.n_layers - MODEL.n_mla_layers;
    MODEL.gdn_n_heads = MODEL.gdn_d_inner / MODEL.gdn_headdim;  // 40

    var reduced = window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ------------------------------------------------------------------
    // 1. Hybrid State Dynamics & Phase Space Manifold (FIG · A0)
    //    Interactive high-DPI polar geometry oscilloscope & state dynamics:
    //    - GDN: Linear-attention 1D selective scan with chunked recurrence
    //    - MLA: Multi-Head Latent Attention with low-rank KV compression
    //    - MoE: Asymmetric 16+1 expert routing on MLA layers
    //    - MTP: Multi-Token Prediction depth-2 speculative drafting
    //    - Hybrid: Full 3:1 interleaved 32-layer macro stack
    // ------------------------------------------------------------------
    var heroController = {
        setMode: null,
        triggerPulse: null
    };

    function initHeroCanvas() {
        var canvas = document.getElementById('heroStateCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var probeEl = document.getElementById('heroProbeHUD');
        var probeTag = document.getElementById('probeTag');
        var probeCoords = document.getElementById('probeCoords');
        var probeDecay = document.getElementById('probeDecay');
        var modeBtns = document.querySelectorAll('.hero-figure .fig-btn[data-mode]');
        var speedBtn = document.getElementById('heroSpeedBtn');
        var pauseBtn = document.getElementById('heroPauseBtn');
        var modeLabel = document.getElementById('hudModeLabel');
        var kvCut = document.getElementById('hudKVCut');
        var moeStatus = document.getElementById('hudMoEStatus');
        var overlap = document.getElementById('hudOverlap');
        var formulaBar = document.getElementById('heroFormulaBar');

        var currentMode = 'gdn'; // 'gdn' | 'mla' | 'moe' | 'mtp' | 'hybrid'
        var simSpeed = 1.0;
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;
        var flashEffect = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperCenter: '#181410',
            gridRule: 'rgba(58, 50, 38, 0.45)',
            gridAxis: 'rgba(201, 163, 92, 0.35)',
            unitCircle: 'rgba(201, 163, 92, 0.22)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.65)',
            oliveTint: 'rgba(154, 148, 64, 0.15)',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.65)',
            terracottaTint: 'rgba(224, 122, 63, 0.15)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.7)',
            coreHot: '#fffaf0',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var shockwaves = [];
        var particles = [];
        var numParticles = 54;
        for (var p = 0; p < numParticles; p++) {
            particles.push({
                radiusFrac: 0.15 + 0.80 * Math.random(),
                theta: Math.random() * Math.PI * 2,
                speed: (0.2 + 0.6 * Math.random()) * (Math.random() > 0.5 ? 1 : -1),
                size: 1.2 + 1.4 * Math.random(),
                alpha: 0.3 + 0.5 * Math.random(),
                lane: Math.floor(Math.random() * 4)
            });
        }

        // 16 GDN Head / State Channels
        var CHANNELS_COUNT = 16;
        var channels = [];
        for (var i = 0; i < CHANNELS_COUNT; i++) {
            var frac = i / (CHANNELS_COUNT - 1);
            var baseRadius = 0.22 + 0.72 * Math.pow(frac, 0.85);
            var omega = 0.35 + 1.85 * Math.pow(1 - frac, 1.15);
            var phase0 = (i * 2.399963229728653) % (Math.PI * 2);
            channels.push({
                id: i,
                r: baseRadius,
                omega: omega,
                phase: phase0,
                decay: 0.15 + 0.7 * (i / CHANNELS_COUNT),
                theta: phase0,
                x: 0, y: 0
            });
        }

        var width = 0, height = 0, cx = 0, cy = 0, radius = 0;

        function resize() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 420;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            cx = width / 2;
            cy = height / 2;
            radius = Math.min(width, height) * 0.43;
        }

        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                resize();
                if (reduced || isPaused) renderFrame(0, true);
            }).observe(container);
        } else {
            window.addEventListener('resize', resize);
        }
        resize();

        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999; mouseY = -9999;
            if (probeEl) probeEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;
            triggerPulse(clickX, clickY);
        });

        function triggerPulse(originX, originY) {
            var ox = originX !== undefined ? originX : cx;
            var oy = originY !== undefined ? originY : cy;
            shockwaves.push({
                x: ox,
                y: oy,
                radius: 4,
                maxRadius: radius * 0.95,
                life: 1.0,
                color: currentMode === 'moe' ? PALETTE.terracotta : (currentMode === 'mtp' ? PALETTE.gold : (currentMode === 'mla' ? PALETTE.olive : PALETTE.terracotta))
            });
        }

        function setMode(mode) {
            currentMode = mode;
            modeBtns.forEach(function (b) {
                b.classList.toggle('active', b.getAttribute('data-mode') === mode);
            });
            updateFormulaAndLabels();
            triggerPulse(cx, cy);
            if (reduced || isPaused) renderFrame(0, true);
        }

        heroController.setMode = setMode;
        heroController.triggerPulse = triggerPulse;

        modeBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                var m = btn.getAttribute('data-mode') || 'gdn';
                setMode(m);
            });
        });

        if (speedBtn) {
            speedBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (simSpeed === 1.0) simSpeed = 2.0;
                else if (simSpeed === 2.0) simSpeed = 0.5;
                else simSpeed = 1.0;
                speedBtn.textContent = simSpeed + '×';
                speedBtn.setAttribute('aria-label', 'Speed: ' + simSpeed + 'x');
            });
        }

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume simulation' : 'Pause simulation');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(renderLoop);
                }
            });
        }

        function updateFormulaAndLabels() {
            if (currentMode === 'gdn') {
                if (modeLabel) modeLabel.textContent = 'GDN · 1D SELECTIVE SCAN (O(N))';
                if (kvCut) kvCut.innerHTML = 'O(1) STATE <span class="unit">(O(N) COMPUTE)</span>';
                if (moeStatus) moeStatus.innerHTML = '24 LAYERS <span class="num">(DENSE SwiGLU)</span>';
                if (overlap) overlap.innerHTML = 'FUSED TRITON <span class="unit">(CHUNK 64)</span>';
                if (formulaBar) {
                    formulaBar.innerHTML = '<span class="formula-sym">h<sub class="f-sub">t</sub></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-a">(1 - &beta;<sub class="f-sub">t</sub> q<sub class="f-sub">t</sub> k<sub class="f-sub">t</sub><sup class="f-sub">&top;</sup>) h<sub class="f-sub">t-1</sub> + &beta;<sub class="f-sub">t</sub> v<sub class="f-sub">t</sub> k<sub class="f-sub">t</sub><sup class="f-sub">&top;</sup></span> ' +
                        '<span class="formula-dot">&middot;</span> ' +
                        '<span class="formula-sym">&alpha;<sub class="f-sub">t</sub></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-b">exp(g<sub class="f-sub">t</sub> A)</span>';
                }
            } else if (currentMode === 'mla') {
                if (modeLabel) modeLabel.textContent = 'MLA · 4 KV GROUPS LATENT ATTN';
                if (kvCut) kvCut.innerHTML = '16&times; <span class="unit">(4096 &rarr; 256 DIMS / LAYER)</span>';
                if (moeStatus) moeStatus.innerHTML = '8 LAYERS <span class="num">(MQA-4 + ROPE 25%)</span>';
                if (overlap) overlap.innerHTML = 'LOW-RANK KV <span class="unit">(d_c = 128)</span>';
                if (formulaBar) {
                    formulaBar.innerHTML = '<span class="formula-sym">c<sub class="f-sub">t</sub><sup class="f-sub">KV</sup></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-a">W<sub class="f-sub">DKV</sub> h<sub class="f-sub">t</sub> [128d]</span> ' +
                        '<span class="formula-dot">&middot;</span> ' +
                        '<span class="formula-sym">Attn</span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-b">Softmax(q<sub class="f-sub">i</sub> k<sub class="f-sub">&lfloor;i/4&rfloor;</sub><sup class="f-sub">&top;</sup> / &radic;d) v<sub class="f-sub">&lfloor;i/4&rfloor;</sub></span>';
                }
            } else if (currentMode === 'moe') {
                if (modeLabel) modeLabel.textContent = 'ASYMMETRIC MoE (16+1 TOP-2)';
                if (kvCut) kvCut.innerHTML = '434M ACTIVE <span class="unit">(1.13B STORED)</span>';
                if (moeStatus) moeStatus.innerHTML = '16 ROUTED <span class="num">(TOP-2)</span> + 1 SHARED';
                if (overlap) overlap.innerHTML = 'AUX-LOSS-FREE <span class="unit">(&Delta;b BIAS)</span>';
                if (formulaBar) {
                    formulaBar.innerHTML = '<span class="formula-sym">y<sub class="f-sub">FFN</sub></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-a">FFN<sub class="f-sub">shared</sub>(x)</span> ' +
                        '<span class="formula-op">+</span> ' +
                        '<span class="formula-term term-b">&Sigma;<sub class="f-sub">i&isin;Top2</sub> s<sub class="f-sub">i</sub> FFN<sub class="f-sub">i</sub>(x)</span>';
                }
            } else if (currentMode === 'mtp') {
                if (modeLabel) modeLabel.textContent = 'MTP · DEPTH-2 MULTI-TOKEN SPECULATION';
                if (kvCut) kvCut.innerHTML = '2.0&times; DRAFT <span class="unit">(DEPTH 2)</span>';
                if (moeStatus) moeStatus.innerHTML = 'HEADS [0.3, 0.1] <span class="num">WEIGHTED</span>';
                if (overlap) overlap.innerHTML = 'SHARED EMBED <span class="unit">(DUAL DRAFT)</span>';
                if (formulaBar) {
                    formulaBar.innerHTML = '<span class="formula-sym">L<sub class="f-sub">total</sub></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-a">L<sub class="f-sub">0</sub> (Main)</span> ' +
                        '<span class="formula-op">+</span> ' +
                        '<span class="formula-term term-b">0.3 &middot; L<sub class="f-sub">1</sub> (t+1) + 0.1 &middot; L<sub class="f-sub">2</sub> (t+2)</span>';
                }
            } else if (currentMode === 'hybrid') {
                if (modeLabel) modeLabel.textContent = '3:1 HYBRID STACK (24 GDN ⇄ 8 MLA)';
                if (kvCut) kvCut.innerHTML = '75% LINEAR <span class="unit">(24/32 SUB-QUADRATIC)</span>';
                if (moeStatus) moeStatus.innerHTML = '32 BLOCKS <span class="num">(ASYMMETRIC FFN)</span>';
                if (overlap) overlap.innerHTML = 'DUAL OPTIM <span class="unit">(NorMuon + AdamW)</span>';
                if (formulaBar) {
                    formulaBar.innerHTML = '<span class="formula-sym">Block<sub class="f-sub">l</sub></span> ' +
                        '<span class="formula-op">=</span> ' +
                        '<span class="formula-term term-a">GDN + SwiGLU (l mod 4 &ne; 3)</span> ' +
                        '<span class="formula-dot">&harr;</span> ' +
                        '<span class="formula-term term-b">MLA + MoE (l mod 4 = 3)</span>';
                }
            }
        }

        // Draw Radar Dial & Concentric Grid
        function drawDialAndGrid(t) {
            ctx.save();
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.35)';
            ctx.lineWidth = 1;

            // Cardinal Crosshairs
            ctx.beginPath();
            ctx.moveTo(cx - radius - 16, cy); ctx.lineTo(cx + radius + 16, cy);
            ctx.moveTo(cx, cy - radius - 16); ctx.lineTo(cx, cy + radius + 16);
            ctx.stroke();

            // 64-tick perimeter dial
            var numTicks = 64;
            for (var k = 0; k < numTicks; k++) {
                var angle = (k / numTicks) * Math.PI * 2;
                var isMajor = (k % 8 === 0);
                var isMedium = (k % 4 === 0);
                var tickLen = isMajor ? 9 : (isMedium ? 6 : 3);
                var r0 = radius;
                var r1 = radius + tickLen;

                var x0 = cx + Math.cos(angle) * r0;
                var y0 = cy + Math.sin(angle) * r0;
                var x1 = cx + Math.cos(angle) * r1;
                var y1 = cy + Math.sin(angle) * r1;

                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.strokeStyle = isMajor ? PALETTE.gold : (isMedium ? PALETTE.gridAxis : 'rgba(58, 50, 38, 0.45)');
                ctx.lineWidth = isMajor ? 1.5 : 1;
                ctx.stroke();
            }

            // Concentric boundary circles
            var rings = [
                { r: radius * 0.35, style: 'rgba(224, 122, 63, 0.35)', dash: [4, 4], label: 'O(1) STATE HORIZON' },
                { r: radius * 0.65, style: 'rgba(154, 148, 64, 0.35)', dash: [2, 6], label: 'RECURRENCE MANIFOLD' },
                { r: radius * 0.88, style: 'rgba(201, 163, 92, 0.28)', dash: [], label: '16-HEAD RECURRENCE ORBIT' }
            ];

            rings.forEach(function (ring) {
                ctx.beginPath();
                ctx.arc(cx, cy, ring.r, 0, Math.PI * 2);
                ctx.strokeStyle = ring.style;
                ctx.lineWidth = 1;
                ctx.setLineDash(ring.dash);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            // Axis labels
            ctx.font = 'bold 8px "JetBrains Mono", monospace';
            ctx.fillStyle = PALETTE.inkFaint;
            ctx.textAlign = 'center';
            ctx.fillText('+Q_c', cx + radius + 24, cy + 3);
            ctx.fillText('-Q_c', cx - radius - 24, cy + 3);
            ctx.fillText('+K_c', cx, cy - radius - 18);
            ctx.fillText('-K_c', cx, cy + radius + 22);

            ctx.restore();
        }

        // Mode 1: GDN Recurrence Dynamics
        function drawGDNManifold(t, probeNode) {
            ctx.save();
            var stateR = radius * 0.35;
            var orbitR = radius * 0.88;

            // 1. Central Recurrent State Core S_t
            var pulse = Math.sin(t * 2.4) * 0.08 + 1.0 + (flashEffect * 0.2);
            var coreGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, stateR * pulse);
            coreGrad.addColorStop(0, 'rgba(224, 122, 63, 0.45)');
            coreGrad.addColorStop(0.5, 'rgba(201, 163, 92, 0.20)');
            coreGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.beginPath();
            ctx.arc(cx, cy, stateR * pulse, 0, Math.PI * 2);
            ctx.fillStyle = coreGrad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, stateR, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.terracotta;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('GDN RECURRENT STATE S_t', cx, cy - 8);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('O(1) MEMORY · CHUNK-64 SCAN', cx, cy + 6);
            ctx.fillText('(1 - β q kᵀ) h + β v kᵀ', cx, cy + 18);

            // 2. 16 Rotating Head Phasors & State Update Rays
            for (var h = 0; h < CHANNELS_COUNT; h++) {
                var ch = channels[h];
                var baseAngle = (h / CHANNELS_COUNT) * Math.PI * 2;
                var angle = baseAngle + t * (0.18 + (h % 3) * 0.04);
                var hx = cx + Math.cos(angle) * orbitR;
                var hy = cy + Math.sin(angle) * orbitR;
                ch.x = hx; ch.y = hy;

                // State update beam from Head to Recurrent Core
                var beamAlpha = 0.30 + 0.45 * Math.sin(t * 3.0 + h);
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(cx, cy);
                ctx.strokeStyle = (h % 2 === 0) ?
                    'rgba(224, 122, 63, ' + Math.min(1, beamAlpha).toFixed(2) + ')' :
                    'rgba(154, 148, 64, ' + Math.min(1, beamAlpha).toFixed(2) + ')';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Head Node
                ctx.beginPath();
                ctx.arc(hx, hy, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = (h % 2 === 0) ? PALETTE.terracotta : PALETTE.olive;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Head Label
                ctx.fillStyle = PALETTE.ink;
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('H' + (h < 10 ? '0' + h : h), hx + Math.cos(angle) * 12, hy + Math.sin(angle) * 12 + 3);

                var dHead = Math.hypot(mouseX - hx, mouseY - hy);
                if (dHead < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.tag = 'PROBE · GDN HEAD #' + (h < 10 ? '0' + h : h);
                    probeNode.coords = 'State Decay α = ' + (0.85 + 0.14 * Math.sin(h)).toFixed(3) + ' · Chunk #04';
                    probeNode.desc = 'Recurrence Update Δh = (1 - β q kᵀ) h + β v kᵀ';
                }
            }

            // Inflowing Chunked Scan Particles
            particles.forEach(function (p) {
                if (!reduced) {
                    p.radiusFrac -= 0.12 * simSpeed * 0.016;
                    if (p.radiusFrac < 0.35) p.radiusFrac = 0.95;
                    p.theta += p.speed * 0.016 * simSpeed;
                }
                var pr = radius * p.radiusFrac;
                var px = cx + Math.cos(p.theta) * pr;
                var py = cy + Math.sin(p.theta) * pr;

                ctx.beginPath();
                ctx.arc(px, py, p.size, 0, Math.PI * 2);
                ctx.fillStyle = (p.radiusFrac > 0.65 ? PALETTE.oliveGlow : PALETTE.terracottaGlow);
                ctx.fill();
            });

            ctx.restore();
        }

        // Mode 2: MLA Latent Attention Manifold
        function drawMLAManifold(t, probeNode) {
            ctx.save();
            var latentR = radius * 0.38;
            var ropeR = radius * 0.65;
            var headR = radius * 0.88;

            // 1. Central 128d Latent KV Manifold Core
            var pulse = Math.sin(t * 2.4) * 0.08 + 1.0;
            var coreGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, latentR * pulse);
            coreGrad.addColorStop(0, 'rgba(154, 148, 64, 0.45)');
            coreGrad.addColorStop(0.5, 'rgba(201, 163, 92, 0.20)');
            coreGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.beginPath();
            ctx.arc(cx, cy, latentR * pulse, 0, Math.PI * 2);
            ctx.fillStyle = coreGrad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, latentR, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.olive;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('128d LATENT KV CORE', cx, cy - 8);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('16× KV REDUCTION / LAYER', cx, cy + 6);
            ctx.fillText('W_DKV · h_t', cx, cy + 18);

            // 2. 16 Query Heads & Decoupled RoPE Beams
            for (var q = 0; q < 16; q++) {
                var baseAngle = (q / 16) * Math.PI * 2;
                var angle = baseAngle + t * (0.18 + (q % 3) * 0.04);
                var qx = cx + Math.cos(angle) * headR;
                var qy = cy + Math.sin(angle) * headR;

                var kvGrp = Math.floor(q / 4);
                var kvAngle = (kvGrp / 4) * Math.PI * 2 + t * 0.08;
                var kvX = cx + Math.cos(kvAngle) * latentR;
                var kvY = cy + Math.sin(kvAngle) * latentR;

                // Attention Ray from Query to KV Group
                var beamAlpha = 0.30 + 0.45 * Math.sin(t * 3.0 + q);
                ctx.beginPath();
                ctx.moveTo(qx, qy);
                ctx.lineTo(kvX, kvY);
                ctx.strokeStyle = 'rgba(224, 122, 63, ' + Math.min(1, beamAlpha).toFixed(2) + ')';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Decoupled RoPE node along 25% orbit
                var ropeAngle = baseAngle - t * 0.35;
                var rx = cx + Math.cos(ropeAngle) * ropeR;
                var ry = cy + Math.sin(ropeAngle) * ropeR;
                ctx.beginPath();
                ctx.arc(rx, ry, 2.8, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.goldGlow;
                ctx.fill();

                // Head Node
                ctx.beginPath();
                ctx.arc(qx, qy, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = PALETTE.terracotta;
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = PALETTE.ink;
                ctx.font = 'bold 8px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText('Q' + (q < 10 ? '0' + q : q), qx + Math.cos(angle) * 12, qy + Math.sin(angle) * 12 + 3);

                var dQ = Math.hypot(mouseX - qx, mouseY - qy);
                if (dQ < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.tag = 'PROBE · QUERY HEAD #' + q + ' → KV GROUP #' + kvGrp;
                    probeNode.coords = 'Latent KV c_t [' + MODEL.kv_lora_rank + 'd] + RoPE [' + MODEL.qk_rope_head_dim + 'd] (25% head_dim)';
                    probeNode.desc = 'Compression: 4096 dims → 256 dims / layer (16× cut)';
                }
            }

            ctx.restore();
        }

        // Mode 3: Asymmetric MoE Field
        function drawMoERoutingField(t, probeNode) {
            ctx.save();
            var numExperts = 16;
            var expertR = radius * 0.90;
            var activeExperts = [];

            // Compute active top-2 experts deterministically based on time
            var stepCycle = Math.floor(t * 1.5);
            activeExperts.push((stepCycle * 3) % numExperts);
            activeExperts.push((stepCycle * 3 + 5) % numExperts);

            // Central Shared Expert
            var sharedR = radius * 0.26;
            var sharedGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, sharedR);
            sharedGrad.addColorStop(0, 'rgba(201, 163, 92, 0.45)');
            sharedGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.beginPath();
            ctx.arc(cx, cy, sharedR, 0, Math.PI * 2);
            ctx.fillStyle = sharedGrad;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, sharedR * 0.75, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('SHARED EXPERT', cx, cy - 4);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('ALWAYS ON · ' + MODEL.moe_inter_dim + 'd', cx, cy + 8);

            // Draw 16 expert stations
            for (var e = 0; e < numExperts; e++) {
                var angle = (e / numExperts) * Math.PI * 2 - Math.PI / 2;
                var ex = cx + Math.cos(angle) * expertR;
                var ey = cy + Math.sin(angle) * expertR;

                var isActive = (activeExperts.indexOf(e) !== -1);

                // Dispatch Conduit from Center
                if (isActive) {
                    var beamAlpha = 0.5 + 0.4 * Math.sin(t * 6.0 + e);
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(ex, ey);
                    ctx.strokeStyle = 'rgba(224, 122, 63, ' + Math.min(1, beamAlpha).toFixed(2) + ')';
                    ctx.lineWidth = 2.0;
                    ctx.stroke();

                    // Active pulse ring
                    ctx.beginPath();
                    ctx.arc(ex, ey, 10 + 4 * Math.sin(t * 8.0 + e), 0, Math.PI * 2);
                    ctx.strokeStyle = PALETTE.terracottaGlow;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }

                // Expert Station Node
                ctx.beginPath();
                ctx.arc(ex, ey, isActive ? 6.5 : 4.5, 0, Math.PI * 2);
                ctx.fillStyle = isActive ? PALETTE.terracotta : PALETTE.paperBg;
                ctx.fill();
                ctx.strokeStyle = isActive ? PALETTE.gold : PALETTE.gridAxis;
                ctx.lineWidth = isActive ? 2 : 1;
                ctx.stroke();

                // Dynamic Bias Bar Arc
                var biasVal = (Math.sin(e * 1.3 + t * 0.8) * 0.4 + 0.5);
                ctx.beginPath();
                ctx.arc(ex, ey, 8.5, angle - Math.PI * 0.4, angle - Math.PI * 0.4 + biasVal * Math.PI * 0.8);
                ctx.strokeStyle = isActive ? PALETTE.gold : 'rgba(154, 148, 64, 0.5)';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Label
                ctx.fillStyle = isActive ? '#ffffff' : PALETTE.inkFaint;
                ctx.font = (isActive ? 'bold 8.5px' : '7.5px') + ' "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                var labelR = expertR + (isActive ? 16 : 14);
                ctx.fillText((e < 10 ? '0' : '') + e, cx + Math.cos(angle) * labelR, cy + Math.sin(angle) * labelR + 3);

                var dExp = Math.hypot(mouseX - ex, mouseY - ey);
                if (dExp < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.tag = 'PROBE · ROUTED EXPERT #' + (e < 10 ? '0' + e : e);
                    probeNode.coords = 'Status: ' + (isActive ? 'ACTIVE (Top-2)' : 'IDLE') + ' · Bias Δb = ' + (Math.sin(e * 0.7) * 0.05).toFixed(4);
                    probeNode.desc = 'Dim ' + MODEL.moe_inter_dim + ' · SwiGLU FFN · Shared Expert Active [' + MODEL.moe_inter_dim + 'd]';
                }
            }

            ctx.restore();
        }

        // Mode 4: MTP Speculative Tree Field
        function drawMTPTreeField(t, probeNode) {
            ctx.save();
            var rootX = cx - radius * 0.45;
            var rootY = cy;
            var d1X = cx + radius * 0.15;
            var d1Y = cy - radius * 0.35;
            var d2X = cx + radius * 0.55;
            var d2Y = cy + radius * 0.35;

            // Speculative Bridges
            ctx.strokeStyle = PALETTE.olive;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(rootX, rootY);
            ctx.lineTo(d1X, d1Y);
            ctx.stroke();

            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.moveTo(d1X, d1Y);
            ctx.lineTo(d2X, d2Y);
            ctx.stroke();

            // Root Main Token
            ctx.beginPath();
            ctx.arc(rootX, rootY, 26, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(224, 122, 63, 0.25)';
            ctx.fill();
            ctx.strokeStyle = PALETTE.terracotta;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('MAIN MODEL', rootX, rootY - 4);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('Token x_t (wt=1.0)', rootX, rootY + 8);

            // Draft 1
            ctx.beginPath();
            ctx.arc(d1X, d1Y, 22, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(154, 148, 64, 0.25)';
            ctx.fill();
            ctx.strokeStyle = PALETTE.olive;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('MTP-1 (t+1)', d1X, d1Y - 4);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('Loss wt: 0.30', d1X, d1Y + 8);

            // Draft 2
            ctx.beginPath();
            ctx.arc(d2X, d2Y, 18, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(201, 163, 92, 0.25)';
            ctx.fill();
            ctx.strokeStyle = PALETTE.gold;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 8px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText('MTP-2 (t+2)', d2X, d2Y - 4);
            ctx.fillStyle = PALETTE.inkSoft;
            ctx.font = '7.5px "JetBrains Mono", monospace';
            ctx.fillText('Loss wt: 0.10', d2X, d2Y + 8);

            ctx.restore();
        }

        // Mode 5: 3:1 Hybrid Macro Stack Field
        function drawHybridStackField(t, probeNode) {
            ctx.save();
            var totalLayers = 32;
            var stackR = radius * 0.85;

            for (var l = 0; l < totalLayers; l++) {
                var angle = (l / totalLayers) * Math.PI * 2 - Math.PI / 2;
                var isMLA = (l % 4 === 3);
                var lx = cx + Math.cos(angle) * stackR;
                var ly = cy + Math.sin(angle) * stackR;

                // Connecting beam
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(lx, ly);
                ctx.strokeStyle = isMLA ? 'rgba(154, 148, 64, 0.35)' : 'rgba(224, 122, 63, 0.20)';
                ctx.lineWidth = isMLA ? 1.5 : 0.8;
                ctx.stroke();

                // Layer node
                ctx.beginPath();
                ctx.arc(lx, ly, isMLA ? 5.5 : 3.5, 0, Math.PI * 2);
                ctx.fillStyle = isMLA ? PALETTE.olive : PALETTE.terracotta;
                ctx.fill();
                ctx.strokeStyle = isMLA ? PALETTE.gold : PALETTE.paperBg;
                ctx.lineWidth = 1;
                ctx.stroke();

                var dL = Math.hypot(mouseX - lx, mouseY - ly);
                if (dL < 16 && !probeNode.found) {
                    probeNode.found = true;
                    probeNode.tag = 'PROBE · STACK LAYER #' + l + ' (' + (isMLA ? 'MLA + MoE' : 'GDN (recurrence-only, no FFN)') + ')';
                    probeNode.coords = isMLA ? 'Full Attention 4 KV Groups + 16+1 Experts' : 'Linear Recurrence 1D Selective Scan (no FFN)';
                    probeNode.desc = 'Stack Ratio: 3:1 (75% Sub-Quadratic)';
                }
            }

            ctx.restore();
        }

        // Render Frame
        function renderFrame(dt, force) {
            if (isPaused && !force) return;
            simTime += dt * simSpeed;

            // Clear Background
            var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.35);
            grad.addColorStop(0, PALETTE.paperCenter);
            grad.addColorStop(0.65, PALETTE.paperBg);
            grad.addColorStop(1, '#070605');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, width, height);

            drawDialAndGrid(simTime);

            var probeNode = { found: false, tag: '', coords: '', desc: '' };

            if (currentMode === 'gdn') drawGDNManifold(simTime, probeNode);
            else if (currentMode === 'mla') drawMLAManifold(simTime, probeNode);
            else if (currentMode === 'moe') drawMoERoutingField(simTime, probeNode);
            else if (currentMode === 'mtp') drawMTPTreeField(simTime, probeNode);
            else if (currentMode === 'hybrid') drawHybridStackField(simTime, probeNode);

            // Handle shockwaves
            for (var w = shockwaves.length - 1; w >= 0; w--) {
                var sw = shockwaves[w];
                sw.radius += 360 * dt;
                sw.life = Math.max(0, 1.0 - sw.radius / sw.maxRadius);
                if (sw.life <= 0) {
                    shockwaves.splice(w, 1);
                    continue;
                }
                ctx.beginPath();
                ctx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
                ctx.strokeStyle = sw.color;
                ctx.lineWidth = 1.8;
                ctx.globalAlpha = sw.life * 0.75;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            // Update Probe HUD
            if (probeEl) {
                if (probeNode.found) {
                    probeEl.style.opacity = '1';
                    var px = Math.min(Math.max(probeNode.x, 150), width - 150);
                    if (probeNode.y < 85) {
                        probeEl.style.transform = 'translate(-50%, 14px)';
                        probeEl.style.marginTop = '0px';
                    } else {
                        probeEl.style.transform = 'translate(-50%, -100%)';
                        probeEl.style.marginTop = '-10px';
                    }
                    probeEl.style.left = px + 'px';
                    probeEl.style.top = probeNode.y + 'px';
                    if (probeTag) probeTag.textContent = probeNode.tag;
                    if (probeCoords) probeCoords.textContent = probeNode.coords;
                    if (probeDecay) probeDecay.textContent = probeNode.desc;
                } else if (!isHovered) {
                    probeEl.style.opacity = '0';
                }
            }
        }

        function renderLoop(now) {
            var dt = Math.min((now - lastTime) / 1000, 0.1);
            lastTime = now;
            renderFrame(dt, false);
            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(renderLoop);
            }
        }

        updateFormulaAndLabels();
        lastTime = performance.now();
        if (!reduced) animId = requestAnimationFrame(renderLoop);
        else renderFrame(0, true);
    }


    // ------------------------------------------------------------------
    // 2. Full 32-Layer Training Step Pipeline Telemetry (FIG · A1)
    //    DeepSeek-v3-Lite & Mamba-3-Lite style synchronized multi-stage
    //    logic analyzer and systems execution graph.
    // ------------------------------------------------------------------
    function initPassDiagram() {
        var canvas = document.getElementById('passDiagramCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var tooltipEl = document.getElementById('passStageTooltip');
        var stTag = document.getElementById('stTag');
        var stOp = document.getElementById('stOp');
        var stShape = document.getElementById('stShape');
        var stDesc = document.getElementById('stDesc');
        var phaseEl = document.getElementById('phCurrentPhase');
        var tickerText = document.getElementById('passTickerText');
        var tickerBeacon = document.getElementById('passTickerBeacon');
        var pauseBtn = document.getElementById('passPauseBtn');
        var speedBtn = document.getElementById('passSpeedBtn');
        var phaseBtns = document.querySelectorAll('.pass-controls .pass-btn[data-phase]');

        var currentPhaseMode = 'cycle';
        var isPaused = false;
        var isHovered = false;
        var simSpeed = 1.0;
        var mouseX = -9999, mouseY = -9999;
        var hoveredStation = null;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            terracottaTint: 'rgba(224, 122, 63, 0.16)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            oliveTint: 'rgba(154, 148, 64, 0.16)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.85)',
            goldTint: 'rgba(201, 163, 92, 0.20)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var STAGES = [
            {
                id: 1,
                tag: 'STAGE 01 · EMBEDDING',
                badge: '01 · EMB',
                title: 'Embedding',
                sub: 'x_t → 896d',
                chip: 'Tied Weights',
                op: 'h_0 = Embedding(x_t) · Weight-Tied with LM Head',
                shape: 'Input: [B=4, 4096] uint32 → Output: [B, 4096, 896] bf16',
                desc: 'BPE-64k + 256-byte vocab (64,256) · FSDP-2 on-demand parameter prefetch',
                isParam: true
            },
            {
                id: 2,
                tag: 'STAGE 02 · GDN RECURRENCE',
                badge: '02 · GDN',
                title: 'Triton GDN',
                sub: 'Chunk-64 O(1)',
                chip: '75% Sub-Quad',
                op: 'S_t = α_t (I - β_t k_t k_tᵀ) S_{t-1} + β_t v_t k_tᵀ · y_t = (S_t q_t) ⊙ σ(z_t)',
                shape: 'State: [' + MODEL.gdn_n_heads + ', ' + MODEL.gdn_d_state + ', ' + MODEL.gdn_headdim + '] bf16 · Recur Output: [B, 4096, 1280]',
                desc: 'Fused Triton chunked Gated Delta Rule (chunk=64) · Constant O(1) state memory · No FFN/SwiGLU on GDN blocks',
                isParam: true
            },
            {
                id: 3,
                tag: 'STAGE 03 · MLA ATTENTION',
                badge: '03 · MLA',
                title: 'MLA Attention',
                sub: '4 KV Groups',
                chip: '16× KV Cut',
                op: 'c_t = W_DKV · h_t [' + MODEL.kv_lora_rank + 'd] · q\'_t = W_UQ(RMSNorm(W_DQ h_t))',
                shape: 'Latent KV: [B, 4096, ' + MODEL.kv_lora_rank + '] + RoPE [B, 4096, ' + MODEL.qk_rope_head_dim + '] bf16',
                desc: 'Multi-Head Latent Attention with MQA-4 & decoupled RoPE (16× per-layer KV cut: 4096 → 256 dims)',
                isParam: true
            },
            {
                id: 4,
                tag: 'STAGE 04 · ASYMMETRIC MoE',
                badge: '04 · MOE',
                title: 'Asymmetric MoE',
                sub: '16+1 Top-2',
                chip: 'Aux-Loss-Free',
                op: 's = sigmoid(W_g x) + b · y = ∑_{i∈Top2} s_i E_i(x) + E_shared(x)',
                shape: MODEL.n_routed_experts + ' Routed [' + MODEL.moe_inter_dim + 'd] + ' + MODEL.n_shared_experts + ' Shared [' + MODEL.moe_inter_dim + 'd] · FP32 Router Δb',
                desc: 'Auxiliary-loss-free load balancing on ' + MODEL.n_mla_layers + ' MLA blocks; GDN blocks have no FFN (recurrence-only)',
                isParam: true
            },
            {
                id: 5,
                tag: 'STAGE 05 · MTP HEADS & LOSS',
                badge: '05 · MTP/LOSS',
                title: 'MTP & Loss',
                sub: 'Softcap 15.0',
                chip: 'Depth-2 Draft',
                op: 'ℓ = CE(y_0) + 0.3·CE(y_{+1}) + 0.1·CE(y_{+2}) · Logit softcap 15.0',
                shape: 'Logits: [B, 4096, 64256] softcapped at 15.0 → Joint Loss ℓ',
                desc: 'Depth-2 Multi-Token Prediction speculative heads + softcapped cross-entropy loss',
                isParam: true
            },
            {
                id: 6,
                tag: 'STAGE 06 · DUAL OPT STEP',
                badge: '06 · DUAL OPT',
                title: 'Dual Optimizer',
                sub: 'NorMuon+AdamW',
                chip: 'FP32 Master',
                op: 'NorMuon(2D Attn/GDN matrices) ‖ Cautious AdamW(1D, Embed, MoE)',
                shape: 'NorMuon LR 0.02 (5 iter Newton-Schulz) · AdamW LR 3e-4 (WSD)',
                desc: 'Matrix whitening for 2D weights; Cautious AdamW for 1D, embeddings and MoE experts',
                isParam: true
            }
        ];

        var forwardParticles = [];
        for (var f = 0; f < 24; f++) {
            forwardParticles.push({
                xFrac: f / 24,
                speed: 0.18 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (f % 3) - 1
            });
        }

        var backwardParticles = [];
        for (var b = 0; b < 24; b++) {
            backwardParticles.push({
                xFrac: b / 24,
                speed: 0.20 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (b % 3) - 1
            });
        }

        var adamParticles = [];
        var width = 0, height = 0;
        var stations = [];

        function layoutStations() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 270;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            stations = [];
            var numStages = STAGES.length;
            var marginX = 16;
            var availW = width - (marginX * 2);
            var gap = Math.max(8, Math.min(16, (availW - (numStages * 82)) / (numStages - 1)));
            var stationW = Math.max(76, (availW - (gap * (numStages - 1))) / numStages);
            var stationH = Math.min(124, height * 0.52);
            var stationY = (height - stationH) / 2 + 2;

            for (var i = 0; i < numStages; i++) {
                var sx = marginX + i * (stationW + gap);
                stations.push({
                    stage: STAGES[i],
                    x: sx,
                    y: stationY,
                    w: stationW,
                    h: stationH,
                    cx: sx + stationW / 2,
                    cy: stationY + stationH / 2,
                    glowForward: 0,
                    glowBackward: 0,
                    glowAdam: 0
                });
            }
        }

        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                layoutStations();
                if (reduced || isPaused) render(0, true);
            }).observe(container);
        } else {
            window.addEventListener('resize', layoutStations);
        }
        layoutStations();

        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
            if (isPaused) render(0, true);
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            hoveredStation = null;
            if (tooltipEl) tooltipEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;

            stations.forEach(function (st) {
                if (clickX >= st.x && clickX <= st.x + st.w && clickY >= st.y && clickY <= st.y + st.h) {
                    st.glowForward = 1.0;
                    st.glowBackward = 1.0;
                    for (var k = 0; k < 12; k++) {
                        adamParticles.push({
                            x: st.cx,
                            y: st.cy,
                            vx: (Math.random() - 0.5) * 160,
                            vy: (Math.random() - 0.5) * 160,
                            life: 1.0,
                            color: PALETTE.gold
                        });
                    }
                }
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume pipeline' : 'Pause pipeline');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        if (speedBtn) {
            speedBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (simSpeed === 1.0) simSpeed = 2.0;
                else if (simSpeed === 2.0) simSpeed = 0.5;
                else simSpeed = 1.0;
                speedBtn.textContent = simSpeed + '×';
            });
        }

        phaseBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                phaseBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentPhaseMode = btn.getAttribute('data-phase') || 'cycle';
                if (phaseEl) {
                    if (currentPhaseMode === 'cycle') phaseEl.textContent = 'AUTO CYCLE (FWD → BWD → DUAL OPT)';
                    else if (currentPhaseMode === 'forward') phaseEl.textContent = 'FORWARD ACTIVATION PASS & FSDP-2 PREFETCH';
                    else if (currentPhaseMode === 'backward') phaseEl.textContent = 'AUTOGRAD BACKWARD GRADIENT FLOW & RECOMPUTE';
                    else if (currentPhaseMode === 'opt') phaseEl.textContent = 'DUAL OPTIMIZER STEP (NorMuon 2D + AdamW 1D)';
                }
                if (reduced || isPaused) render(0, true);
            });
        });

        var CYCLE_DURATION = 8.6;

        function getCycleState(t) {
            if (currentPhaseMode === 'forward') {
                return { phase: 'forward', progress: (t * 0.4) % 1.0, subText: 'FORWARD ACTIVATIONS STREAMING (32-LAYER 3:1 HYBRID STACK)' };
            }
            if (currentPhaseMode === 'backward') {
                return { phase: 'backward', progress: (t * 0.4) % 1.0, subText: 'AUTOGRAD GRADIENT PROPAGATION & CHECKPOINT RECOMPUTE' };
            }
            if (currentPhaseMode === 'opt') {
                return { phase: 'adam', progress: (t * 0.5) % 1.0, subText: 'DUAL OPTIMIZER · NorMuon (2D Whitening) ‖ Cautious AdamW (1D/MoE)' };
            }

            var cycleT = t % CYCLE_DURATION;
            if (cycleT < 3.2) {
                return { phase: 'forward', progress: cycleT / 3.2, subText: 'FORWARD · Tokens x_t → 24× GDN (Linear) ⇄ 8× MLA (MoE 16+1) → MTP Heads' };
            } else if (cycleT < 4.2) {
                return { phase: 'loss', progress: (cycleT - 3.2) / 1.0, subText: 'LOSS COMPUTATION · Softcapped Cross-Entropy + Depth-2 MTP Losses' };
            } else if (cycleT < 7.0) {
                return { phase: 'backward', progress: (cycleT - 4.2) / 2.8, subText: 'AUTOGRAD BACKWARD · Checkpointing Recomputes Activations (FA2 Pattern)' };
            } else if (cycleT < 8.0) {
                return { phase: 'adam', progress: (cycleT - 7.0) / 1.0, subText: 'DUAL OPT STEP · NorMuon (2D) ‖ Cautious AdamW (1D/MoE) & Master FP32 Sync' };
            } else {
                return { phase: 'rest', progress: (cycleT - 8.0) / 0.6, subText: 'STEP COMMITTED · Next Mini-Batch Ingest' };
            }
        }

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 8]);
            for (var x = 20; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            // Upper Forward Conduit
            var fwdY = height * 0.18;
            ctx.beginPath();
            ctx.moveTo(14, fwdY);
            ctx.lineTo(width - 14, fwdY);
            ctx.strokeStyle = 'rgba(224, 122, 63, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();

            // Lower Backward Conduit
            var bwdY = height * 0.82;
            ctx.beginPath();
            ctx.moveTo(14, bwdY);
            ctx.lineTo(width - 14, bwdY);
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FORWARD CONDUIT (ACTIVATIONS) →', 20, fwdY - 6);

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('← BACKWARD CONDUIT (GRADIENTS)', width - 20, bwdY + 14);
            ctx.restore();
        }

        function drawParticles(state, dt) {
            ctx.save();
            var fwdY = height * 0.18;
            var bwdY = height * 0.82;

            // Forward streaming particles
            if (state.phase === 'forward' || state.phase === 'loss' || currentPhaseMode === 'forward') {
                forwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt * simSpeed;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = 16 + p.xFrac * (width - 32);
                    var py = fwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.terracottaGlow;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px - 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(224, 122, 63, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // Backward streaming particles
            if (state.phase === 'backward' || currentPhaseMode === 'backward') {
                backwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt * simSpeed;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = width - 16 - p.xFrac * (width - 32);
                    var py = bwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.oliveGlow;
                    ctx.fill();

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // AdamW / NorMuon Burst Particles
            for (var k = adamParticles.length - 1; k >= 0; k--) {
                var ap = adamParticles[k];
                ap.x += ap.vx * dt * simSpeed;
                ap.y += ap.vy * dt * simSpeed;
                ap.life -= dt * 1.6 * simSpeed;

                if (ap.life <= 0) {
                    adamParticles.splice(k, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(ap.x, ap.y, 2.0 * ap.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(201, 163, 92, ' + ap.life.toFixed(2) + ')';
                ctx.fill();
            }

            ctx.restore();
        }

        function drawStations(state, dt) {
            ctx.save();
            var closest = null;
            var numStations = stations.length;

            stations.forEach(function (st, idx) {
                var frac = idx / (numStations - 1);

                var isFwdActive = false;
                var isBwdActive = false;
                var isAdamActive = (state.phase === 'adam');

                if (state.phase === 'forward') {
                    isFwdActive = state.progress >= (frac - 0.12) && state.progress <= (frac + 0.18);
                } else if (state.phase === 'loss') {
                    isFwdActive = (idx === numStations - 2);
                } else if (state.phase === 'backward') {
                    var bwdFrac = 1.0 - frac;
                    isBwdActive = state.progress >= (bwdFrac - 0.12) && state.progress <= (bwdFrac + 0.18);
                }

                if (isFwdActive) st.glowForward = 1.0;
                else st.glowForward = Math.max(0, st.glowForward - dt * 2.2);

                if (isBwdActive) st.glowBackward = 1.0;
                else st.glowBackward = Math.max(0, st.glowBackward - dt * 2.2);

                if (isAdamActive) st.glowAdam = 1.0;
                else st.glowAdam = Math.max(0, st.glowAdam - dt * 2.0);

                var isHover = isHovered && (mouseX >= st.x && mouseX <= st.x + st.w && mouseY >= st.y && mouseY <= st.y + st.h);
                if (isHover) closest = st;

                var cardBg = isHover ? PALETTE.paperStationHover : PALETTE.paperStation;
                var borderColor = PALETTE.ruleStrong;

                if (st.glowAdam > 0.1) {
                    borderColor = PALETTE.gold;
                    cardBg = 'rgba(201, 163, 92, ' + (0.15 * st.glowAdam).toFixed(2) + ')';
                } else if (st.glowForward > 0.1) {
                    borderColor = PALETTE.terracotta;
                    cardBg = 'rgba(224, 122, 63, ' + (0.14 * st.glowForward).toFixed(2) + ')';
                } else if (st.glowBackward > 0.1) {
                    borderColor = PALETTE.olive;
                    cardBg = 'rgba(154, 148, 64, ' + (0.14 * st.glowBackward).toFixed(2) + ')';
                }

                ctx.fillStyle = cardBg;
                ctx.strokeStyle = isHover ? PALETTE.gold : borderColor;
                ctx.lineWidth = (isHover || st.glowForward > 0.3 || st.glowBackward > 0.3) ? 1.5 : 1.0;

                ctx.beginPath();
                ctx.rect(st.x, st.y, st.w, st.h);
                ctx.fill();
                ctx.stroke();

                // Corner Technical Brackets
                var crLen = 4;
                ctx.strokeStyle = isHover ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(st.x, st.y + crLen); ctx.lineTo(st.x, st.y); ctx.lineTo(st.x + crLen, st.y);
                ctx.moveTo(st.x + st.w - crLen, st.y); ctx.lineTo(st.x + st.w, st.y); ctx.lineTo(st.x + st.w, st.y + crLen);
                ctx.moveTo(st.x, st.y + st.h - crLen); ctx.lineTo(st.x, st.y + st.h); ctx.lineTo(st.x + crLen, st.y + st.h);
                ctx.moveTo(st.x + st.w - crLen, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h - crLen);
                ctx.stroke();

                // Status Beacon LED
                var ledColor = PALETTE.inkFaint;
                if (st.glowAdam > 0.2) ledColor = PALETTE.gold;
                else if (st.glowForward > 0.2) ledColor = PALETTE.terracotta;
                else if (st.glowBackward > 0.2) ledColor = PALETTE.olive;

                ctx.beginPath();
                ctx.arc(st.x + 8, st.y + 10, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                var isCompact = st.w < 70;
                var displayBadge = isCompact ? '0' + st.stage.id : st.stage.badge;
                var displayTitle = isCompact ? (st.stage.id === 1 ? 'Embed' : st.stage.id === 2 ? 'GDN' : st.stage.id === 3 ? 'MLA' : st.stage.id === 4 ? 'MoE' : st.stage.id === 5 ? 'Loss' : 'DualOpt') : st.stage.title;

                // Stage Number Badge
                ctx.font = isCompact ? 'bold 7.5px "JetBrains Mono", monospace' : 'bold 8.5px "JetBrains Mono", monospace';
                ctx.fillStyle = (st.glowForward > 0.2 ? PALETTE.terracotta : (st.glowBackward > 0.2 ? PALETTE.olive : PALETTE.inkSoft));
                ctx.textAlign = 'left';
                ctx.fillText(displayBadge, st.x + (isCompact ? 13 : 14), st.y + 13);

                // Stage Title
                ctx.font = isCompact ? 'bold 9px "JetBrains Mono", monospace' : 'bold 10px "JetBrains Mono", monospace';
                ctx.fillStyle = isHover ? '#ffffff' : PALETTE.ink;
                ctx.textAlign = 'center';
                ctx.fillText(displayTitle, st.cx, st.y + st.h * 0.40);

                // Subtitle
                if (st.h > 80) {
                    ctx.font = isCompact ? '7.5px "JetBrains Mono", monospace' : '8.5px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.gold;
                    ctx.fillText(st.stage.sub, st.cx, st.y + st.h * 0.60);
                }

                // Chip Tag
                if (st.h > 100 && !isCompact) {
                    ctx.font = '8px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.inkSoft;
                    ctx.fillText(st.stage.chip, st.cx, st.y + st.h * 0.78);
                }

                // Re-compute Checkpoint Tag during Backward
                if (st.glowBackward > 0.3 && (idx >= 1 && idx <= 4)) {
                    ctx.fillStyle = PALETTE.olive;
                    ctx.font = 'bold 7px "JetBrains Mono", monospace';
                    ctx.fillText('RE-COMPUTE', st.cx, st.y + st.h - 6);
                }
            });

            // Update Tooltip
            if (tooltipEl) {
                if (closest) {
                    tooltipEl.style.opacity = '1';
                    var tx = Math.min(Math.max(closest.cx, 180), width - 180);
                    tooltipEl.style.left = tx + 'px';
                    tooltipEl.style.top = closest.y + 'px';
                    if (stTag) stTag.textContent = closest.stage.tag;
                    if (stOp) stOp.textContent = closest.stage.op;
                    if (stShape) stShape.textContent = closest.stage.shape;
                    if (stDesc) stDesc.textContent = closest.stage.desc;
                } else {
                    tooltipEl.style.opacity = '0';
                }
            }

            ctx.restore();
        }

        function render(dt, force) {
            if (isPaused && !force) return;
            simTime += dt * simSpeed;

            drawBackground();
            var state = getCycleState(simTime);

            if (phaseEl && currentPhaseMode === 'cycle') {
                phaseEl.textContent = state.phase.toUpperCase() + ' PASS (' + Math.round(state.progress * 100) + '%)';
            }
            if (tickerText) {
                tickerText.textContent = state.subText;
            }

            drawParticles(state, dt);
            drawStations(state, dt);
        }

        function loop(now) {
            var dt = Math.min((now - lastTime) / 1000, 0.1);
            lastTime = now;
            render(dt, false);
            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        lastTime = performance.now();
        if (!reduced) animId = requestAnimationFrame(loop);
        else render(0, true);
    }


    // ------------------------------------------------------------------
    // 3. Interactive Mechanism Benchmark Labs
    // ------------------------------------------------------------------
    function initMechanismLabs() {
        // Lab 1: GDN Recurrence & Chunkwise Scan
        var gdnChunkSlider = document.getElementById('gdnChunkSlider');
        var gdnChunkLabel = document.getElementById('gdnChunkLabel');
        var gdnVerifyBtn = document.getElementById('gdnVerifyBtn');
        var gdnMatchText = document.getElementById('gdnMatchText');
        var gdnPipeline = document.getElementById('gdnPipelineDisplay');

        function renderGdnChunk(q) {
            if (gdnChunkLabel) gdnChunkLabel.textContent = q + ' tokens / chunk';
            if (gdnPipeline) {
                var stages = gdnPipeline.querySelectorAll('.chunk-stage');
                if (stages.length >= 2) {
                    var spans = stages[1].getElementsByTagName('span');
                    if (spans.length >= 2) {
                        spans[1].textContent = 'Fused 1D selective scan with chunked recurrence (Q=' + q + ')';
                    }
                }
            }
        }

        if (gdnChunkSlider) {
            gdnChunkSlider.addEventListener('input', function () {
                renderGdnChunk(parseInt(this.value, 10));
            });
            renderGdnChunk(parseInt(gdnChunkSlider.value, 10));
        }

        if (gdnVerifyBtn) {
            gdnVerifyBtn.addEventListener('click', function () {
                var original = gdnVerifyBtn.dataset.label || gdnVerifyBtn.textContent;
                gdnVerifyBtn.dataset.label = original;
                gdnVerifyBtn.textContent = '▸ SCANNING CHUNK…';
                gdnVerifyBtn.disabled = true;
                if (heroController.setMode) heroController.setMode('gdn');
                if (heroController.triggerPulse) heroController.triggerPulse();

                var sweep = ['< 5.0e-4 max |Δ|', '< 1.2e-5 max |Δ|', '< 3.7e-7 max |Δ|'];
                sweep.forEach(function (txt, i) {
                    setTimeout(function () {
                        if (gdnMatchText) {
                            gdnMatchText.textContent = txt;
                            gdnMatchText.style.color = '#9a9440';
                        }
                    }, 200 + i * 220);
                });

                setTimeout(function () {
                    if (gdnMatchText) {
                        gdnMatchText.textContent = 'VERIFIED (< 1e-6 max |Δ| match)';
                        gdnMatchText.style.color = '#9a9440';
                    }
                    gdnVerifyBtn.textContent = original;
                    gdnVerifyBtn.disabled = false;
                }, 900);
            });
        }

        // Lab 2: Asymmetric MoE Router
        var moeGrid = document.getElementById('moeMiniGrid');
        var moeRouteBtn = document.getElementById('moeRouteBatchBtn');
        var moeActiveList = document.getElementById('moeActiveList');

        if (moeGrid) {
            moeGrid.innerHTML = '';
            for (var i = 0; i < 16; i++) {
                var cell = document.createElement('div');
                cell.className = 'moe-mini-cell' + (i === 2 || i === 7 ? ' active' : '');
                cell.id = 'moeCell_' + i;
                cell.textContent = '#' + (i < 10 ? '0' + i : i);
                moeGrid.appendChild(cell);
            }
            var shCell = document.createElement('div');
            shCell.className = 'moe-mini-cell shared';
            shCell.textContent = 'SHARED EXPERT (ALWAYS ON · ' + MODEL.moe_inter_dim + 'd)';
            moeGrid.appendChild(shCell);
        }

        if (moeRouteBtn) {
            moeRouteBtn.addEventListener('click', function () {
                var original = moeRouteBtn.dataset.label || moeRouteBtn.textContent;
                moeRouteBtn.dataset.label = original;
                moeRouteBtn.textContent = '\u25b8 ROUTING TOP-2 EXPERTS\u2026';
                moeRouteBtn.disabled = true;

                var e1 = Math.floor(Math.random() * 16);
                var e2 = Math.floor(Math.random() * 16);
                while (e2 === e1) e2 = Math.floor(Math.random() * 16);

                for (var j = 0; j < 16; j++) {
                    var c = document.getElementById('moeCell_' + j);
                    if (c) c.className = 'moe-mini-cell' + (j === e1 || j === e2 ? ' active' : '');
                }

                if (moeActiveList) {
                    var str1 = '#' + (e1 < 10 ? '0' + e1 : e1);
                    var str2 = '#' + (e2 < 10 ? '0' + e2 : e2);
                    moeActiveList.textContent = str1 + ', ' + str2 + ' + shared (' + MODEL.moe_inter_dim + 'd)';
                }
                if (heroController.setMode) heroController.setMode('moe');
                if (heroController.triggerPulse) heroController.triggerPulse();

                setTimeout(function () {
                    moeRouteBtn.textContent = original;
                    moeRouteBtn.disabled = false;
                }, 600);
            });
        }

        // Lab 3: MLA KV Cache Footprint Analyzer
        var mlaSlider = document.getElementById('mlaContextSlider');
        var mlaLabel = document.getElementById('mlaContextLabel');
        var statMha = document.getElementById('statMhaVram');
        var statMla = document.getElementById('statMlaVram');
        var statSaved = document.getElementById('statMlaSaved');
        var mlaAbsorbBtn = document.getElementById('mlaAbsorbAnimBtn');

        if (mlaSlider) {
            mlaSlider.addEventListener('input', function () {
                var ctxLen = parseInt(this.value, 10);
                if (mlaLabel) mlaLabel.textContent = ctxLen.toLocaleString() + ' tokens';

                // Per-layer KV bytes/token (see docs/concepts/model-architecture.md §4.2.1):
                //   MHA : n_heads * 2 * head_dim * 2 bytes  (K+V)
                //   MLA : (kv_lora_rank + n_kv_groups * qk_rope_head_dim) * 2 bytes
                var mhaPerLayer = MODEL.n_heads * 2 * MODEL.head_dim * 2;
                var mlaPerLayer = (MODEL.kv_lora_rank + MODEL.n_kv_groups * MODEL.qk_rope_head_dim) * 2;
                var mhaBytes = ctxLen * mhaPerLayer * MODEL.n_layers;
                var mlaBytes = ctxLen * mlaPerLayer * MODEL.n_mla_layers;
                // GDN layers: O(1) state per sequence, not per token.
                // State shape: [B, n_heads_gdn, d_state, headdim] bf16 -> derive the bytes once.
                var gdnStateBytes = MODEL.micro_batch_size * MODEL.gdn_n_heads * MODEL.gdn_d_state * MODEL.gdn_headdim * 2;
                var gdnStateGB = gdnStateBytes / (1024 * 1024 * 1024);
                var mlaGB = (mlaBytes / (1024 * 1024 * 1024) + gdnStateGB).toFixed(2);
                var mhaGB = (mhaBytes / (1024 * 1024 * 1024)).toFixed(2);
                var savedGB = (mhaBytes / (1024 * 1024 * 1024) - mlaBytes / (1024 * 1024 * 1024)).toFixed(2);
                var perLayerRatio = (mhaPerLayer / mlaPerLayer).toFixed(2);

                if (statMha) statMha.textContent = mhaGB + ' GB';
                if (statMla) statMla.textContent = mlaGB + ' GB';
                if (statSaved) statSaved.textContent = savedGB + ' GB (' + perLayerRatio + '× cut per layer)';
            });
        }

        if (mlaAbsorbBtn) {
            mlaAbsorbBtn.addEventListener('click', function () {
                var original = mlaAbsorbBtn.dataset.label || mlaAbsorbBtn.textContent;
                mlaAbsorbBtn.dataset.label = original;
                mlaAbsorbBtn.textContent = '▸ OBSERVING MQA-4 PROJECTION…';
                mlaAbsorbBtn.disabled = true;
                if (heroController.setMode) heroController.setMode('mla');
                if (heroController.triggerPulse) heroController.triggerPulse();
                setTimeout(function () {
                    mlaAbsorbBtn.textContent = original;
                    mlaAbsorbBtn.disabled = false;
                }, 900);
            });
        }

        // Lab 4: MTP Speculative Tree Verification
        var mtpVerifyBtn = document.getElementById('mtpVerifyStepBtn');
        var mtpStatusText = document.getElementById('mtpStatusText');
        var mtpSpeedText = document.getElementById('mtpSpeedText');
        var mtpStepCount = 0;

        if (mtpVerifyBtn) {
            mtpVerifyBtn.addEventListener('click', function () {
                var original = mtpVerifyBtn.dataset.label || mtpVerifyBtn.textContent;
                mtpVerifyBtn.dataset.label = original;
                mtpVerifyBtn.textContent = '\u25b8 SPECULATING DEPTH-2\u2026';
                mtpVerifyBtn.disabled = true;
                mtpStepCount++;

                var p1 = (0.68 + Math.random() * 0.26).toFixed(2);
                var p2 = (0.66 + Math.random() * 0.22).toFixed(2);
                var accepted = (parseFloat(p1) >= 0.80 && parseFloat(p2) >= 0.75);

                var treeBox = document.getElementById('mtpTreeDisplay');
                if (treeBox) {
                    var candidates = [
                        ['"linear"', '"recurrence"', '"scaling"'],
                        ['"attention"', '"compressed"', '"manifold"'],
                        ['"optimal"', '"convergence"', '"dynamics"'],
                        ['"sub-quadratic"', '"efficiency"', '"speedup"']
                    ];
                    var set = candidates[mtpStepCount % candidates.length];
                    treeBox.innerHTML =
                        '<div class="mtp-branch"><span class="mtp-badge draft">MAIN HEAD</span><span>Token t &rarr; ' + set[0] + ' (p = 0.99)</span></div>' +
                        '<div class="mtp-branch"><span class="mtp-badge ' + (parseFloat(p1) >= 0.80 ? 'acc' : 'draft') + '">MTP DRAFT 1</span><span>Token t+1 &rarr; ' + set[1] + ' (p = ' + p1 + (parseFloat(p1) >= 0.80 ? ' &ge; 0.80' : ' &lt; 0.80') + ')</span></div>' +
                        '<div class="mtp-branch"><span class="mtp-badge ' + (parseFloat(p2) >= 0.75 ? 'acc' : 'draft') + '">MTP DRAFT 2</span><span>Token t+2 &rarr; ' + set[2] + ' (p = ' + p2 + (parseFloat(p2) >= 0.75 ? ' &ge; 0.75' : ' &lt; 0.75') + ')</span></div>';
                }

                if (mtpStatusText) {
                    mtpStatusText.textContent = accepted ? 'ACCEPTED (3 Tokens / Step)' : 'PARTIAL (2 Tokens / Step)';
                    mtpStatusText.style.color = accepted ? '#9a9440' : '#e07a3f';
                }
                if (mtpSpeedText) {
                    mtpSpeedText.textContent = accepted ? '2.00× effective speedup' : '1.50× effective speedup';
                }
                if (heroController.setMode) heroController.setMode('mtp');
                if (heroController.triggerPulse) heroController.triggerPulse();

                setTimeout(function () {
                    mtpVerifyBtn.textContent = original;
                    mtpVerifyBtn.disabled = false;
                }, 600);
            });
        }
    }


    // ------------------------------------------------------------------
    // 4. Code Block Expand/Collapse & Copy
    // ------------------------------------------------------------------
    window.toggleCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var isCollapsed = wrapper.classList.toggle('collapsed');
        var nLines = wrapper.getAttribute('data-lines') || '';
        btn.textContent = isCollapsed ? ('expand ▾ · ' + nLines + ' lines') : 'collapse ▴';
    };

    window.copyCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var code = wrapper.querySelector('pre code');
        if (!code) return;

        navigator.clipboard.writeText(code.innerText).then(function () {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.style.color = '#e07a3f';
            setTimeout(function () {
                btn.textContent = orig;
                btn.style.color = '';
            }, 1800);
        });
    };


    // ------------------------------------------------------------------
    // 5. Sidebar Filter & Mobile Toggle
    // ------------------------------------------------------------------
    window.filterNav = function () {
        var q = (document.getElementById('navSearch').value || '').toLowerCase();
        var navItems = document.querySelectorAll('.nav-item');
        var navGroups = document.querySelectorAll('.nav-group');

        navItems.forEach(function (item) {
            var text = item.textContent.toLowerCase();
            item.style.display = text.indexOf(q) !== -1 ? '' : 'none';
        });

        navGroups.forEach(function (group) {
            var visibleCount = group.querySelectorAll('.nav-item:not([style*="display: none"])').length;
            group.style.display = visibleCount > 0 ? '' : 'none';
        });
    };

    window.toggleSidebar = function () {
        var sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.toggle('open');
    };


    // ------------------------------------------------------------------
    // 6. Table of Contents Scrollspy
    // ------------------------------------------------------------------
    function initScrollspy() {
        var headings = document.querySelectorAll('.markdown-body h2, .markdown-body h3');
        var tocLinks = document.querySelectorAll('.toc-link');
        if (!headings.length || !tocLinks.length) return;

        function onScroll() {
            var scrollPos = window.scrollY + 120;
            var currentId = '';

            headings.forEach(function (h) {
                if (h.offsetTop <= scrollPos) {
                    currentId = h.id;
                }
            });

            tocLinks.forEach(function (link) {
                var href = link.getAttribute('href') || '';
                if (href === '#' + currentId) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }


    // ------------------------------------------------------------------
    // Initialize everything on DOMContentLoaded
    // ------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function () {
        initHeroCanvas();
        initPassDiagram();
        initMechanismLabs();
        initScrollspy();
    });

})();
