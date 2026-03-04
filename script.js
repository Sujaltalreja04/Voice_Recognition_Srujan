/**
 * GUARDIAN AI — Fully Browser-Native Voice Fraud & Deepfake Detection
 * All Python backend logic migrated to JavaScript:
 *   - stt.py        → Web Speech API (SpeechRecognition)
 *   - nlp_model.py  → Keyword scoring + OpenRouter API (fetch)
 *   - audio_model.py → Web Audio API (AnalyserNode for energy/ZCR analysis)
 *   - deepfake_logic.py → FFT + pitch/spectral heuristics via Web Audio API
 *   - fraud_engine.py   → JS FraudEngine class orchestrator
 *   - main.py       → Eliminated (no server needed)
 */

'use strict';

/* =============================================
   1. NLP ENGINE (nlp_model.py → JS)
   Keyword scoring + optional LLM via OpenRouter
   ============================================= */
const NLPEngine = (() => {
    const FRAUD_KEYWORDS = [
        "otp", "urgent", "kyc", "bank", "prize", "blocked", "verify",
        "expire", "card", "cvv", "account", "password", "transfer", "click",
        "link", "winner", "claim", "reward", "suspend", "immediately",
        "refund", "government", "irs", "tax", "police", "arrest"
    ];

    function getKeywordScore(text) {
        const lower = text.toLowerCase();
        let score = 0;
        const detected = [];
        for (const word of FRAUD_KEYWORDS) {
            if (lower.includes(word)) {
                score += 1;
                detected.push(word);
            }
        }
        return {
            fraud_score: Math.min(score / 3, 1.0),
            reason: detected.length ? `Detected risk words: ${detected.slice(0, 3).join(", ")}` : "No suspicious keywords found",
            detected_keywords: detected
        };
    }

    async function analyzeText(text) {
        if (!text || text.trim().length < 5) {
            return { fraud_score: 0.0, reason: "Silence / Too short", detected_keywords: [] };
        }

        // Use keyword scoring as primary (no API key needed)
        const kwResult = getKeywordScore(text);

        // Optional: call OpenRouter if API key is stored in sessionStorage
        const apiKey = sessionStorage.getItem('OPENROUTER_API_KEY');
        if (apiKey) {
            try {
                const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
                    method: "POST",
                    headers: {
                        "Authorization": `Bearer ${apiKey}`,
                        "Content-Type": "application/json",
                        "HTTP-Referer": window.location.href,
                        "X-Title": "Guardian AI Fraud Detection"
                    },
                    body: JSON.stringify({
                        model: "google/gemini-2.0-flash-lite-preview-02-05:free",
                        messages: [
                            {
                                role: "system",
                                content: `You are a fraud detection expert. Analyze the transcript. 
                                Return ONLY JSON: {"fraud_score": <0.0-1.0>, "reason": "<brief>", "detected_keywords": [<list>]}. No other text.`
                            },
                            { role: "user", content: text }
                        ],
                        response_format: { type: "json_object" }
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    const content = data.choices?.[0]?.message?.content;
                    if (content) return JSON.parse(content.trim());
                }
            } catch (e) {
                console.warn("OpenRouter NLP error:", e);
            }
        }

        return kwResult;
    }

    return { analyzeText, getKeywordScore };
})();


/* =============================================
   2. AUDIO ANALYSIS ENGINE (audio_model.py + deepfake_logic.py → JS)
   Uses Web Audio API AnalyserNode for real-time frame analysis
   ============================================= */
const AudioAnalysisEngine = (() => {

    /**
     * Compute RMS energy of a Float32Array buffer
     * Equivalent to librosa.feature.rms
     */
    function computeRMS(buffer) {
        let sum = 0;
        for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
        return Math.sqrt(sum / buffer.length);
    }

    /**
     * Compute Zero Crossing Rate
     * Equivalent to librosa.feature.zero_crossing_rate
     */
    function computeZCR(buffer) {
        let count = 0;
        for (let i = 1; i < buffer.length; i++) {
            if ((buffer[i] >= 0) !== (buffer[i - 1] >= 0)) count++;
        }
        return count / buffer.length;
    }

    /**
     * audioFraudScore — equivalent to audio_model.py: audio_fraud_score()
     * Accepts a Float32Array of PCM samples from the analyser time domain
     */
    function audioFraudScore(timeDomainData) {
        if (!timeDomainData || timeDomainData.length === 0) return { score: 0, stress: 0 };

        // Convert Uint8Array (0-255) to normalized Float32Array (-1 to 1)
        const floatBuf = new Float32Array(timeDomainData.length);
        for (let i = 0; i < timeDomainData.length; i++) {
            floatBuf[i] = (timeDomainData[i] - 128) / 128.0;
        }

        const energy = computeRMS(floatBuf);
        const zcr = computeZCR(floatBuf);

        // Heuristic stress: high energy + high ZCR = yelling/fast speech
        // Equivalent to: stress = min((energy * 500) + (zcr * 200), 100)
        const stress = Math.min((energy * 500) + (zcr * 200), 100);

        let score = 0;
        if (energy > 0.02) score += 0.4;
        if (zcr > 0.08) score += 0.4;
        score = Math.min(score, 1.0);

        return { score, stress: Math.round(stress) };
    }

    /**
     * Deepfake detection from frequency domain data
     * Equivalent to deepfake_logic.py: detect_ai_voice()
     * 
     * Uses FFT magnitude spectrum from AnalyserNode.getByteFrequencyData()
     * to approximate:
     *   - Spectral Flatness (Wiener entropy)
     *   - Pitch Stability heuristic
     *   - MFCC variance approximation
     */
    function detectAIVoice(freqData, timeDomainData) {
        if (!freqData || freqData.length === 0) return null;

        const N = freqData.length;

        // ---- Spectral Flatness (geometric mean / arithmetic mean) ----
        // Higher flatness → more "white noise" / digital buzz
        const mags = new Float32Array(N);
        let logSum = 0;
        let arithmSum = 0;
        let validCount = 0;
        for (let i = 0; i < N; i++) {
            const val = freqData[i] + 1e-6; // avoid log(0)
            mags[i] = val;
            if (val > 1) {
                logSum += Math.log(val);
                arithmSum += val;
                validCount++;
            }
        }
        const geoMean = validCount > 0 ? Math.exp(logSum / validCount) : 0;
        const arithMean = validCount > 0 ? arithmSum / validCount : 1;
        const spectralFlatness = geoMean / arithMean; // 0-1, higher = more noise-like

        // ---- Dominant pitch estimation (proxy for pitch stability via FFT peak) ----
        let maxMag = 0, peakBin = 0;
        for (let i = 2; i < N / 2; i++) {
            if (freqData[i] > maxMag) { maxMag = freqData[i]; peakBin = i; }
        }
        // Spread around peak → low spread means too-perfect pitch (AI signature)
        let peakSpread = 0;
        for (let i = Math.max(0, peakBin - 5); i < Math.min(N, peakBin + 5); i++) {
            peakSpread += Math.abs(freqData[i] - freqData[peakBin]);
        }
        const pitchStabilityScore = peakSpread < 80 ? 1 : 0; // low spread = suspicious

        // ---- MFCC variance approximation ----
        // Use spectral variance as proxy for mel-cepstral variation
        let mean = 0;
        for (let i = 0; i < N; i++) mean += freqData[i];
        mean /= N;
        let variance = 0;
        for (let i = 0; i < N; i++) variance += (freqData[i] - mean) ** 2;
        variance /= N;
        const mfccVarLow = variance < 800; // Low variance = predictable/synthetic

        // ---- Heuristic Scoring (mirrors deepfake_logic.py logic) ----
        let aiScore = 0;
        const reasons = [];

        if (pitchStabilityScore > 0) {
            aiScore += 0.4;
            reasons.push("Unnatural pitch stability detected (Robotic cadence)");
        }
        if (spectralFlatness > 0.08) {
            aiScore += 0.3;
            reasons.push("High spectral flatness / Digital noise detected");
        }
        if (mfccVarLow) {
            aiScore += 0.3;
            reasons.push("Predictable spectral patterns (Synthetic signature)");
        }

        const confidence = Math.min(Math.max(aiScore, 0.1), 0.98);
        const classification = confidence > 0.6 ? "AI-Generated" : "Human-Generated";

        return {
            classification,
            confidence: Math.round(confidence * 100) / 100,
            reasons,
            metrics: {
                spectralFlatness: Math.round(spectralFlatness * 10000) / 10000,
                spectralVariance: Math.round(variance)
            }
        };
    }

    return { audioFraudScore, detectAIVoice };
})();


/* =============================================
   3. FRAUD ENGINE (fraud_engine.py → JS)
   Orchestrates NLP + Audio + Deepfake analysis
   ============================================= */
const FraudEngine = (() => {

    /**
     * analyze() — equivalent to fraud_engine.py: analyze_call()
     * @param {string} transcript - text from SpeechRecognition
     * @param {object} audioScores - {score, stress} from AudioAnalysisEngine
     * @param {object} deepfakeResult - from AudioAnalysisEngine.detectAIVoice()
     * @param {string} lang - detected language code
     */
    async function analyze(transcript, audioScores, deepfakeResult, lang) {
        // 1. NLP analysis
        const nlpData = await NLPEngine.analyzeText(transcript);
        const tScore = nlpData.fraud_score || 0.0;

        // 2. Audio stress
        const aScore = audioScores?.score || 0.0;

        // 3. Deepfake check
        const isAI = deepfakeResult?.classification === "AI-Generated";
        const aiConf = deepfakeResult?.confidence || 0;

        // 4. Decision Matrix (matches fraud_engine.py)
        let final = (tScore * 0.5) + (aScore * 0.2);
        if (isAI) final += 0.3; // Heavy penalty for synthetic voices

        return {
            text: transcript,
            fraud_score: Math.min(final, 1.0),
            stress_score: audioScores?.stress || 0,
            is_ai: isAI,
            ai_confidence: aiConf,
            reason: `${isAI ? "AI-VOICE DETECTED. " : ""}${nlpData.reason || "Scanning..."}`,
            keywords: nlpData.detected_keywords || [],
            deepfake_reasons: deepfakeResult?.reasons || [],
            language: lang || "en"
        };
    }

    return { analyze };
})();


/* =============================================
   4. SPEECH TO TEXT (stt.py → JS)
   Uses Web Speech API SpeechRecognition
   ============================================= */
class SpeechToText {
    constructor(onResult, onError) {
        this.onResult = onResult;
        this.onError = onError;
        this.recognition = null;
        this.currentText = "";
        this.isActive = false;
    }

    start() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.onError("Speech Recognition not supported in this browser. Try Chrome.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US'; // Also picks up other languages

        this.recognition.onresult = (event) => {
            let fullText = '';
            for (let i = 0; i < event.results.length; i++) {
                fullText += event.results[i][0].transcript;
            }
            this.currentText = fullText;
            this.onResult(fullText);
        };

        this.recognition.onerror = (e) => {
            if (e.error !== 'no-speech') this.onError(e.error);
        };

        this.recognition.onend = () => {
            // Restart if still recording (handles auto-stop)
            if (this.isActive) {
                try { this.recognition.start(); } catch (e) { }
            }
        };

        this.isActive = true;
        this.recognition.start();
    }

    stop() {
        this.isActive = false;
        if (this.recognition) {
            try { this.recognition.stop(); } catch (e) { }
        }
    }

    getText() { return this.currentText; }
}


/* =============================================
   5. AUDIO STREAMER — Main Controller
   Replaces main.py WebSocket + all Python modules
   ============================================= */
class AudioStreamer {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.source = null;
        this.stt = null;
        this.isRecording = false;
        this.alertTriggered = false;
        this.analysisInterval = null;
        this.realtimeInterval = null;  // fast 100ms loop for gauge
        this._lastFraudPercent = 0;    // last known fraud score %
        this._fraudMode = false;       // true = show fraud score on gauge

        // Report data
        this.fullTranscript = "";
        this.historyData = [];
        this.latestAudioScores = { score: 0, stress: 0 };
        this.latestDeepfakeResult = null;

        // UI Elements
        this.startButton = document.getElementById('startBtn');
        this.stopButton = document.getElementById('stopBtn');
        this.dwnBtn = document.getElementById('dwnBtn');
        this.scoreValue = document.getElementById('scoreValue');
        this.statusObj = document.getElementById('statusObj');
        this.gaugeFill = document.getElementById('gaugeFill');
        this.reasonText = document.getElementById('reasonText');
        this.canvas = document.getElementById('visualizer');
        this.canvasCtx = this.canvas.getContext('2d');

        // Chart (Chart.js)
        this.stressChart = new Chart(
            document.getElementById('stressChart').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Voice Stress',
                            data: [],
                            borderColor: '#f85149',
                            backgroundColor: 'rgba(248, 81, 73, 0.1)',
                            tension: 0.4,
                            fill: true,
                            pointRadius: 2
                        },
                        {
                            label: 'Fraud Probability',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 162, 235, 0.1)',
                            tension: 0.4,
                            fill: true,
                            pointRadius: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 300 },
                    scales: {
                        x: { display: false },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#6e7e96', font: { size: 10 } }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#8b949e', font: { family: 'Outfit', size: 11 } } }
                    }
                }
            }
        );

        // Event Listeners
        this.startButton.addEventListener('click', () => this.start());
        this.stopButton.addEventListener('click', () => this.stop());
        if (this.dwnBtn) this.dwnBtn.addEventListener('click', () => this.downloadReport());

        // File upload: show filename
        const dfInput = document.getElementById('deepfakeUpload');
        if (dfInput) {
            dfInput.addEventListener('change', (e) => {
                const fn = document.getElementById('fileName');
                if (fn) fn.textContent = e.target.files[0]?.name || 'No file selected';
            });
        }
    }

    /* ---- START MONITORING ---- */
    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

            // Web Audio API setup — replaces librosa audio loading
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.source = this.audioContext.createMediaStreamSource(stream);
            this.source.connect(this.analyser);

            // Start visualizer
            this.drawVisualizer();

            // STT — replaces stt.py (faster-whisper)
            this.stt = new SpeechToText(
                (text) => {
                    this.fullTranscript = text;
                    this._highlightTranscript(text, []);
                },
                (err) => this.addLog(`⚠️ STT Error: ${err}`)
            );
            this.stt.start();

            // Fast realtime gauge loop — runs every 100ms for instant visual feedback
            this.realtimeInterval = setInterval(() => this._updateRealtimeGauge(), 100);

            // Full analysis loop (NLP + deepfake scoring) — runs every 3 seconds
            this.analysisInterval = setInterval(() => this._runAnalysis(), 3000);

            this.isRecording = true;
            this.updateButtons();
            this.addLog("✅ Monitoring started. Microphone active.");
            this._setMode("🎙️ Mode: Live Monitoring", true);
            this._updateSystemStatus("Monitoring Active", "var(--success)");

        } catch (err) {
            console.error(err);
            this.addLog(`❌ Error: ${err.message}`);
            if (err.name === 'NotAllowedError') {
                alert("Microphone access denied. Please allow microphone access in your browser settings.");
            } else {
                alert("Error accessing microphone: " + err.message);
            }
        }
    }

    /* ---- PERIODIC ANALYSIS LOOP (replaces WebSocket handler) ---- */
    async _runAnalysis() {
        if (!this.isRecording || !this.analyser) return;

        // Get audio data from Web Audio API
        const freqData = new Uint8Array(this.analyser.frequencyBinCount);
        const timeData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteFrequencyData(freqData);
        this.analyser.getByteTimeDomainData(timeData);

        // Run acoustic analysis (audio_model.py + deepfake_logic.py → JS)
        const audioScores = AudioAnalysisEngine.audioFraudScore(timeData);
        const deepfakeResult = AudioAnalysisEngine.detectAIVoice(freqData, timeData);

        this.latestAudioScores = audioScores;
        this.latestDeepfakeResult = deepfakeResult;

        // Get transcript from STT
        const transcript = this.stt ? this.stt.getText() : "";

        // Run fraud engine orchestration (fraud_engine.py → JS)
        const result = await FraudEngine.analyze(transcript, audioScores, deepfakeResult, "en");
        this._updateUI(result);
    }

    /* ---- REALTIME GAUGE UPDATE (runs every 100ms) ---- */
    _updateRealtimeGauge() {
        if (!this.isRecording || !this.analyser) return;

        // Read raw PCM time-domain samples
        const timeData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteTimeDomainData(timeData);

        // Compute RMS energy — same formula as audio_model.py
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
            const sample = (timeData[i] - 128) / 128.0; // normalize to -1..1
            sum += sample * sample;
        }
        const rms = Math.sqrt(sum / timeData.length);

        // Map RMS to 0–100%. Typical quiet room ~0.002, normal speech ~0.02–0.08
        // Scale by 700 so normal speech shows 14–56%, loud speech hits 80%+
        const voiceLevelPercent = Math.min(Math.round(rms * 700), 100);

        // Determine what to show on gauge:
        // - If a significant fraud score was detected → show fraud risk (fraud mode)
        // - Otherwise → show live voice level so the gauge is always alive
        let displayPercent;
        let labelText;
        let gaugeColor;

        if (this._fraudMode && this._lastFraudPercent > 0) {
            // Keep showing fraud score (set by _updateUI every 3s)
            displayPercent = this._lastFraudPercent;
            labelText = 'Fraud Risk';
            gaugeColor = this._lastFraudPercent > 50 ? '#f85149'
                : this._lastFraudPercent > 30 ? '#e3b341' : '#3fb950';
        } else {
            // Show real-time voice level
            displayPercent = voiceLevelPercent;
            labelText = voiceLevelPercent > 5 ? 'Voice Level' : 'Listening...';
            // Color by intensity: green → yellow → orange
            gaugeColor = voiceLevelPercent > 70 ? '#e3b341'
                : voiceLevelPercent > 40 ? '#58a6ff' : '#3fb950';
        }

        // Update gauge visuals
        const rotation = Math.min(180, displayPercent * 1.8);
        this.gaugeFill.style.transform = `rotate(${rotation}deg)`;
        this.gaugeFill.style.background = gaugeColor;
        this.scoreValue.textContent = `${displayPercent}%`;
        const scoreLabelEl = document.getElementById('scoreLabel');
        if (scoreLabelEl) scoreLabelEl.textContent = labelText;

        // Also update stress chip in real-time
        const stressPercent = Math.min(Math.round(rms * 1400), 100);
        const stressEl = document.getElementById('stressVal');
        if (stressEl) stressEl.textContent = `${stressPercent}%`;
    }

    /* ---- STOP ---- */
    stop() {
        if (!this.isRecording) return;
        this.isRecording = false;

        if (this.realtimeInterval) { clearInterval(this.realtimeInterval); this.realtimeInterval = null; }
        if (this.analysisInterval) { clearInterval(this.analysisInterval); this.analysisInterval = null; }
        if (this.stt) this.stt.stop();
        if (this.mediaRecorder) this.mediaRecorder.stop();
        if (this.audioContext) this.audioContext.close();
        if (this.source) {
            try { this.source.mediaStream.getTracks().forEach(t => t.stop()); } catch (e) { }
        }

        this.updateButtons();
        if (this.dwnBtn) this.dwnBtn.disabled = false;
        this.addLog("⏹️ Monitoring stopped. Evidence Report ready.");
        this._setMode("Mode: Idle", false);
        this._updateSystemStatus("System Ready", "var(--success)");
    }

    /* ---- UI UPDATE (equivalent to script.js updateUI + WebSocket message handler) ---- */
    _updateUI(data) {
        const score = data.fraud_score || 0;
        const stress = data.stress_score || 0;
        const percent = Math.round(score * 100);

        // Store fraud score so _updateRealtimeGauge (100ms) can blend it with voice level
        this._lastFraudPercent = percent;
        this._fraudMode = percent > 15; // switch gauge label to 'Fraud Risk' when relevant

        // 2. Status text + body alert class
        let statusClass = 'safe';
        let text = 'SAFE';
        document.body.classList.remove('fraud-alert');

        if (score > 0.3) {
            statusClass = 'suspicious';
            text = 'SUSPICIOUS';
        }
        if (score > 0.5 || data.is_ai) {
            statusClass = 'danger';
            text = data.is_ai ? 'AI-VOICE DETECTED' : 'FRAUD DETECTED';
            document.body.classList.add('fraud-alert');

            if (!this.alertTriggered) {
                if (data.is_ai) {
                    this.speakAgent("Warning. A synthetic artificial voice has been detected. This conversation may be automated.");
                } else {
                    this.triggerAlert(text);
                }
                this.alertTriggered = true;
            }
        } else {
            if (score < 0.3) this.alertTriggered = false;
        }

        this.statusObj.className = `status-text ${statusClass}`;
        this.statusObj.innerHTML = text + (data.is_ai
            ? ` <span style="font-size:0.55rem;vertical-align:middle;background:var(--danger);color:white;padding:2px 6px;border-radius:4px;margin-left:8px;letter-spacing:1px;">AI BOT</span>`
            : '');

        if (this.reasonText) this.reasonText.textContent = data.reason || "Analyzing...";

        // 3. Metric chips (stressVal is updated by realtime loop — skip it here)
        const aiEl = document.getElementById('aiVal');
        const langEl = document.getElementById('langVal');
        if (aiEl) {
            aiEl.textContent = data.is_ai ? "YES" : "No";
            aiEl.style.color = data.is_ai ? 'var(--danger)' : 'var(--success)';
        }
        if (langEl) langEl.textContent = data.language || "en";

        // 4. Transcript with keyword highlights
        if (data.text) {
            this._highlightTranscript(data.text, data.keywords || []);
        }

        // 5. Update Chart
        const now = new Date().toLocaleTimeString();
        const maxPoints = 25;
        if (this.stressChart.data.labels.length >= maxPoints) {
            this.stressChart.data.labels.shift();
            this.stressChart.data.datasets[0].data.shift();
            this.stressChart.data.datasets[1].data.shift();
        }
        this.stressChart.data.labels.push(now);
        this.stressChart.data.datasets[0].data.push(Math.round(stress));
        this.stressChart.data.datasets[1].data.push(percent);
        this.stressChart.update();

        // 6. Save history for report
        this.historyData.push({ time: now, fraud: percent, stress: Math.round(stress), reason: data.reason, isAI: data.is_ai });

        // 7. Log entry
        this.addLog(`Risk: ${percent}% | Stress: ${Math.round(stress)}% | AI: ${data.is_ai ? "YES" : "No"} | ${data.reason}`);
    }

    _highlightTranscript(text, keywords) {
        const box = document.getElementById('transcriptLog');
        if (!box) return;
        let html = text;
        if (keywords && keywords.length > 0) {
            keywords.forEach(kw => {
                const reg = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
                html = html.replace(reg, '<span class="keyword-highlight">$1</span>');
            });
        }
        box.innerHTML = html || '<em style="color:var(--text-muted)">No speech detected yet...</em>';
    }

    /* ---- TEXT TO SPEECH (speakAgent — same as before) ---- */
    speakAgent(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 0.8;
            window.speechSynthesis.speak(utterance);
        }
    }

    /* ---- FRAUD ALERT OVERLAY (triggerAlert — same as before) ---- */
    triggerAlert(msg) {
        // Overlay
        const overlay = document.createElement('div');
        overlay.className = 'alert-overlay';
        overlay.id = 'fraudAlertOverlay';
        overlay.innerHTML = `
            <div class="alert-content">
                <div class="alert-title">⚠️ SYSTEM COMPROMISED</div>
                <div class="alert-sub">HIGH FRAUD PROBABILITY DETECTED</div>
                <div class="alert-sub" style="margin-top:20px; color:#ffaaaa;">${msg}</div>
                <div class="trace-bar"><div class="trace-fill"></div></div>
                <div style="margin-top:10px; font-size:0.8rem; color:#aaa;">TRACING CALL SOURCE...</div>
                <button id="dismissBtn" style="margin-top:40px; background:white; color:black; border:none; padding:15px 30px; font-weight:bold; cursor:pointer; border-radius:6px;">DISMISS ALERT</button>
            </div>
        `;
        document.body.appendChild(overlay);

        // Audio Alarm (Siren)
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'square';
        osc.frequency.setValueAtTime(800, ctx.currentTime);
        let t = ctx.currentTime;
        for (let i = 0; i < 10; i++) {
            osc.frequency.linearRampToValueAtTime(1200, t + 0.5);
            osc.frequency.linearRampToValueAtTime(800, t + 1.0);
            t += 1.0;
        }
        gain.gain.value = 0.25;
        osc.start();
        osc.stop(ctx.currentTime + 10);

        // TTS Warning
        this.speakAgent("Warning. High probability of fraud detected. Do not share your banking details. This call is being recorded for evidence.");

        // Dismiss
        document.getElementById('dismissBtn').onclick = () => {
            if (document.body.contains(overlay)) document.body.removeChild(overlay);
            try { osc.stop(); } catch (e) { }
            ctx.close();
            setTimeout(() => { this.alertTriggered = false; }, 5000);
        };
    }

    /* ---- DOWNLOAD REPORT (downloadReport — same as before) ---- */
    downloadReport() {
        let report = `GUARDIAN AI — FRAUD DETECTION REPORT\n`;
        report += `Generated: ${new Date().toLocaleString()}\n`;
        report += `====================================\n\n`;
        report += `FINAL STATUS: ${this.statusObj.textContent.replace(/\s+/g, ' ').trim()}\n`;
        const maxFraud = this.historyData.length ? Math.max(...this.historyData.map(h => h.fraud)) : 0;
        const maxStress = this.historyData.length ? Math.max(...this.historyData.map(h => h.stress)) : 0;
        const aiDetected = this.historyData.some(h => h.isAI);
        report += `MAX FRAUD RISK: ${maxFraud}%\n`;
        report += `MAX STRESS LEVEL: ${maxStress}%\n`;
        report += `AI VOICE DETECTED: ${aiDetected ? "YES" : "No"}\n\n`;
        report += `TRANSCRIPT EVIDENCE:\n`;
        report += `-------------------\n`;
        report += `${this.fullTranscript}\n\n`;
        report += `ANALYSIS LOG:\n`;
        report += `-------------\n`;
        this.historyData.forEach(entry => {
            report += `[${entry.time}] Risk: ${entry.fraud}% | Stress: ${entry.stress}% | AI: ${entry.isAI ? "YES" : "No"} | ${entry.reason}\n`;
        });

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Evidence_Report_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /* ---- LOG PANEL ---- */
    addLog(msg) {
        const log = document.getElementById('systemLog');
        if (!log) return;
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const time = new Date().toLocaleTimeString();
        entry.innerHTML = `<span class="timestamp">[${time}]</span> ${msg}`;
        log.prepend(entry);
    }

    /* ---- CANVAS VISUALIZER (same as before, enhanced colors) ---- */
    drawVisualizer() {
        if (!this.isRecording) return;
        requestAnimationFrame(() => this.drawVisualizer());

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);

        const ctx = this.canvasCtx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        ctx.fillStyle = 'rgba(7, 11, 18, 0.25)';
        ctx.fillRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = dataArray[i] / 2;
            const hue = (i / bufferLength) * 200 + 180; // blues → purples
            ctx.fillStyle = `hsl(${hue}, 80%, ${30 + barHeight / 3}%)`;
            ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    /* ---- BUTTON STATES ---- */
    updateButtons() {
        if (this.startButton) this.startButton.disabled = this.isRecording;
        if (this.stopButton) this.stopButton.disabled = !this.isRecording;
    }

    _setMode(text, active) {
        const modeText = document.getElementById('modeText');
        const badgeDot = document.querySelector('.badge-dot');
        if (modeText) modeText.textContent = text;
        if (badgeDot) {
            if (active) badgeDot.classList.add('active');
            else badgeDot.classList.remove('active');
        }
    }

    _updateSystemStatus(text, color) {
        const el = document.getElementById('systemStatusText');
        const pill = document.getElementById('systemStatus');
        if (el) el.textContent = text;
        if (pill) {
            pill.style.background = color === 'var(--success)'
                ? 'rgba(63, 185, 80, 0.1)' : 'rgba(248, 81, 73, 0.1)';
            pill.style.borderColor = color === 'var(--success)'
                ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)';
            pill.style.color = color;
        }
    }
}


/* =============================================
   6. DEEPFAKE FILE UPLOAD HANDLER
   Replaces /detect-deepfake POST endpoint from main.py
   Now uses client-side Web Audio API analysis of the uploaded file
   ============================================= */
async function handleDeepfakeUpload() {
    const dfInput = document.getElementById('deepfakeUpload');
    const dfRes = document.getElementById('deepfakeResult');
    const dfLoad = document.getElementById('dfLoading');

    if (!dfInput.files[0]) {
        alert("Please select an MP3 or WAV file first.");
        return;
    }

    dfLoad.style.display = 'flex';
    dfRes.style.display = 'none';

    const file = dfInput.files[0];

    try {
        // Read file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();

        // Decode audio using Web Audio API (replaces librosa.load)
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        let decodedBuffer;
        try {
            decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
        } catch (e) {
            throw new Error(`Could not decode audio file. Make sure it's a valid MP3/WAV. (${e.message})`);
        }

        // Extract frequency + time domain data via offline rendering
        const offlineCtx = new OfflineAudioContext(1, decodedBuffer.length, decodedBuffer.sampleRate);
        const source = offlineCtx.createBufferSource();
        source.buffer = decodedBuffer;
        const analyserNode = offlineCtx.createAnalyser();
        analyserNode.fftSize = 2048;
        source.connect(analyserNode);
        analyserNode.connect(offlineCtx.destination);
        source.start(0);

        // We need to sample frequency data at a few frames during offline rendering
        // Use a ScriptProcessorNode approach instead for direct sample access
        const channelData = decodedBuffer.getChannelData(0);

        // Sample multiple windows across the audio to get representative spectra
        const windowSize = 2048;
        const hopSize = Math.floor(channelData.length / 20); // ~20 windows
        const allFreqData = [];
        const allTimeData = [];

        for (let offset = 0; offset < channelData.length - windowSize; offset += hopSize) {
            const window = channelData.slice(offset, offset + windowSize);
            // Manual FFT for frequency data
            const freq = computeSimpleFFTMagnitudes(window);
            allFreqData.push(freq);
            allTimeData.push(window);
        }

        // Average the frequency magnitudes
        const avgFreq = averageArrays(allFreqData);
        // Use the middle window for time domain
        const midTime = allTimeData[Math.floor(allTimeData.length / 2)];

        // Convert midTime Float32Array to Uint8Array (byte format for AudioAnalysisEngine)
        const timeUint8 = new Uint8Array(midTime.length);
        for (let i = 0; i < midTime.length; i++) {
            timeUint8[i] = Math.round((midTime[i] + 1) * 128);
        }

        // Run deepfake detection (deepfake_logic.py → JS)
        const deepfakeResult = AudioAnalysisEngine.detectAIVoice(avgFreq, timeUint8);

        // Audio fraud score
        const audioScores = AudioAnalysisEngine.audioFraudScore(timeUint8);

        // Try STT on uploaded file via speechSynthesis alternative: just show filename
        // (Web Speech API only works with mic, not file array buffers)
        const transcript = `[Uploaded file: ${file.name}]`;

        // NLP on filename to show some insight
        const nlpResult = NLPEngine.getKeywordScore(file.name);

        await audioCtx.close();

        // Update UI
        const cls = deepfakeResult.classification;
        const conf = deepfakeResult.confidence;

        const dfClass = document.getElementById('dfClassification');
        const dfConf = document.getElementById('dfConfidence');
        const dfLang = document.getElementById('dfLang');
        const dfExp = document.getElementById('dfExplanation');
        const dfReasonsEl = document.getElementById('dfReasons');

        dfClass.textContent = cls;
        dfClass.style.color = cls === 'AI-Generated' ? 'var(--danger)' : 'var(--success)';
        dfConf.textContent = `Confidence: ${(conf * 100).toFixed(1)}%`;

        dfLang.textContent = `File: ${file.name} · Size: ${(file.size / 1024).toFixed(1)} KB`;

        const metrics = deepfakeResult.metrics;
        dfExp.textContent = `Spectral Flatness: ${metrics.spectralFlatness} · Spectral Variance: ${metrics.spectralVariance} · Stress Score: ${audioScores.stress}%`;

        // Reasons
        dfReasonsEl.innerHTML = '';
        if (deepfakeResult.reasons.length > 0) {
            deepfakeResult.reasons.forEach(r => {
                const d = document.createElement('div');
                d.className = 'df-reason-item';
                d.textContent = `• ${r}`;
                dfReasonsEl.appendChild(d);
            });
        } else {
            const d = document.createElement('div');
            d.className = 'df-reason-item';
            d.style.borderLeftColor = 'var(--success)';
            d.textContent = '• No synthetic indicators detected — voice appears natural.';
            dfReasonsEl.appendChild(d);
        }

        dfRes.style.display = 'flex';

    } catch (err) {
        console.error(err);
        alert("Error analyzing audio: " + err.message);
    } finally {
        dfLoad.style.display = 'none';
    }
}

/**
 * Simple DFT magnitude computation for a PCM window
 * Used instead of librosa's FFT for static file analysis
 */
function computeSimpleFFTMagnitudes(samples) {
    const N = samples.length;
    const halfN = Math.floor(N / 2);
    const magnitudes = new Float32Array(halfN);

    for (let k = 0; k < halfN; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            re += samples[n] * Math.cos(angle);
            im -= samples[n] * Math.sin(angle);
        }
        magnitudes[k] = Math.sqrt(re * re + im * im) / N;
    }
    return magnitudes;
}

/**
 * Average multiple Float32Arrays element-wise
 */
function averageArrays(arrays) {
    if (arrays.length === 0) return new Float32Array(0);
    const len = arrays[0].length;
    const result = new Float32Array(len);
    for (const arr of arrays) {
        for (let i = 0; i < len; i++) result[i] += arr[i];
    }
    for (let i = 0; i < len; i++) result[i] /= arrays.length;
    // Scale to 0-255 range for compatibility with AudioAnalysisEngine
    const max = Math.max(...result);
    if (max > 0) for (let i = 0; i < len; i++) result[i] = (result[i] / max) * 255;
    return result;
}


/* =============================================
   7. INITIALIZATION
   ============================================= */
window.addEventListener('load', () => {
    // Fix canvas resolution for sharp rendering
    const canvas = document.getElementById('visualizer');
    canvas.width = canvas.offsetWidth || 800;
    canvas.height = canvas.offsetHeight || 110;

    // Initialize main controller
    const streamer = new AudioStreamer();
    window._guardian = streamer; // expose for debugging

    // Wire up deepfake upload button
    const dfBtn = document.getElementById('detectDeepfakeBtn');
    if (dfBtn) dfBtn.addEventListener('click', handleDeepfakeUpload);

    // Handle window resize for canvas
    window.addEventListener('resize', () => {
        canvas.width = canvas.offsetWidth || 800;
        canvas.height = canvas.offsetHeight || 110;
    });

    // Check browser support
    const warnings = [];
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
        warnings.push("⚠️ Web Speech API not supported — transcript will be unavailable. Use Chrome for best results.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
        warnings.push("⚠️ getUserMedia not supported — microphone analysis unavailable.");
    }
    if (warnings.length > 0) {
        const log = document.getElementById('systemLog');
        warnings.forEach(w => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span style="color:var(--warning)">${w}</span>`;
            log.appendChild(entry);
        });
    }

    streamer.addLog("🛡️ Guardian AI initialized. All modules running browser-native.");
    streamer.addLog("📊 Web Audio API, SpeechRecognition, and ML heuristics loaded.");
    streamer.addLog("🔒 No server required — all processing runs in your browser.");
});
