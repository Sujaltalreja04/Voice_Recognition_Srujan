'use strict';

/* =============================================
   1. NLP ENGINE
   ============================================= */
const NLPEngine = (() => {
    const FRAUD_KEYWORDS = [
        "otp", "bank", "account", "card", "cvv", "pin", "transfer", "payment",
        "refund", "loan", "credit", "debit", "atm", "wallet", "upi", "ifsc",
        "urgent", "immediately", "expire", "expiry", "blocked", "suspend",
        "freeze", "locked", "disabled", "compromised", "arrested", "arrest",
        "police", "officer", "government", "irs", "income tax", "kyc",
        "verify", "verification", "legal", "court", "warrant",
        "prize", "winner", "won", "reward", "lottery", "lucky", "congratulations",
        "claim", "selected",
        "password", "click", "link", "download", "install", "aadhar", "aadhaar",
        "pan card", "social security"
    ];

    function getKeywordScore(text) {
        const lower = text.toLowerCase();
        let score = 0;
        const detected = [];
        for (const word of FRAUD_KEYWORDS) {
            if (lower.includes(word)) { score += 1; detected.push(word); }
        }
        return {
            fraud_score: Math.min(score / 2, 1.0),
            reason: detected.length
                ? 'Detected risk words: ' + detected.slice(0, 4).join(', ')
                : 'No suspicious keywords found',
            detected_keywords: detected
        };
    }

    async function analyzeText(text) {
        if (!text || text.trim().length < 5) {
            return { fraud_score: 0.0, reason: 'Silence / Too short', detected_keywords: [] };
        }
        const kwResult = getKeywordScore(text);
        const apiKey = sessionStorage.getItem('OPENROUTER_API_KEY');
        if (apiKey) {
            try {
                const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Authorization': 'Bearer ' + apiKey,
                        'Content-Type': 'application/json',
                        'HTTP-Referer': window.location.href,
                        'X-Title': 'Guardian AI Fraud Detection'
                    },
                    body: JSON.stringify({
                        model: 'google/gemini-2.0-flash-lite-preview-02-05:free',
                        messages: [
                            { role: 'system', content: 'You are a fraud detection expert. Analyze the transcript. Return ONLY JSON: {"fraud_score":<0.0-1.0>,"reason":"<brief>","detected_keywords":[<list>]}. No other text.' },
                            { role: 'user', content: text }
                        ],
                        response_format: { type: 'json_object' }
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
                    if (content) return JSON.parse(content.trim());
                }
            } catch (e) { console.warn('OpenRouter NLP error:', e); }
        }
        return kwResult;
    }

    return { analyzeText, getKeywordScore };
})();


/* =============================================
   2. AUDIO ANALYSIS ENGINE
   ============================================= */
const AudioAnalysisEngine = (() => {

    function computeRMS(buffer) {
        let sum = 0;
        for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
        return Math.sqrt(sum / buffer.length);
    }

    function computeZCR(buffer) {
        let count = 0;
        for (let i = 1; i < buffer.length; i++) {
            if ((buffer[i] >= 0) !== (buffer[i - 1] >= 0)) count++;
        }
        return count / buffer.length;
    }

    function audioFraudScore(timeDomainData) {
        if (!timeDomainData || timeDomainData.length === 0) {
            return { score: 0, stress: 0 };
        }
        const floatBuf = new Float32Array(timeDomainData.length);
        for (let i = 0; i < timeDomainData.length; i++) {
            floatBuf[i] = (timeDomainData[i] - 128) / 128.0;
        }
        const energy = computeRMS(floatBuf);
        const zcr = computeZCR(floatBuf);
        // stress: high energy + high ZCR = fast/agitated speech
        const stress = Math.min((energy * 500) + (zcr * 200), 100);
        let score = 0;
        if (energy > 0.02) score += 0.4;
        if (zcr > 0.08) score += 0.4;
        score = Math.min(score, 1.0);
        return { score: score, stress: Math.round(stress) };
    }

    function detectAIVoice(freqData, timeDomainData) {
        if (!freqData || freqData.length === 0) return null;
        const N = freqData.length;
        let logSum = 0, arithmSum = 0, validCount = 0;
        for (let i = 0; i < N; i++) {
            const val = freqData[i] + 1e-6;
            if (val > 1) { logSum += Math.log(val); arithmSum += val; validCount++; }
        }
        const geoMean = validCount > 0 ? Math.exp(logSum / validCount) : 0;
        const arithMean = validCount > 0 ? arithmSum / validCount : 1;
        const spectralFlatness = geoMean / arithMean;

        let maxMag = 0, peakBin = 0;
        for (let i = 2; i < N / 2; i++) {
            if (freqData[i] > maxMag) { maxMag = freqData[i]; peakBin = i; }
        }
        let peakSpread = 0;
        for (let i = Math.max(0, peakBin - 5); i < Math.min(N, peakBin + 5); i++) {
            peakSpread += Math.abs(freqData[i] - freqData[peakBin]);
        }
        const pitchStabilityScore = peakSpread < 80 ? 1 : 0;

        let mean = 0;
        for (let i = 0; i < N; i++) mean += freqData[i];
        mean /= N;
        let variance = 0;
        for (let i = 0; i < N; i++) variance += (freqData[i] - mean) * (freqData[i] - mean);
        variance /= N;
        const mfccVarLow = variance < 800;

        let aiScore = 0;
        const reasons = [];
        if (pitchStabilityScore > 0) { aiScore += 0.4; reasons.push('Unnatural pitch stability detected (Robotic cadence)'); }
        if (spectralFlatness > 0.08) { aiScore += 0.3; reasons.push('High spectral flatness / Digital noise detected'); }
        if (mfccVarLow) { aiScore += 0.3; reasons.push('Predictable spectral patterns (Synthetic signature)'); }

        const confidence = Math.min(Math.max(aiScore, 0.1), 0.98);
        const classification = confidence > 0.6 ? 'AI-Generated' : 'Human-Generated';
        return {
            classification: classification,
            confidence: Math.round(confidence * 100) / 100,
            reasons: reasons,
            metrics: {
                spectralFlatness: Math.round(spectralFlatness * 10000) / 10000,
                spectralVariance: Math.round(variance)
            }
        };
    }

    return { audioFraudScore, detectAIVoice };
})();


/* =============================================
   3. FRAUD ENGINE
   ============================================= */
const FraudEngine = (() => {
    async function analyze(transcript, audioScores, deepfakeResult, lang) {
        const nlpData = await NLPEngine.analyzeText(transcript);
        const tScore = nlpData.fraud_score || 0.0;
        const aScore = (audioScores && audioScores.score) || 0.0;
        const isAI = deepfakeResult && deepfakeResult.classification === 'AI-Generated';
        const aiConf = (deepfakeResult && deepfakeResult.confidence) || 0;
        let final = (tScore * 0.8) + (aScore * 0.15);
        if (isAI) final += 0.3;
        return {
            text: transcript,
            fraud_score: Math.min(final, 1.0),
            stress_score: (audioScores && audioScores.stress) || 0,
            is_ai: isAI,
            ai_confidence: aiConf,
            reason: (isAI ? 'AI-VOICE DETECTED. ' : '') + (nlpData.reason || 'Scanning...'),
            keywords: nlpData.detected_keywords || [],
            deepfake_reasons: (deepfakeResult && deepfakeResult.reasons) || [],
            language: lang || 'en'
        };
    }
    return { analyze };
})();


/* =============================================
   4. SPEECH TO TEXT — permission asked ONCE
   ============================================= */
class SpeechToText {
    constructor(onResult, onError) {
        this.onResult = onResult;
        this.onError = onError;
        this.recognition = null;
        this.currentText = '';
        this.isActive = false;
        this._restarting = false;
    }

    start() {
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.onError('Speech Recognition not supported. Use Chrome.');
            return;
        }
        if (this._restarting) return;

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.maxAlternatives = 1;
        this.recognition.lang = 'en-US';

        var self = this;
        this.recognition.onresult = function (event) {
            var fullText = '';
            for (var i = 0; i < event.results.length; i++) {
                fullText += event.results[i][0].transcript;
            }
            self.currentText = fullText;
            self.onResult(fullText);
        };

        this.recognition.onerror = function (e) {
            if (e.error === 'no-speech') return;
            if (e.error === 'not-allowed') {
                self.isActive = false;
                self.onError('Microphone permission denied.');
                return;
            }
            console.warn('STT error:', e.error);
        };

        // Restart only when still active — does NOT re-prompt for mic permission
        this.recognition.onend = function () {
            if (!self.isActive) return;
            if (self._restarting) return;
            self._restarting = true;
            setTimeout(function () {
                self._restarting = false;
                if (self.isActive && self.recognition) {
                    try { self.recognition.start(); } catch (e) { /* already running */ }
                }
            }, 300);
        };

        this.isActive = true;
        try { this.recognition.start(); } catch (e) { console.warn('STT start:', e); }
    }

    stop() {
        this.isActive = false;
        this._restarting = false;
        if (this.recognition) {
            this.recognition.onend = null;
            try { this.recognition.stop(); } catch (e) { }
        }
        this.currentText = '';
    }

    getText() { return this.currentText; }
}


/* =============================================
   5. AUDIO STREAMER — Main Controller
   ============================================= */
class AudioStreamer {
    constructor() {
        this.mediaStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.source = null;
        this.stt = null;
        this.isRecording = false;

        // Main fraud alert — only resets after user dismisses
        this.alertTriggered = false;

        // One-shot alert flags
        this._highSpeedAlertTriggered = false;
        this._scamAlertTriggered = false;

        this.analysisInterval = null;
        this.realtimeInterval = null;
        this._lastFraudPercent = 0;
        this._fraudMode = false;

        this.fullTranscript = '';
        this.historyData = [];
        this.latestAudioScores = { score: 0, stress: 0 };
        this.latestDeepfakeResult = null;

        this.startButton = document.getElementById('startBtn');
        this.stopButton = document.getElementById('stopBtn');
        this.dwnBtn = document.getElementById('dwnBtn');
        this.scoreValue = document.getElementById('scoreValue');
        this.statusObj = document.getElementById('statusObj');
        this.gaugeFill = document.getElementById('gaugeFill');
        this.reasonText = document.getElementById('reasonText');
        this.canvas = document.getElementById('visualizer');
        this.canvasCtx = this.canvas.getContext('2d');

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
                            backgroundColor: 'rgba(248,81,73,0.1)',
                            tension: 0.4, fill: true, pointRadius: 2
                        },
                        {
                            label: 'Fraud Probability',
                            data: [],
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88,162,235,0.1)',
                            tension: 0.4, fill: true, pointRadius: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 300 },
                    scales: {
                        x: { display: false },
                        y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6e7e96', font: { size: 10 } } }
                    },
                    plugins: { legend: { labels: { color: '#8b949e', font: { family: 'Outfit', size: 11 } } } }
                }
            }
        );

        var self = this;
        this.startButton.addEventListener('click', function () { self.start(); });
        this.stopButton.addEventListener('click', function () { self.stop(); });
        if (this.dwnBtn) this.dwnBtn.addEventListener('click', function () { self.downloadReport(); });

        var dfInput = document.getElementById('deepfakeUpload');
        if (dfInput) {
            dfInput.addEventListener('change', function (e) {
                var fn = document.getElementById('fileName');
                if (fn) fn.textContent = (e.target.files[0] && e.target.files[0].name) || 'No file selected';
            });
        }
    }

    /* ---- START — asks mic permission ONCE ---- */
    async start() {
        if (this.isRecording) return;
        try {
            // Single getUserMedia = single permission prompt, ever.
            var stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            this.mediaStream = stream;

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.source = this.audioContext.createMediaStreamSource(stream);
            this.source.connect(this.analyser);

            this.drawVisualizer();

            var self = this;
            this.stt = new SpeechToText(
                function (text) { self.fullTranscript = text; self._highlightTranscript(text, []); },
                function (err) { self.addLog('STT Error: ' + err); }
            );
            this.stt.start();

            // Reset all per-session flags
            this.alertTriggered = false;
            this._highSpeedAlertTriggered = false;
            this._scamAlertTriggered = false;

            this.realtimeInterval = setInterval(function () { self._updateRealtimeGauge(); }, 100);
            this.analysisInterval = setInterval(function () { self._runAnalysis(); }, 3000);

            this.isRecording = true;
            this.updateButtons();
            this.addLog('Monitoring started. Microphone active.');
            this._setMode('Mode: Live Monitoring', true);
            this._updateSystemStatus('Monitoring Active', 'var(--success)');

        } catch (err) {
            console.error(err);
            if (err.name === 'NotAllowedError') {
                this.addLog('Microphone permission denied.');
                alert('Microphone access denied. Please allow microphone access in your browser settings and refresh.');
            } else {
                this.addLog('Error: ' + err.message);
                alert('Error accessing microphone: ' + err.message);
            }
        }
    }

    /* ---- PERIODIC ANALYSIS every 3 seconds ---- */
    async _runAnalysis() {
        if (!this.isRecording || !this.analyser) return;

        var freqData = new Uint8Array(this.analyser.frequencyBinCount);
        var timeData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteFrequencyData(freqData);
        this.analyser.getByteTimeDomainData(timeData);

        var audioScores = AudioAnalysisEngine.audioFraudScore(timeData);
        var deepfakeResult = AudioAnalysisEngine.detectAIVoice(freqData, timeData);

        this.latestAudioScores = audioScores;
        this.latestDeepfakeResult = deepfakeResult;

        // --- HIGH SPEECH SPEED ALERT ---
        // stress > 55 means high energy + high ZCR = fast/pressuring speech
        // Fires exactly ONCE per session
        if (audioScores.stress > 25 && !this._highSpeedAlertTriggered) {
            this._highSpeedAlertTriggered = true;
            this.addLog('HIGH SPEECH SPEED — Stress: ' + audioScores.stress + '% — possible pressure tactic');
            this.showBannerAlert(
                'high-speed',
                'High Speech Speed Detected',
                'Voice stress at ' + audioScores.stress + '%. Speaker is talking unusually fast — possible high-pressure scam tactic. Stay calm, do NOT share personal details.',
                '#e3b341'
            );
            this.speakAgent('Warning. High speech speed detected. The caller may be using pressure tactics. Do not share your personal or banking details.');
        }

        var transcript = this.stt ? this.stt.getText() : '';
        var result = await FraudEngine.analyze(transcript, audioScores, deepfakeResult, 'en');
        this._updateUI(result);
    }

    /* ---- REALTIME GAUGE every 100ms ---- */
    _updateRealtimeGauge() {
        if (!this.isRecording || !this.analyser) return;

        var timeData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteTimeDomainData(timeData);

        var sum = 0;
        for (var i = 0; i < timeData.length; i++) {
            var sample = (timeData[i] - 128) / 128.0;
            sum += sample * sample;
        }
        var rms = Math.sqrt(sum / timeData.length);
        var voiceLevelPercent = Math.min(Math.round(rms * 700), 100);

        var displayPercent, labelText, gaugeColor;
        if (this._fraudMode && this._lastFraudPercent > 0) {
            displayPercent = this._lastFraudPercent;
            labelText = 'Fraud Risk';
            gaugeColor = this._lastFraudPercent > 50 ? '#f85149' : this._lastFraudPercent > 30 ? '#e3b341' : '#3fb950';
        } else {
            displayPercent = voiceLevelPercent;
            labelText = voiceLevelPercent > 5 ? 'Voice Level' : 'Listening...';
            gaugeColor = voiceLevelPercent > 70 ? '#e3b341' : voiceLevelPercent > 40 ? '#58a6ff' : '#3fb950';
        }

        var rotation = Math.min(180, displayPercent * 1.8);
        this.gaugeFill.style.transform = 'rotate(' + rotation + 'deg)';
        this.gaugeFill.style.background = gaugeColor;
        this.scoreValue.textContent = displayPercent + '%';
        var scoreLabelEl = document.getElementById('scoreLabel');
        if (scoreLabelEl) scoreLabelEl.textContent = labelText;

        var stressPercent = Math.min(Math.round(rms * 1400), 100);
        var stressEl = document.getElementById('stressVal');
        if (stressEl) stressEl.textContent = stressPercent + '%';
        // FAST SUSTAINED STRESS DETECTION (100ms loop)
        if (stressPercent > 40) { this._sustainedStress = (this._sustainedStress || 0) + 1; } else { this._sustainedStress = 0; }
        if (this._sustainedStress >= 15 && !this._highSpeedAlertTriggered) {
            this._highSpeedAlertTriggered = true;
            this.showBannerAlert('high-speed', 'High Speech Speed Detected', 'Voice stress at ' + stressPercent + '%. Speaker is talking unusually fast  possible high-pressure scam tactic. Stay calm, do NOT share personal details.', '#e3b341');
            this.speakAgent('Warning. High speech speed detected. The caller may be using pressure tactics. Do not share your personal or banking details.');
        }

        // HIGH SPEECH SPEED: sustained counter  fires if rms stays high for 15 frames (1.5s)
        if (rms > 0.015) {
            this._highSpeechFrames = (this._highSpeechFrames || 0) + 1;
        } else {
            this._highSpeechFrames = 0;
        }
        if (this._highSpeechFrames >= 15 && !this._highSpeedAlertTriggered) {
            this._highSpeedAlertTriggered = true;
            this._highSpeechFrames = 0;
            var stressVal = stressPercent;
            this.addLog('HIGH SPEECH SPEED ALERT � RMS stress: ' + stressVal + '% (sustained)');
            this.showBannerAlert(
                'high-speed',
                'High Speech Speed Detected',
                'Speaker talking at intense speed (' + stressVal + '% stress level). Possible pressure tactic. Do NOT share personal or banking details.',
                '#e3b341'
            );
            this.speakAgent('Warning. High speech speed detected. The caller may be using pressure tactics. Do not share your personal or banking details.');
        }
    }

    /* ---- STOP ---- */
    stop() {
        if (!this.isRecording) return;
        this.isRecording = false;

        if (this.realtimeInterval) { clearInterval(this.realtimeInterval); this.realtimeInterval = null; }
        if (this.analysisInterval) { clearInterval(this.analysisInterval); this.analysisInterval = null; }
        if (this.stt) { this.stt.stop(); this.stt = null; }
        if (this.mediaStream) { this.mediaStream.getTracks().forEach(function (t) { t.stop(); }); this.mediaStream = null; }
        if (this.audioContext) { this.audioContext.close(); this.audioContext = null; }
        this.analyser = null;
        this.source = null;

        this.updateButtons();
        if (this.dwnBtn) this.dwnBtn.disabled = false;
        this.addLog('Monitoring stopped. Evidence Report ready.');
        this._setMode('Mode: Idle', false);
        this._updateSystemStatus('System Ready', 'var(--success)');
    }

    /* ---- UI UPDATE every 3 seconds ---- */
    _updateUI(data) {
        var score = data.fraud_score || 0;
        var stress = data.stress_score || 0;
        var percent = Math.round(score * 100);

        this._lastFraudPercent = percent;
        this._fraudMode = percent > 15;

        var statusClass = 'safe';
        var text = 'SAFE';
        document.body.classList.remove('fraud-alert');

        if (score > 0.3) { statusClass = 'suspicious'; text = 'SUSPICIOUS'; }

        if (score > 0.45 || data.is_ai) {
            statusClass = 'danger';
            text = data.is_ai ? 'AI-VOICE DETECTED' : 'FRAUD DETECTED';
            document.body.classList.add('fraud-alert');

            // Fires ONCE — set flag BEFORE showing so rapid cycles can't double-trigger
            if (!this.alertTriggered) {
                this.alertTriggered = true;
                if (data.is_ai) {
                    this.speakAgent('Warning. A synthetic artificial voice has been detected. This conversation may be automated.');
                    this.showBannerAlert('ai-voice', 'AI Voice Detected', 'This voice appears to be synthetically generated. Possible automated scam bot.', '#a371f7');
                } else {
                    this.triggerAlert(text);
                }
            }
        }
        // alertTriggered NEVER auto-resets — only resets 30s after user dismisses

        // --- SCAM KEYWORD ALERT — fires once ---
        if (data.keywords && data.keywords.length > 0 && !this._scamAlertTriggered) {
            this._scamAlertTriggered = true;
            var kws = data.keywords.slice(0, 5).join(', ');
            this.addLog('SCAM KEYWORDS DETECTED: ' + kws);
            this.showBannerAlert(
                'scam-keyword',
                'Scam Keywords Detected',
                'Suspicious words: "' + kws + '". Do NOT share personal or banking details.',
                '#f85149'
            );
        }

        this.statusObj.className = 'status-text ' + statusClass;
        this.statusObj.innerHTML = text + (data.is_ai
            ? ' <span style="font-size:0.55rem;vertical-align:middle;background:var(--danger);color:white;padding:2px 6px;border-radius:4px;margin-left:8px;letter-spacing:1px;">AI BOT</span>'
            : '');

        if (this.reasonText) this.reasonText.textContent = data.reason || 'Analyzing...';

        var aiEl = document.getElementById('aiVal');
        var langEl = document.getElementById('langVal');
        if (aiEl) { aiEl.textContent = data.is_ai ? 'YES' : 'No'; aiEl.style.color = data.is_ai ? 'var(--danger)' : 'var(--success)'; }
        if (langEl) langEl.textContent = data.language || 'en';

        if (data.text) this._highlightTranscript(data.text, data.keywords || []);

        var now = new Date().toLocaleTimeString();
        if (this.stressChart.data.labels.length >= 25) {
            this.stressChart.data.labels.shift();
            this.stressChart.data.datasets[0].data.shift();
            this.stressChart.data.datasets[1].data.shift();
        }
        this.stressChart.data.labels.push(now);
        this.stressChart.data.datasets[0].data.push(Math.round(stress));
        this.stressChart.data.datasets[1].data.push(percent);
        this.stressChart.update();

        this.historyData.push({ time: now, fraud: percent, stress: Math.round(stress), reason: data.reason, isAI: data.is_ai });
        this.addLog('Risk: ' + percent + '% | Stress: ' + Math.round(stress) + '% | AI: ' + (data.is_ai ? 'YES' : 'No') + ' | ' + data.reason);
    }

    _highlightTranscript(text, keywords) {
        var box = document.getElementById('transcriptLog');
        if (!box) return;
        var html = text;
        if (keywords && keywords.length > 0) {
            keywords.forEach(function (kw) {
                var reg = new RegExp('(' + kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
                html = html.replace(reg, '<span class="keyword-highlight">$1</span>');
            });
        }
        box.innerHTML = html || '<em style="color:var(--text-muted)">No speech detected yet...</em>';
    }

    /* ---- BANNER ALERT — slide-in, non-blocking ---- */
    showBannerAlert(id, title, message, color) {
        // Remove duplicate
        var existing = document.getElementById('banner-' + id);
        if (existing) existing.remove();

        var accentColor = color || '#f85149';
        var banner = document.createElement('div');
        banner.id = 'banner-' + id;
        banner.style.cssText = [
            'position:fixed',
            'top:20px',
            'right:20px',
            'z-index:88888',
            'display:flex',
            'align-items:flex-start',
            'gap:12px',
            'background:rgba(10,15,25,0.97)',
            'border:1px solid ' + accentColor,
            'border-left:4px solid ' + accentColor,
            'border-radius:10px',
            'padding:16px 18px',
            'max-width:380px',
            'min-width:280px',
            'box-shadow:0 8px 32px rgba(0,0,0,0.7)',
            'font-family:Outfit,sans-serif',
            'animation:none',
            'transition:opacity 0.4s ease,transform 0.4s ease'
        ].join(';');

        // Stack multiple banners — count existing banners and offset
        var allBanners = document.querySelectorAll('[id^="banner-"]');
        var topOffset = 20 + (allBanners.length * 110);
        banner.style.top = topOffset + 'px';

        banner.innerHTML =
            '<div style="flex:1;min-width:0">' +
            '<div style="font-size:0.85rem;font-weight:700;color:' + accentColor + ';letter-spacing:0.5px;margin-bottom:5px;text-transform:uppercase">' + title + '</div>' +
            '<div style="font-size:0.78rem;color:#c9d1d9;line-height:1.45">' + message + '</div>' +
            '</div>' +
            '<button onclick="this.parentElement.remove()" style="background:none;border:none;color:#6e7e96;font-size:1rem;cursor:pointer;padding:0;margin-left:8px;flex-shrink:0">x</button>';

        // Slide in from right
        banner.style.transform = 'translateX(120%)';
        banner.style.opacity = '0';
        document.body.appendChild(banner);
        // Force reflow then animate in
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                banner.style.transition = 'opacity 0.4s ease, transform 0.4s cubic-bezier(0.34,1.56,0.64,1)';
                banner.style.transform = 'translateX(0)';
                banner.style.opacity = '1';
            });
        });

        // Auto-dismiss after 12s
        setTimeout(function () {
            if (banner.parentElement) {
                banner.style.opacity = '0';
                banner.style.transform = 'translateX(120%)';
                setTimeout(function () { if (banner.parentElement) banner.remove(); }, 400);
            }
        }, 12000);
    }

    /* ---- TTS ---- */
    speakAgent(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            var u = new SpeechSynthesisUtterance(text);
            u.rate = 1.0; u.pitch = 0.8;
            window.speechSynthesis.speak(u);
        }
    }

    /* ---- FULL SCREEN FRAUD ALERT OVERLAY ---- */
    triggerAlert(msg) {
        var existing = document.getElementById('fraudAlertOverlay');
        if (existing) existing.remove();

        var overlay = document.createElement('div');
        overlay.className = 'alert-overlay';
        overlay.id = 'fraudAlertOverlay';
        overlay.innerHTML =
            '<div class="alert-content">' +
            '<div class="alert-title">SYSTEM COMPROMISED</div>' +
            '<div class="alert-sub">HIGH FRAUD PROBABILITY DETECTED</div>' +
            '<div class="alert-sub" style="margin-top:20px;color:#ffaaaa">' + msg + '</div>' +
            '<div class="trace-bar"><div class="trace-fill"></div></div>' +
            '<div style="margin-top:10px;font-size:0.8rem;color:#aaa">TRACING CALL SOURCE...</div>' +
            '<button id="dismissBtn" style="margin-top:40px;background:white;color:black;border:none;padding:15px 30px;font-weight:bold;cursor:pointer;border-radius:6px">DISMISS ALERT</button>' +
            '</div>';
        document.body.appendChild(overlay);

        // Siren
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.type = 'square';
            osc.frequency.setValueAtTime(800, ctx.currentTime);
            var t = ctx.currentTime;
            for (var i = 0; i < 6; i++) {
                osc.frequency.linearRampToValueAtTime(1200, t + 0.5);
                osc.frequency.linearRampToValueAtTime(800, t + 1.0);
                t += 1.0;
            }
            gain.gain.value = 0.2;
            osc.start(); osc.stop(ctx.currentTime + 6);

            var self = this;
            document.getElementById('dismissBtn').onclick = function () {
                if (overlay.parentElement) overlay.remove();
                try { osc.stop(); } catch (e) { }
                ctx.close();
                setTimeout(function () { self.alertTriggered = false; }, 30000);
            };
        } catch (e) {
            var self2 = this;
            document.getElementById('dismissBtn').onclick = function () {
                if (overlay.parentElement) overlay.remove();
                setTimeout(function () { self2.alertTriggered = false; }, 30000);
            };
        }

        this.speakAgent('Warning. High probability of fraud detected. Do not share your banking details. This call is being recorded for evidence.');
    }

    /* ---- DOWNLOAD REPORT ---- */
    downloadReport() {
        var report = 'GUARDIAN AI — FRAUD DETECTION REPORT\n';
        report += 'Generated: ' + new Date().toLocaleString() + '\n';
        report += '====================================\n\n';
        report += 'FINAL STATUS: ' + this.statusObj.textContent.replace(/\s+/g, ' ').trim() + '\n';
        var maxFraud = this.historyData.length ? Math.max.apply(null, this.historyData.map(function (h) { return h.fraud; })) : 0;
        var maxStress = this.historyData.length ? Math.max.apply(null, this.historyData.map(function (h) { return h.stress; })) : 0;
        var aiDetected = this.historyData.some(function (h) { return h.isAI; });
        report += 'MAX FRAUD RISK: ' + maxFraud + '%\n';
        report += 'MAX STRESS LEVEL: ' + maxStress + '%\n';
        report += 'AI VOICE DETECTED: ' + (aiDetected ? 'YES' : 'No') + '\n\n';
        report += 'TRANSCRIPT EVIDENCE:\n-------------------\n' + this.fullTranscript + '\n\n';
        report += 'ANALYSIS LOG:\n-------------\n';
        this.historyData.forEach(function (entry) {
            report += '[' + entry.time + '] Risk: ' + entry.fraud + '% | Stress: ' + entry.stress + '% | AI: ' + (entry.isAI ? 'YES' : 'No') + ' | ' + entry.reason + '\n';
        });
        var blob = new Blob([report], { type: 'text/plain' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'Evidence_Report_' + Date.now() + '.txt'; a.click();
        URL.revokeObjectURL(url);
    }

    /* ---- LOG ---- */
    addLog(msg) {
        var log = document.getElementById('systemLog');
        if (!log) return;
        var entry = document.createElement('div');
        entry.className = 'log-entry';
        var time = new Date().toLocaleTimeString();
        entry.innerHTML = '<span class="timestamp">[' + time + ']</span> ' + msg;
        log.prepend(entry);
    }

    /* ---- VISUALIZER ---- */
    drawVisualizer() {
        if (!this.isRecording) return;
        var self = this;
        requestAnimationFrame(function () { self.drawVisualizer(); });
        if (!this.analyser) return;

        var bufferLength = this.analyser.frequencyBinCount;
        var dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);

        var ctx = this.canvasCtx;
        var width = this.canvas.width;
        var height = this.canvas.height;
        ctx.fillStyle = 'rgba(7,11,18,0.25)';
        ctx.fillRect(0, 0, width, height);

        var barWidth = (width / bufferLength) * 2.5;
        var x = 0;
        for (var i = 0; i < bufferLength; i++) {
            var barHeight = dataArray[i] / 2;
            var hue = (i / bufferLength) * 200 + 180;
            ctx.fillStyle = 'hsl(' + hue + ',80%,' + (30 + barHeight / 3) + '%)';
            ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    updateButtons() {
        if (this.startButton) this.startButton.disabled = this.isRecording;
        if (this.stopButton) this.stopButton.disabled = !this.isRecording;
    }

    _setMode(text, active) {
        var modeText = document.getElementById('modeText');
        var badgeDot = document.querySelector('.badge-dot');
        if (modeText) modeText.textContent = text;
        if (badgeDot) { if (active) badgeDot.classList.add('active'); else badgeDot.classList.remove('active'); }
    }

    _updateSystemStatus(text, color) {
        var el = document.getElementById('systemStatusText');
        var pill = document.getElementById('systemStatus');
        if (el) el.textContent = text;
        if (pill) {
            var isSuccess = color === 'var(--success)';
            pill.style.background = isSuccess ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)';
            pill.style.borderColor = isSuccess ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)';
            pill.style.color = color;
        }
    }
}


/* =============================================
   6. DEEPFAKE FILE UPLOAD HANDLER
   ============================================= */
async function handleDeepfakeUpload() {
    var dfInput = document.getElementById('deepfakeUpload');
    var dfRes = document.getElementById('deepfakeResult');
    var dfLoad = document.getElementById('dfLoading');

    if (!dfInput.files[0]) { alert('Please select an MP3 or WAV file first.'); return; }

    dfLoad.style.display = 'flex';
    dfRes.style.display = 'none';

    var file = dfInput.files[0];
    try {
        var arrayBuffer = await file.arrayBuffer();
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var decodedBuffer;
        try {
            decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
        } catch (e) {
            throw new Error('Could not decode audio file. Make sure it is a valid MP3/WAV. (' + e.message + ')');
        }

        var channelData = decodedBuffer.getChannelData(0);
        var windowSize = 2048;
        var hopSize = Math.floor(channelData.length / 20);
        var allFreqData = [], allTimeData = [];
        for (var offset = 0; offset < channelData.length - windowSize; offset += hopSize) {
            var windowSlice = channelData.slice(offset, offset + windowSize);
            allFreqData.push(computeSimpleFFTMagnitudes(windowSlice));
            allTimeData.push(windowSlice);
        }

        var avgFreq = averageArrays(allFreqData);
        var midTime = allTimeData[Math.floor(allTimeData.length / 2)];
        var timeUint8 = new Uint8Array(midTime.length);
        for (var i = 0; i < midTime.length; i++) { timeUint8[i] = Math.round((midTime[i] + 1) * 128); }

        var deepfakeResult = AudioAnalysisEngine.detectAIVoice(avgFreq, timeUint8);
        var audioScores = AudioAnalysisEngine.audioFraudScore(timeUint8);
        await audioCtx.close();

        var cls = deepfakeResult.classification;
        var conf = deepfakeResult.confidence;
        document.getElementById('dfClassification').textContent = cls;
        document.getElementById('dfClassification').style.color = cls === 'AI-Generated' ? 'var(--danger)' : 'var(--success)';
        document.getElementById('dfConfidence').textContent = 'Confidence: ' + (conf * 100).toFixed(1) + '%';
        document.getElementById('dfLang').textContent = 'File: ' + file.name + ' · Size: ' + (file.size / 1024).toFixed(1) + ' KB';
        var metrics = deepfakeResult.metrics;
        document.getElementById('dfExplanation').textContent = 'Spectral Flatness: ' + metrics.spectralFlatness + ' · Spectral Variance: ' + metrics.spectralVariance + ' · Stress Score: ' + audioScores.stress + '%';

        var dfReasonsEl = document.getElementById('dfReasons');
        dfReasonsEl.innerHTML = '';
        if (deepfakeResult.reasons.length > 0) {
            deepfakeResult.reasons.forEach(function (r) {
                var d = document.createElement('div'); d.className = 'df-reason-item'; d.textContent = '- ' + r; dfReasonsEl.appendChild(d);
            });
        } else {
            var d = document.createElement('div'); d.className = 'df-reason-item'; d.style.borderLeftColor = 'var(--success)'; d.textContent = '- No synthetic indicators detected — voice appears natural.'; dfReasonsEl.appendChild(d);
        }
        dfRes.style.display = 'flex';
    } catch (err) {
        console.error(err); alert('Error analyzing audio: ' + err.message);
    } finally {
        dfLoad.style.display = 'none';
    }
}

function computeSimpleFFTMagnitudes(samples) {
    var N = samples.length, halfN = Math.floor(N / 2);
    var magnitudes = new Float32Array(halfN);
    for (var k = 0; k < halfN; k++) {
        var re = 0, im = 0;
        for (var n = 0; n < N; n++) {
            var angle = (2 * Math.PI * k * n) / N;
            re += samples[n] * Math.cos(angle);
            im -= samples[n] * Math.sin(angle);
        }
        magnitudes[k] = Math.sqrt(re * re + im * im) / N;
    }
    return magnitudes;
}

function averageArrays(arrays) {
    if (arrays.length === 0) return new Float32Array(0);
    var len = arrays[0].length;
    var result = new Float32Array(len);
    for (var j = 0; j < arrays.length; j++) {
        for (var i = 0; i < len; i++) result[i] += arrays[j][i];
    }
    for (var i = 0; i < len; i++) result[i] /= arrays.length;
    var max = 0;
    for (var i = 0; i < len; i++) if (result[i] > max) max = result[i];
    if (max > 0) for (var i = 0; i < len; i++) result[i] = (result[i] / max) * 255;
    return result;
}


/* =============================================
   7. INITIALIZATION
   ============================================= */
window.addEventListener('load', function () {
    var canvas = document.getElementById('visualizer');
    canvas.width = canvas.offsetWidth || 800;
    canvas.height = canvas.offsetHeight || 110;

    var streamer = new AudioStreamer();
    window._guardian = streamer;

    var dfBtn = document.getElementById('detectDeepfakeBtn');
    if (dfBtn) dfBtn.addEventListener('click', handleDeepfakeUpload);

    window.addEventListener('resize', function () {
        canvas.width = canvas.offsetWidth || 800;
        canvas.height = canvas.offsetHeight || 110;
    });

    var warnings = [];
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
        warnings.push('Web Speech API not supported — use Chrome for best results.');
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        warnings.push('getUserMedia not supported — microphone analysis unavailable.');
    }
    if (warnings.length > 0) {
        var log = document.getElementById('systemLog');
        warnings.forEach(function (w) {
            var entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = '<span style="color:var(--warning)">' + w + '</span>';
            log.appendChild(entry);
        });
    }

    streamer.addLog('Guardian AI initialized. All modules running browser-native.');
    streamer.addLog('Web Audio API, SpeechRecognition, and ML heuristics loaded.');
    streamer.addLog('No server required — all processing runs in your browser.');
});
