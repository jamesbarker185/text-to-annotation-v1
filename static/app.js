
// State
let currentResults = []; // { class, count, detections: [{box, score}] }
let currentTextRegions = []; // DBNet results
let extractedTexts = []; // OCR results {box, text, confidence}
let currentThresholds = {}; // { class: 0.5 }
let currentFile = null; // Store for rerunning
let imageDims = { w: 0, h: 0 };
let colorPalette = ['#5865F2', '#EB459E', '#F2A900', '#3BA55C', '#ED4245', '#9B59B6'];

// Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const promptInput = document.getElementById('prompt-input');
const btnUpdatePrompts = document.getElementById('btn-update-prompts');
// Add new button for re-uploading
const btnNewUpload = document.createElement('button');
btnNewUpload.className = 'btn btn-sm btn-secondary';
btnNewUpload.innerText = 'New Image';
btnNewUpload.style.marginLeft = '10px';
btnNewUpload.onclick = () => {
    document.querySelector('.upload-card').classList.remove('hidden');
    currentFile = null;
    currentResults = []; // clear?
    // Optionally hide viewer?
    // viewerSection.classList.add('hidden');
};
// Add to input group
document.querySelector('.input-group').appendChild(btnNewUpload);

const sourceImg = document.getElementById('source-image');
const overlayCanvas = document.getElementById('overlay-canvas');
const ctx = overlayCanvas.getContext('2d');
const viewerSection = document.getElementById('viewer-section');
const slidersContainer = document.getElementById('sliders-container');
const loadingOverlay = document.getElementById('loading-overlay');
const initSection = document.querySelector('.control-panel');
const batchSection = document.getElementById('batch-section');
const btnRunBatch = document.getElementById('btn-run-batch');
const btnBack = document.getElementById('btn-back');
const batchInput = document.getElementById('batch-input');
const batchDropArea = document.getElementById('batch-drop-area');

const btnExtractText = document.getElementById('btn-extract-text');
const ocrModelSelect = document.getElementById('ocr-model-select');

// Listeners
// dropArea logic handled by input inside it
fileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));

btnExtractText.addEventListener('click', runOCR);

btnUpdatePrompts.addEventListener('click', () => {
    if (currentFile) handleUpload(currentFile);
    else alert("Please upload an image first.");
});

promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        if (currentFile) handleUpload(currentFile);
    }
});

btnRunBatch.addEventListener('click', showBatchUI);
btnBack.addEventListener('click', () => {
    batchSection.classList.add('hidden');
    viewerSection.classList.remove('hidden');
});

batchDropArea.addEventListener('click', () => batchInput.click());
batchInput.addEventListener('change', (e) => runBatch(e.target.files));

// --- Core Workflow ---

async function handleUpload(file) {
    if (!file) return;
    currentFile = file;

    // User Feedback
    loadingOverlay.classList.remove('hidden');
    viewerSection.classList.remove('hidden'); // Show viewer structure early
    initSection.style.opacity = '0.5';
    initSection.style.pointerEvents = 'none';

    // Show Image
    const url = URL.createObjectURL(file);
    sourceImg.src = url;
    sourceImg.onload = () => {
        // Resize canvas to match image display size (which might be scaled by CSS)
        // Wait, best is to match intrinsics.
        // Actually, for drawing, canvas width/height should match image natural width/height
        // Then styling scales both down.
        imageDims.w = sourceImg.naturalWidth;
        imageDims.h = sourceImg.naturalHeight;
        overlayCanvas.width = imageDims.w; // High res canvas
        overlayCanvas.height = imageDims.h;

        // CSS styling ensures they align visually
        overlayCanvas.style.width = sourceImg.clientWidth + 'px';
        overlayCanvas.style.height = sourceImg.clientHeight + 'px';
    };

    // Prepare API
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompts', promptInput.value);

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();


        if (data.status === 'success') {
            currentResults = data.results;
            currentTextRegions = data.text_regions || []; // Default to empty if missing

            // Performance Logging
            if (data.timings) {
                // Console
                console.group("🚀 Detection Performance");
                console.log(`SAM3 Inference:   ${data.timings.sam3.toFixed(4)}s`);
                console.log(`DBNet Detection:  ${data.timings.dbnet.toFixed(4)}s`);
                console.log(`Total Time:       ${data.timings.total.toFixed(4)}s`);
                console.groupEnd();

                // UI
                const tSam = document.getElementById('time-sam3');
                const tDb = document.getElementById('time-dbnet');
                const tTot = document.getElementById('time-total');
                if (tSam) tSam.innerText = data.timings.sam3.toFixed(3) + 's';
                if (tDb) tDb.innerText = data.timings.dbnet.toFixed(3) + 's';
                if (tTot) tTot.innerText = data.timings.total.toFixed(3) + 's';
            }

            // Initialize thresholds map if empty
            data.results.forEach(g => {
                if (!(g.class in currentThresholds)) {
                    currentThresholds[g.class] = 0.50; // default 50%
                }
            });

            renderSliders();
            drawDetections();

            // UI Transition
            // Don't hide the whole section, just the upload card to save space?
            // User request: "rerun inference and edit text prompt"
            // So we MUST keep the prompt input visible.
            // Let's just hide the upload card part, or keep both? 
            // If we keep both, it pushes content down.
            // Let's hide the upload card but keep the prompt card.

            document.querySelector('.upload-card').classList.add('hidden');
            // initSection.classList.add('hidden'); // REMOVED THIS LINE
        } else {
            alert('Error: ' + JSON.stringify(data));
        }
    } catch (e) {
        console.error(e);
        alert('Failed to connect to backend.');
    } finally {
        loadingOverlay.classList.add('hidden');
        initSection.style.opacity = '1';
        initSection.style.pointerEvents = 'auto';
    }
}

// --- Visualization ---

function drawDetections() {
    // Clear canvas
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    let total = 0;

    currentResults.forEach((group, idx) => {
        const color = colorPalette[idx % colorPalette.length];
        const threshold = currentThresholds[group.class] || 0.5;

        // Count for this class
        let count = 0;

        group.detections.forEach(det => {
            if (det.score >= threshold) {
                // Draw Box
                const [x1, y1, x2, y2] = det.box;
                const w = x2 - x1;
                const h = y2 - y1;

                // Box stroke
                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(2, imageDims.w / 300); // Scale line width
                ctx.strokeRect(x1, y1, w, h);

                // Fill (slight tint)
                ctx.fillStyle = color + '33'; // 20% opacity hex
                ctx.fillRect(x1, y1, w, h);

                count++;
                total++;
            }
        });

        // Update badge in slider
        const badge = document.getElementById(`count-${group.class}`);
        if (badge) badge.innerText = count;
    });

    document.getElementById('total-count-display').innerText = total;

    // Draw Text Regions (DBNet)
    // Draw Text Regions (DBNet + OCR)
    // If we have extracted text, use that. Otherwise use raw regions.
    const textData = (extractedTexts && extractedTexts.length > 0) ? extractedTexts : currentTextRegions;

    if (textData && textData.length > 0) {
        const textColor = '#00FF00'; // Bright Green
        
        textData.forEach(region => {
            const [x1, y1, x2, y2] = region.box;
            const w = x2 - x1;
            const h = y2 - y1;
            
            ctx.strokeStyle = textColor;
            ctx.lineWidth = Math.max(2, imageDims.w / 300);
            ctx.strokeRect(x1, y1, w, h);
            
            // Optional: slight fill
            ctx.fillStyle = textColor + '22';
            ctx.fillRect(x1, y1, w, h);

            // Draw Text Label if valid
            if (region.text) {
                ctx.fillStyle = textColor;
                ctx.font = `bold ${Math.max(12, h/2)}px Arial`;
                ctx.fillText(region.text, x1, y1 - 5);
            }
        });
    }
}

// --- Controls ---

function renderSliders() {
    slidersContainer.innerHTML = '';

    currentResults.forEach((group, idx) => {
        const color = colorPalette[idx % colorPalette.length];
        const cls = group.class;
        const threshold = currentThresholds[cls] || 0.5;

        // Inverted Logic for "Sensitivity"
        // Threshold 0.9 (Strict) -> Slider 10% (Low Sensitivity)
        // Threshold 0.1 (Loose)  -> Slider 90% (High Sensitivity)
        // SliderVal = (1 - threshold) * 100
        const sliderVal = Math.round((1 - threshold) * 100);

        const div = document.createElement('div');
        div.className = 'slider-item';
        div.innerHTML = `
            <div class="slider-header">
                <span style="color:${color}; font-weight:600">${cls}</span>
                <div>
                    <span class="label-muted">Sensitivity: </span>
                    <span id="val-${cls}">${sliderVal}%</span>
                    <span class="badge-count" id="count-${cls}">0</span>
                </div>
            </div>
            <input type="range" min="0" max="100" value="${sliderVal}" 
                   oninput="updateThreshold('${cls}', this.value)">
        `;
        slidersContainer.appendChild(div);
    });
}

window.updateThreshold = (cls, val) => {
    // Inverted Logic
    // Val 90 -> Threshold 0.1
    // Threshold = 1 - (Val / 100)
    const sensitivity = parseInt(val);
    const floatVal = 1.0 - (sensitivity / 100.0);

    // Clamp slightly to avoid 0.0 or 1.0 if needed, but simple is fine
    currentThresholds[cls] = floatVal;

    document.getElementById(`val-${cls}`).innerText = sensitivity + '%';
    drawDetections(); // fast client-side redraw
};

// --- Batch Processing ---

function showBatchUI() {
    viewerSection.classList.add('hidden');
    batchSection.classList.remove('hidden');
    document.getElementById('batch-results').innerHTML = '';
}

async function runBatch(files) {
    if (!files || files.length === 0) return;

    const fd = new FormData();
    for (let f of files) {
        fd.append('files', f);
    }

    fd.append('prompts', promptInput.value);
    fd.append('thresholds', JSON.stringify(currentThresholds));

    // Show simple loading state in grid
    const grid = document.getElementById('batch-results');
    grid.innerHTML = '<div style="grid-column: 1/-1; text-align:center; padding:20px;">Processing ' + files.length + ' images...</div>';

    try {
        const res = await fetch('/api/batch-detect', {
            method: 'POST',
            body: fd
        });
        const data = await res.json();

        if (data.status === 'success') {
            displayBatchResults(data.batch_summary);
        } else {
            grid.innerHTML = 'Error: ' + JSON.stringify(data);
        }
    } catch (e) {
        grid.innerHTML = 'Error connecting to server.';
    }
}

function displayBatchResults(summary) {
    const grid = document.getElementById('batch-results');
    grid.innerHTML = '';

    summary.forEach(item => {
        const div = document.createElement('div');
        div.className = 'batch-item';

        // Build stats string
        let statsHtml = '';
        for (let [cls, count] of Object.entries(item.counts)) {
            statsHtml += `<div class="stats-row"><span>${cls}</span> <span>${count}</span></div>`;
        }

        div.innerHTML = `
            <div class="batch-item-header" title="${item.filename}">${item.filename}</div>
            ${statsHtml}
        `;
        grid.appendChild(div);
    });
}

async function runOCR() {
    if (!currentFile || !currentTextRegions.length) {
        alert("No text regions detected to extract from.");
        return;
    }

    const btn = document.getElementById('btn-extract-text');
    const statusDiv = document.getElementById('ocr-status');
    const listDiv = document.getElementById('ocr-results-list');
    
    btn.disabled = true;
    btn.innerText = "Extracting...";
    statusDiv.innerText = "Running OCR...";
    listDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('regions', JSON.stringify(currentTextRegions));
    formData.append('model', document.getElementById('ocr-model-select').value);

    try {
        const response = await fetch('/api/extract-text', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
            extractedTexts = data.extracted_text;
            
            // Stats
            if (data.perf_stats) {
                const s = data.perf_stats;
                console.log(`OCR Layer: Preprocessing=${s.preprocess.toFixed(4)}s, Inference=${s.inference.toFixed(4)}s`);
                statusDiv.innerText = `Extracted in ${(s.preprocess + s.inference).toFixed(2)}s (${extractedTexts.length} regions)`;
            } else {
                statusDiv.innerText = `Extracted text from ${extractedTexts.length} regions.`;
            }

            // Render list
            extractedTexts.forEach(item => {
                const div = document.createElement('div');
                div.style.padding = "4px 0";
                div.style.borderBottom = "1px solid #333";
                div.innerHTML = `<span style="color:#00FF00; font-weight:bold;">${item.text}</span> <span style="color:#888; font-size:0.75em;">(${Math.round(item.confidence*100)}%)</span>`;
                listDiv.appendChild(div);
            });

            drawDetections();
        } else {
            statusDiv.innerText = "Error: " + JSON.stringify(data);
        }
    } catch (e) {
        console.error(e);
        statusDiv.innerText = "Failed to run OCR.";
    } finally {
        btn.disabled = false;
        btn.innerText = "Extract Text";
    }
}

