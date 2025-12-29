
// State
let currentResults = []; // { class, count, detections: [{box, score}] }
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

// Listeners
// dropArea logic handled by input inside it
fileInput.addEventListener('change', (e) => handleUpload(e.target.files[0]));

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
