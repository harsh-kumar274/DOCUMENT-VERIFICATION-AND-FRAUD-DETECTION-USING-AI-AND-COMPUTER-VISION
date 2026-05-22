/* Antigravity PAN-Verify Frontend JS Controller
   Orchestrates Drag-and-Drop, API communication, and dynamic layout rendering.
*/

document.addEventListener("DOMContentLoaded", () => {
    // API config
    const API_URL = window.location.origin;
    
    // Cache UI elements
    const serverStatus = document.getElementById("serverStatus");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const uploadPreview = document.getElementById("uploadPreview");
    const uploadFilename = document.getElementById("uploadFilename");
    const removeUploadBtn = document.getElementById("removeUploadBtn");
    const verifyBtn = document.getElementById("verifyBtn");
    
    const idleState = document.getElementById("idleState");
    const processingState = document.getElementById("processingState");
    const pipelineStageText = document.getElementById("pipelineStageText");
    const progressBarFill = document.getElementById("progressBarFill");
    const resultsDashboard = document.getElementById("resultsDashboard");
    
    const decisionBadge = document.getElementById("decisionBadge");
    const gaugeFillCircle = document.getElementById("gaugeFillCircle");
    const gaugeNumber = document.getElementById("gaugeNumber");
    const decisionReasoningText = document.getElementById("decisionReasoningText");
    const scorecardGrid = document.getElementById("scorecardGrid");
    
    const explorerSection = document.getElementById("explorerSection");
    const stepperNodes = document.querySelectorAll(".step-node");
    const stepPanes = document.querySelectorAll(".step-content-pane");
    
    // State variables
    let selectedFile = null;
    let pipelineData = null; // Store complete response report from API
    
    // --- 1. Check API Server Status ---
    async function checkApiHealth() {
        try {
            const res = await fetch(`${API_URL}/health`);
            const data = await res.json();
            if (data.status === "ok") {
                serverStatus.innerHTML = `
                    <span class="status-dot success"></span>
                    <span class="status-text">Backend API Connected</span>
                `;
            } else {
                throw new Error("Invalid response");
            }
        } catch (e) {
            serverStatus.innerHTML = `
                <span class="status-dot danger"></span>
                <span class="status-text">API Disconnected</span>
            `;
            console.error("Health check failure:", e);
        }
    }
    checkApiHealth();
    // Poll health every 15 seconds
    setInterval(checkApiHealth, 15000);
    
    // --- 2. Uploader Drag & Drop Events ---
    dropzone.addEventListener("click", () => fileInput.click());
    
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--primary)";
        dropzone.style.background = "rgba(99, 102, 241, 0.08)";
    });
    
    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(99, 102, 241, 0.3)";
        dropzone.style.background = "rgba(99, 102, 241, 0.02)";
    });
    
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(99, 102, 241, 0.3)";
        dropzone.style.background = "rgba(99, 102, 241, 0.02)";
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
    
    function handleFileSelection(file) {
        // Enforce basic type checking
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (JPG, PNG, WebP).");
            return;
        }
        
        selectedFile = file;
        uploadFilename.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            dropzone.classList.add("hidden");
            previewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }
    
    removeUploadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        dropzone.classList.remove("hidden");
        previewContainer.classList.add("hidden");
        resetDashboard();
    });
    
    function resetDashboard() {
        idleState.classList.remove("hidden");
        processingState.classList.add("hidden");
        resultsDashboard.classList.add("hidden");
        explorerSection.classList.add("hidden");
        pipelineData = null;
    }
    
    // --- 3. Execute Verification API Request ---
    verifyBtn.addEventListener("click", async () => {
        if (!selectedFile) return;
        
        resetDashboard();
        idleState.classList.add("hidden");
        processingState.classList.remove("hidden");
        
        // Progress steps simulator
        const stages = [
            { text: "Initializing pipeline & normalizing dimensions...", progress: "10%" },
            { text: "Running Phase 1 image enhancements & CLAHE...", progress: "25%" },
            { text: "Applying FFT notch filters to periodic scan noise...", progress: "35%" },
            { text: "Detecting card boundary via Canny edges & contour polygon...", progress: "50%" },
            { text: "Warping card perspective coordinates via RANSAC...", progress: "65%" },
            { text: "Extracting ORB keypoints & template alignment...", progress: "78%" },
            { text: "Analyzing texture variances & copy-move tampering...", progress: "88%" },
            { text: "Parsing demographic metadata crops via OCR...", progress: "95%" },
            { text: "Compiling decision bands and weights...", progress: "100%" }
        ];
        
        let stageIdx = 0;
        const progressInterval = setInterval(() => {
            if (stageIdx < stages.length - 1) {
                pipelineStageText.textContent = stages[stageIdx].text;
                progressBarFill.style.width = stages[stageIdx].progress;
                stageIdx++;
            }
        }, 450);
        
        // Build FormData
        const formData = new FormData();
        formData.append("file", selectedFile);
        
        try {
            const res = await fetch(`${API_URL}/api/verify`, {
                method: "POST",
                body: formData
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Server pipeline error");
            }
            
            pipelineData = await res.json();
            
            // Finish loader
            clearInterval(progressInterval);
            pipelineStageText.textContent = "Report ready!";
            progressBarFill.style.width = "100%";
            
            setTimeout(() => {
                processingState.classList.add("hidden");
                renderReportResults(pipelineData);
            }, 500);
            
        } catch (e) {
            clearInterval(progressInterval);
            processingState.classList.add("hidden");
            idleState.classList.remove("hidden");
            alert(`Pipeline Failed: ${e.message}`);
            console.error("Pipeline failure:", e);
        }
    });
    
    // --- 4. Render Results report on panels ---
    function renderReportResults(data) {
        resultsDashboard.classList.remove("hidden");
        explorerSection.classList.remove("hidden");
        
        const decision = data.decision_engine;
        const prob = decision.fraud_probability;
        const probPct = Math.round(prob * 100);
        
        // Update threat Badge
        decisionBadge.textContent = decision.decision;
        decisionBadge.className = "badge"; // Reset
        
        if (decision.decision === "LIKELY GENUINE") {
            decisionBadge.classList.add("badge-success");
        } else if (decision.decision === "NEEDS MANUAL REVIEW") {
            decisionBadge.classList.add("badge-warning");
        } else {
            decisionBadge.classList.add("badge-danger");
        }
        
        // Animate circular gauge
        // SVG circle radius=40, circumference = 2 * pi * r = 251.2
        const circumference = 251.2;
        const offset = circumference - (prob * circumference);
        gaugeFillCircle.style.strokeDashoffset = offset;
        gaugeFillCircle.style.stroke = decision.color;
        gaugeNumber.textContent = `${probPct}%`;
        
        // reasoning text
        decisionReasoningText.textContent = decision.explanation;
        
        // Populate Scorecard
        scorecardGrid.innerHTML = "";
        decision.scorecard.forEach(item => {
            const card = document.createElement("div");
            card.className = "scorecard-item";
            
            const threatClass = item.threat === "High" ? "threat-high" : 
                                item.threat === "Medium" ? "threat-medium" : "threat-low";
                                
            card.innerHTML = `
                <div class="metric-meta">
                    <span class="metric-name">${item.name}</span>
                    <span class="metric-desc">${item.description}</span>
                </div>
                <div class="metric-threat-wrap">
                    <div class="metric-score-bar" title="Threat Score: ${item.score}">
                        <div class="metric-score-fill" style="width: ${item.score * 100}%; background-color: ${decision.color}"></div>
                    </div>
                    <span class="metric-threat-badge ${threatClass}">${item.threat}</span>
                </div>
            `;
            scorecardGrid.appendChild(card);
        });
        
        // Setup Explorer data
        setupExplorerData(data);
    }
    
    // --- 5. Stepper explorer configuration ---
    stepperNodes.forEach(node => {
        node.addEventListener("click", () => {
            stepperNodes.forEach(n => n.classList.remove("active"));
            stepPanes.forEach(p => p.classList.remove("active"));
            
            node.classList.add("active");
            const stepName = node.getAttribute("data-step");
            document.getElementById(`stepPane-${stepName}`).classList.add("active");
        });
    });
    
    function setupExplorerData(data) {
        // --- Tab 1: Preprocessing Viewer ---
        const prep = data.preprocessing;
        const prepMetrics = document.getElementById("preprocessingMetrics");
        prepMetrics.innerHTML = `
            <div class="readout-line"><span class="lbl">Width:</span><span class="val">${prep.metrics.width} px</span></div>
            <div class="readout-line"><span class="lbl">Height:</span><span class="val">${prep.metrics.height} px</span></div>
            <div class="readout-line"><span class="lbl">Brightness:</span><span class="val">${prep.metrics.brightness} / 255</span></div>
            <div class="readout-line"><span class="lbl">Contrast (StdDev):</span><span class="val">${prep.metrics.contrast}</span></div>
            <div class="readout-line"><span class="lbl">Laplacian Blur Score:</span><span class="val">${prep.metrics.blur_score}</span></div>
            <div class="readout-line"><span class="lbl">Noise Estimate:</span><span class="val">${prep.metrics.noise_estimate} σ</span></div>
        `;
        
        // Build interactive filter stages
        const togglesContainer = document.getElementById("filterTogglesContainer");
        togglesContainer.innerHTML = "";
        
        const viewerImage = document.getElementById("preprocessingViewerImage");
        const viewerTag = document.getElementById("preprocessingViewerTag");
        
        prep.steps.forEach((step, idx) => {
            const chip = document.createElement("div");
            chip.className = `filter-chip ${idx === 0 ? 'active' : ''}`;
            chip.textContent = step.name.toUpperCase().replace("_", " ");
            
            chip.addEventListener("click", () => {
                document.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                viewerImage.src = step.image_base64;
                viewerTag.textContent = step.description;
            });
            
            togglesContainer.appendChild(chip);
        });
        
        // Initialize viewer to default original step image
        viewerImage.src = prep.steps[0].image_base64;
        viewerTag.textContent = prep.steps[0].description;
        
        // --- Tab 2: Document Detection Viewer ---
        const det = data.document_detection;
        document.getElementById("detectionViewerImage").src = det.preview_base64;
        document.getElementById("detMethodUsed").textContent = det.method_used;
        document.getElementById("detRiskVal").textContent = det.detection_risk > 0 ? "Elevated Risk" : "Normal";
        document.getElementById("detExplanationText").textContent = det.explanation;
        
        // --- Tab 3: Perspective Correction Viewer ---
        const corr = data.perspective_correction;
        document.getElementById("correctionViewerImage").src = corr.corrected_base64;
        document.getElementById("correctionExplanationText").textContent = corr.explanation;
        
        // --- Tab 4: Feature Matching Viewer ---
        const feat = data.feature_matching;
        const matchingViewer = document.getElementById("matchingViewerImage");
        const matchingAlert = document.getElementById("orbAlertContainer");
        
        if (feat.success && feat.matches_base64) {
            matchingViewer.src = feat.matches_base64;
            matchingViewer.classList.remove("hidden");
            document.getElementById("orbMatchesVal").textContent = feat.num_matches;
            document.getElementById("orbCoherenceVal").textContent = `${Math.round(feat.ransac_inlier_ratio * 100)}%`;
            document.getElementById("matchingExplanationText").textContent = feat.explanation;
            matchingAlert.className = "info-alert";
        } else {
            // Show fallback aligned image and alert warning
            matchingViewer.src = corr.corrected_base64;
            document.getElementById("orbMatchesVal").textContent = "0";
            document.getElementById("orbCoherenceVal").textContent = "0%";
            document.getElementById("matchingExplanationText").textContent = "CRITICAL WARNING: Spatial alignment is highly deficient. Uploaded layout fails template ORB keypoint validation.";
            matchingAlert.className = "info-alert alert-threat-active";
        }
        
        // --- Tab 5: Tampering Heatmap Viewer ---
        const tamp = data.tampering_detection;
        const tamperAlert = document.getElementById("tamperAlertContainer");
        
        document.getElementById("tamperingViewerImage").src = tamp.heatmap_base64;
        document.getElementById("tamperCopyMoveVal").textContent = tamp.copy_move_score > 0 ? "Detected" : "None";
        document.getElementById("tamperTextureVal").textContent = `${Math.round(tamp.texture_edge_score * 100)}%`;
        document.getElementById("tamperingExplanationText").textContent = tamp.explanation;
        
        if (tamp.tampering_score > 0.4) {
            tamperAlert.className = "info-alert alert-threat-active";
        } else {
            tamperAlert.className = "info-alert";
        }
        
        // --- Tab 6: OCR & Validations Panel ---
        const ocr = data.ocr_extraction;
        const val = data.pan_validation;
        
        // Populate OCR Fields list
        const ocrFieldsContainer = document.getElementById("ocrFieldsContainer");
        ocrFieldsContainer.innerHTML = "";
        
        Object.entries(ocr.fields).forEach(([field, item]) => {
            const card = document.createElement("div");
            card.className = "ocr-card";
            
            const isMock = item.is_mock ? `<span class="ocr-mock-tag">Mock Mode</span>` : "";
            
            card.innerHTML = `
                <div>
                    <span class="ocr-field-lbl">${field}</span>
                    <h4 class="ocr-field-val">${item.text || "Empty / Unparsed"}</h4>
                </div>
                <div class="ocr-meta">
                    <span class="ocr-conf">${Math.round(item.confidence)}% Conf</span>
                    ${isMock}
                </div>
            `;
            ocrFieldsContainer.appendChild(card);
        });
        
        // Populate Validation Checks list
        const validationRulesList = document.getElementById("validationRulesList");
        validationRulesList.innerHTML = "";
        
        val.checks.forEach(check => {
            const item = document.createElement("div");
            
            let itemClass = "rule-pass";
            let iconClass = "fa-check";
            
            if (check.status === "FAIL") {
                itemClass = "rule-fail";
                iconClass = "fa-xmark";
            } else if (check.status === "WARNING") {
                itemClass = "rule-warning";
                iconClass = "fa-triangle-exclamation";
            }
            
            item.className = `rule-check-item ${itemClass}`;
            item.innerHTML = `
                <div class="rule-status-icon">
                    <i class="fa-solid ${iconClass}"></i>
                </div>
                <div class="rule-content">
                    <h4>${check.check}</h4>
                    <p>${check.message}</p>
                </div>
            `;
            validationRulesList.appendChild(item);
        });
    }
});
