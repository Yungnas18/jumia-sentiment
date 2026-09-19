document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const dropzoneInner = document.getElementById("dropzone-inner");
    const fileInput = document.getElementById("file-input");
    const uploadBtn = document.getElementById("upload-btn");
    const modelSelect = document.getElementById("model-select-batch");
    const errorMsg = document.getElementById("batch-error-msg");
    const progressTrack = document.getElementById("progress-track");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");
    const resultPanel = document.getElementById("batch-result-panel");
    const statRow = document.getElementById("batch-stat-row");
    const chartImg = document.getElementById("batch-chart-img");
    const previewTbody = document.getElementById("preview-tbody");
    const downloadBtn = document.getElementById("download-btn");

    let selectedFile = null;

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            setFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            setFile(fileInput.files[0]);
        }
    });

    function setFile(file) {
        selectedFile = file;
        dropzoneInner.innerHTML = `
            <p class="dropzone-title">${file.name}</p>
            <p class="dropzone-sub">${(file.size / 1024).toFixed(1)} KB — click to change file</p>
        `;
        uploadBtn.disabled = false;
        errorMsg.style.display = "none";
    }

    uploadBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        errorMsg.style.display = "none";
        resultPanel.style.display = "none";
        uploadBtn.disabled = true;
        progressTrack.style.display = "block";
        progressLabel.style.display = "block";
        progressFill.style.width = "20%";

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("model", modelSelect.value);

        try {
            progressFill.style.width = "55%";

            const response = await fetch("/api/batch-predict", {
                method: "POST",
                body: formData,
            });

            progressFill.style.width = "90%";

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.detail || `Server responded with ${response.status}`);
            }

            const data = await response.json();
            progressFill.style.width = "100%";
            setTimeout(() => renderBatchResult(data), 200);
        } catch (err) {
            errorMsg.textContent = err.message || "Something went wrong processing this file.";
            errorMsg.style.display = "block";
            console.error(err);
        } finally {
            setTimeout(() => {
                progressTrack.style.display = "none";
                progressLabel.style.display = "none";
                progressFill.style.width = "0%";
                uploadBtn.disabled = false;
            }, 400);
        }
    });

    function renderBatchResult(data) {
        statRow.innerHTML = `
            <div class="stat">
                <div class="value">${data.total_reviews}</div>
                <div class="label">Reviews processed</div>
            </div>
            <div class="stat">
                <div class="value" style="color: var(--positive);">${data.positive_count}</div>
                <div class="label">Positive</div>
            </div>
            <div class="stat">
                <div class="value" style="color: var(--negative);">${data.negative_count}</div>
                <div class="label">Negative</div>
            </div>
        `;

        chartImg.src = data.chart_url + "?t=" + Date.now();

        previewTbody.innerHTML = "";
        data.preview.forEach((row) => {
            const tr = document.createElement("tr");
            const isPositive = row.predicted_sentiment === "Positive";
            const truncated = row.review_text.length > 80
                ? row.review_text.slice(0, 80) + "…"
                : row.review_text;
            tr.innerHTML = `
                <td>${escapeHtml(truncated)}</td>
                <td><span class="result-tag ${isPositive ? 'positive' : 'negative'}" style="font-size:0.8rem; padding:3px 10px;">${row.predicted_sentiment}</span></td>
                <td class="mono-num">${row.confidence_pct}%</td>
            `;
            previewTbody.appendChild(tr);
        });

        downloadBtn.href = data.download_url;
        resultPanel.style.display = "block";
        resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }
});
