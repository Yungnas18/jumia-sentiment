document.addEventListener("DOMContentLoaded", () => {
    const reviewInput = document.getElementById("review-input");
    const modelSelect = document.getElementById("model-select");
    const predictBtn = document.getElementById("predict-btn");
    const resultPanel = document.getElementById("result-panel");
    const resultTag = document.getElementById("result-tag");
    const resultModel = document.getElementById("result-model");
    const confidenceValue = document.getElementById("confidence-value");
    const confidenceFill = document.getElementById("confidence-fill");
    const errorMsg = document.getElementById("error-msg");

    predictBtn.addEventListener("click", async () => {
        const text = reviewInput.value.trim();
        errorMsg.style.display = "none";

        if (!text) {
            errorMsg.textContent = "Please enter a review before classifying.";
            errorMsg.style.display = "block";
            return;
        }

        predictBtn.disabled = true;
        predictBtn.textContent = "Classifying...";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    review_text: text,
                    model: modelSelect.value,
                }),
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();
            renderResult(data);
        } catch (err) {
            errorMsg.textContent = "Something went wrong reaching the prediction service. Please try again.";
            errorMsg.style.display = "block";
            resultPanel.style.display = "none";
            console.error(err);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Classify Sentiment";
        }
    });

    function renderResult(data) {
        const isPositive = data.sentiment === "Positive";

        resultTag.textContent = data.sentiment;
        resultTag.className = "result-tag " + (isPositive ? "positive" : "negative");
        resultModel.textContent = `via ${data.model_used}`;

        confidenceValue.textContent = `${data.confidence}%`;
        confidenceFill.className = "meter-fill " + (isPositive ? "positive" : "negative");

        resultPanel.style.display = "block";
        // Trigger the width transition on the next frame
        requestAnimationFrame(() => {
            confidenceFill.style.width = `${data.confidence}%`;
        });
    }
});
