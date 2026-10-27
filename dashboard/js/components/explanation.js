/**
 * CloudGuard-Water Dashboard: Model Explanation Component
 * Module: explanation.js
 * 
 * Renders the transparent Decision Tree inference path, node comparisons,
 * primary influencing features, estimated class probabilities, and safety notices.
 */

/**
 * Render the complete explainability view into the specified DOM elements.
 * @param {Object} exp - Response object from /explain endpoint
 */
export function renderExplanation(exp) {
  if (!exp) return;

  // 1. Classification & Probabilities
  const predBadge = document.getElementById("prediction-badge");
  const pSafeVal = document.getElementById("prob-safe-val");
  const pUnsafeVal = document.getElementById("prob-unsafe-val");
  const probBarSafe = document.getElementById("prob-bar-safe");
  const probBarUnsafe = document.getElementById("prob-bar-unsafe");
  const leafNodeEl = document.getElementById("leaf-node-badge");

  const isSafe = exp.model_classification === "Safe";
  if (predBadge) {
    predBadge.textContent = exp.model_classification.toUpperCase();
    predBadge.className = isSafe ? "badge badge-safe badge-lg" : "badge badge-unsafe badge-lg";
  }

  const pSafe = (exp.estimated_probabilities?.Safe ?? 0) * 100;
  const pUnsafe = (exp.estimated_probabilities?.Unsafe ?? 0) * 100;

  if (pSafeVal) pSafeVal.textContent = `${pSafe.toFixed(1)}%`;
  if (pUnsafeVal) pUnsafeVal.textContent = `${pUnsafe.toFixed(1)}%`;

  if (probBarSafe) probBarSafe.style.width = `${pSafe}%`;
  if (probBarUnsafe) probBarUnsafe.style.width = `${pUnsafe}%`;

  if (leafNodeEl) {
    leafNodeEl.textContent = `Leaf Node #${exp.leaf_node_id ?? "--"}`;
  }

  // 2. Primary Influencing Features Pills
  const primaryFeatContainer = document.getElementById("primary-features-container");
  if (primaryFeatContainer) {
    if (exp.primary_influencing_features && exp.primary_influencing_features.length > 0) {
      primaryFeatContainer.innerHTML = exp.primary_influencing_features.map(f => `
        <span class="feature-tag">
          <span class="tag-icon">⚡</span>
          ${f}
        </span>
      `).join("");
    } else {
      primaryFeatContainer.innerHTML = `<span class="text-muted">None specified</span>`;
    }
  }

  // 3. Decision Path Steps (Tree Ladder)
  const stepsContainer = document.getElementById("decision-path-steps");
  if (stepsContainer) {
    if (exp.decision_path_steps && exp.decision_path_steps.length > 0) {
      stepsContainer.innerHTML = exp.decision_path_steps.map((step, idx) => {
        const isLeft = step.direction === "left" || step.direction === "<=";
        const compSymbol = isLeft ? "≤" : ">";
        return `
          <div class="step-item">
            <div class="step-num">${idx + 1}</div>
            <div class="step-content">
              <div class="step-header">
                <span class="step-feature">${step.feature}</span>
                <span class="step-node-id">Node ${step.node_id}</span>
              </div>
              <div class="step-comparison">
                <span class="step-val">Observed: <strong>${step.value}</strong></span>
                <span class="step-threshold-pill">${step.feature} ${compSymbol} ${step.threshold}</span>
                <span class="step-direction">Branch: <em>${step.direction}</em></span>
              </div>
            </div>
          </div>
        `;
      }).join("");
    } else {
      stepsContainer.innerHTML = `<p class="text-muted">Direct root leaf classification or path unavailable.</p>`;
    }
  }

  // 4. Human Explanation Narrative
  const narrativeEl = document.getElementById("human-explanation-text");
  if (narrativeEl) {
    narrativeEl.textContent = exp.human_explanation || "No explanation narrative available.";
  }

  // 5. Training-Range Warnings
  const warningsContainer = document.getElementById("warnings-container");
  const warningsList = document.getElementById("warnings-list");
  if (warningsContainer && warningsList) {
    if (exp.warnings && exp.warnings.length > 0) {
      warningsContainer.style.display = "block";
      warningsList.innerHTML = exp.warnings.map(w => `
        <li class="warning-item">
          <span class="warning-icon">⚠️</span>
          <span>${w}</span>
        </li>
      `).join("");
    } else {
      warningsContainer.style.display = "none";
      warningsList.innerHTML = "";
    }
  }

  // 6. Safety Notice
  const safetyNoticeEl = document.getElementById("safety-notice-text");
  if (safetyNoticeEl && exp.safety_notice) {
    safetyNoticeEl.textContent = exp.safety_notice;
  }
}

/**
 * Reset explanation UI to an empty / standby state.
 */
export function resetExplanationView(statusMessage = "Awaiting inference payload...") {
  const predBadge = document.getElementById("prediction-badge");
  if (predBadge) {
    predBadge.textContent = "AWAITING DATA";
    predBadge.className = "badge badge-neutral badge-lg";
  }

  const pSafeVal = document.getElementById("prob-safe-val");
  const pUnsafeVal = document.getElementById("prob-unsafe-val");
  const probBarSafe = document.getElementById("prob-bar-safe");
  const probBarUnsafe = document.getElementById("prob-bar-unsafe");
  const leafNodeEl = document.getElementById("leaf-node-badge");

  if (pSafeVal) pSafeVal.textContent = "--%";
  if (pUnsafeVal) pUnsafeVal.textContent = "--%";
  if (probBarSafe) probBarSafe.style.width = "0%";
  if (probBarUnsafe) probBarUnsafe.style.width = "0%";
  if (leafNodeEl) leafNodeEl.textContent = "Leaf Node #--";

  const primaryFeatContainer = document.getElementById("primary-features-container");
  if (primaryFeatContainer) primaryFeatContainer.innerHTML = `<span class="text-muted">--</span>`;

  const stepsContainer = document.getElementById("decision-path-steps");
  if (stepsContainer) stepsContainer.innerHTML = `<p class="text-muted">${statusMessage}</p>`;

  const narrativeEl = document.getElementById("human-explanation-text");
  if (narrativeEl) narrativeEl.textContent = statusMessage;

  const warningsContainer = document.getElementById("warnings-container");
  if (warningsContainer) warningsContainer.style.display = "none";
}
