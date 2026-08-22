/**
 * UI component factory functions for ComfyUI 3D viewer widgets
 */

/**
 * Create a container div for viewer + info panel
 * @param {Object} options - Container options
 * @param {string} options.backgroundColor - Background color (default: "#2a2a2a")
 * @returns {HTMLDivElement} Container element
 */
export function createContainer(options = {}) {
    const { backgroundColor = "#2a2a2a" } = options;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.backgroundColor = backgroundColor;
    container.style.overflow = "hidden";

    return container;
}

/**
 * Create an iframe for embedding a viewer
 * @param {string} src - Initial iframe source URL
 * @param {Object} options - Iframe options
 * @param {string} options.minHeight - Minimum height (default: "0")
 * @param {string} options.backgroundColor - Background color (default: "#2a2a2a")
 * @returns {HTMLIFrameElement} Iframe element
 */
export function createIframe(src, options = {}) {
    const {
        minHeight = "0",
        backgroundColor = "#2a2a2a"
    } = options;

    const iframe = document.createElement("iframe");
    iframe.style.width = "100%";
    iframe.style.flex = "1 1 0";
    iframe.style.minHeight = minHeight;
    iframe.style.border = "none";
    iframe.style.backgroundColor = backgroundColor;
    iframe.src = src;

    return iframe;
}

/**
 * Create an info panel for displaying mesh metadata
 * @param {string} placeholder - Initial placeholder text
 * @param {Object} options - Panel options
 * @returns {HTMLDivElement} Info panel element
 */
export function createInfoPanel(placeholder = "Info will appear here after execution", options = {}) {
    const {
        backgroundColor = "#1a1a1a",
        borderTop = "1px solid #444",
        padding = "6px 12px",
        fontSize = "10px",
        fontFamily = "monospace",
        color = "#ccc",
        lineHeight = "1.3"
    } = options;

    const infoPanel = document.createElement("div");
    infoPanel.style.backgroundColor = backgroundColor;
    infoPanel.style.borderTop = borderTop;
    infoPanel.style.padding = padding;
    infoPanel.style.fontSize = fontSize;
    infoPanel.style.fontFamily = fontFamily;
    infoPanel.style.color = color;
    infoPanel.style.lineHeight = lineHeight;
    infoPanel.style.flexShrink = "0";
    infoPanel.style.overflow = "hidden";
    infoPanel.innerHTML = `<span style="color: #888;">${placeholder}</span>`;

    return infoPanel;
}

/**
 * Create an analysis info panel (for connected components, self-intersections, etc.)
 * @param {string} placeholder - Initial placeholder text
 * @param {Object} options - Panel options
 * @returns {HTMLDivElement} Analysis panel element
 */
export function createAnalysisPanel(placeholder = "Run workflow to see results", options = {}) {
    const {
        backgroundColor = "#1a1a1a",
        padding = "8px",
        fontSize = "10px",
        fontFamily = "monospace",
        color = "#ccc",
        lineHeight = "1.4",
        maxHeight = "200px",
        borderRadius = "4px"
    } = options;

    const panel = document.createElement("div");
    panel.style.backgroundColor = backgroundColor;
    panel.style.padding = padding;
    panel.style.fontSize = fontSize;
    panel.style.fontFamily = fontFamily;
    panel.style.color = color;
    panel.style.lineHeight = lineHeight;
    panel.style.maxHeight = maxHeight;
    panel.style.overflowY = "auto";
    panel.style.borderRadius = borderRadius;
    panel.innerHTML = `<span style="color: #666;">${placeholder}</span>`;

    return panel;
}

/**
 * Show error in an info panel
 * @param {HTMLElement} panel - The info panel element
 * @param {string} message - Error message
 */
export function showPanelError(panel, message) {
    panel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error: ${message}</div>`;
}

/**
 * Create default widget options
 * @returns {Object} Widget options object
 */
export function createWidgetOptions() {
    return {
        getValue() { return ""; },
        setValue(v) { }
    };
}
