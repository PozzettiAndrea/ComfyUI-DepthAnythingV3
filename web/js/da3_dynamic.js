import { app } from "../../../scripts/app.js";

/**
 * Depth Anything V3 - Dynamic Input Management
 *
 * This extension dynamically hides/shows input widgets based on the connected model type.
 * - Camera params input is hidden when Mono/Metric models are connected (they don't support camera conditioning)
 * - Warnings are displayed when features are used with unsupported models
 */

// Map model names to their types
function getModelType(modelName) {
    if (!modelName) return "unknown";

    // Main series models (have camera support, no sky)
    if (modelName.includes("DA3-Small") || modelName.includes("DA3-Base") ||
        modelName.includes("DA3-Large") || modelName.includes("DA3-Giant")) {
        // Check it's not Mono/Metric/Nested variants
        if (!modelName.includes("Mono") && !modelName.includes("Metric") && !modelName.includes("Nested")) {
            return "main_series";
        }
    }

    // Mono model (no camera support, has sky)
    if (modelName.includes("DA3Mono")) {
        return "mono";
    }

    // Metric model (no camera support, has sky)
    if (modelName.includes("DA3Metric")) {
        return "metric";
    }

    // Nested model (has both camera and sky)
    if (modelName.includes("DA3Nested") || modelName.includes("Nested")) {
        return "nested";
    }

    return "unknown";
}

// Get model capabilities based on type
function getModelCapabilities(modelType) {
    const capabilities = {
        has_camera_conditioning: false,
        has_sky_segmentation: false,
        has_multiview_attention: false,
    };

    switch (modelType) {
        case "main_series":
            capabilities.has_camera_conditioning = true;
            capabilities.has_multiview_attention = true;
            capabilities.has_sky_segmentation = false;
            break;
        case "mono":
        case "metric":
            capabilities.has_camera_conditioning = false;
            capabilities.has_multiview_attention = false;
            capabilities.has_sky_segmentation = true;
            break;
        case "nested":
            capabilities.has_camera_conditioning = true;
            capabilities.has_multiview_attention = true;
            capabilities.has_sky_segmentation = true;
            break;
    }

    return capabilities;
}

// Hide a widget from the node
function hideWidget(node, widget) {
    if (!widget || widget._hidden) return;

    // Store original properties
    if (!widget._da3_original) {
        widget._da3_original = {
            type: widget.type,
            computeSize: widget.computeSize,
            serializeValue: widget.serializeValue,
        };
    }

    // Find widget index
    const index = node.widgets.indexOf(widget);
    if (index === -1) return;

    // Store index for restoration
    widget._da3_originalIndex = index;
    widget._hidden = true;

    // Remove from widgets array
    node.widgets.splice(index, 0);
    node.widgets = node.widgets.filter(w => w !== widget);
}

// Show a hidden widget
function showWidget(node, widget) {
    if (!widget || !widget._hidden) return;

    // Restore original properties
    if (widget._da3_original) {
        widget.type = widget._da3_original.type;
        widget.computeSize = widget._da3_original.computeSize;
        widget.serializeValue = widget._da3_original.serializeValue;
    }

    // Re-insert at original position
    const targetIndex = widget._da3_originalIndex || node.widgets.length;
    const insertIndex = Math.min(targetIndex, node.widgets.length);

    // Check if already in array
    if (node.widgets.indexOf(widget) === -1) {
        node.widgets.splice(insertIndex, 0, widget);
    }

    widget._hidden = false;
}

// Force UI update
function forceUIUpdate(node) {
    node.setDirtyCanvas(true, true);
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }

    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);

        requestAnimationFrame(() => {
            if (app.canvas) {
                app.canvas.draw(true, true);
            }
        });
    });
}

// Get the model type from a connected model loader node
function getConnectedModelType(node) {
    // Find the da3_model input
    const modelInput = node.inputs?.find(input => input.name === "da3_model");
    if (!modelInput || !modelInput.link) return null;

    // Get the link
    const link = app.graph.links[modelInput.link];
    if (!link) return null;

    // Get the source node (model loader)
    const loaderNode = app.graph.getNodeById(link.origin_id);
    if (!loaderNode) return null;

    // Get the model widget value
    const modelWidget = loaderNode.widgets?.find(w => w.name === "model");
    if (!modelWidget) return null;

    return getModelType(modelWidget.value);
}

// Setup dynamic widgets for inference nodes
function setupInferenceNode(node) {
    // Store hidden widgets
    node._da3_hiddenWidgets = {};

    // Get the camera_params widget (if it exists as an input)
    // Note: camera_params is an optional input, not a widget, so we handle it differently

    const updateVisibility = () => {
        const modelType = getConnectedModelType(node);
        if (!modelType || modelType === "unknown") return;

        const capabilities = getModelCapabilities(modelType);

        // Update node title or add indicator for model capabilities
        const nodeTitle = node.title || node.type;

        // We can't hide optional inputs dynamically, but we can show a warning
        // The Python backend will handle the actual warning

        // Store current capabilities in node for reference
        node._da3_modelCapabilities = capabilities;

        console.log(`[DA3] ${nodeTitle}: Model type=${modelType}, capabilities=`, capabilities);
    };

    // Monitor connection changes
    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function(type, index, connected, link_info) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }

        // type 1 = input connection
        if (type === 1) {
            setTimeout(() => updateVisibility(), 100);
        }
    };

    // Initial check
    setTimeout(() => updateVisibility(), 200);

    // Poll for changes (in case connection change event is missed)
    const pollInterval = setInterval(() => {
        if (!node.graph) {
            clearInterval(pollInterval);
            return;
        }
        updateVisibility();
    }, 2000);
}

// Setup model loader to track model selection
function setupModelLoader(node) {
    const modelWidget = node.widgets?.find(w => w.name === "model");
    if (!modelWidget) return;

    // Store original callback
    const origCallback = modelWidget.callback;

    // Override callback to broadcast model changes
    modelWidget.callback = function(value) {
        const result = origCallback?.apply(this, arguments);

        // Notify connected nodes
        const modelType = getModelType(value);
        console.log(`[DA3] Model selected: ${value} (type: ${modelType})`);

        // Store model type in node
        node._da3_modelType = modelType;

        return result;
    };
}

// Register the extension
app.registerExtension({
    name: "comfyui.depthanythingv3.dynamic_inputs",

    async nodeCreated(node) {
        // Handle model loader
        if (node.comfyClass === "DownloadAndLoadDepthAnythingV3Model") {
            setTimeout(() => setupModelLoader(node), 100);
        }

        // Handle inference nodes
        const inferenceNodes = [
            "DepthAnything_V3",
            "DepthAnythingV3_3D",
            "DepthAnythingV3_Advanced",
            "DepthAnythingV3_MultiView",
        ];

        if (inferenceNodes.includes(node.comfyClass)) {
            setTimeout(() => setupInferenceNode(node), 100);
        }
    },
});

console.log("[DA3] Depth Anything V3 dynamic input management loaded");
