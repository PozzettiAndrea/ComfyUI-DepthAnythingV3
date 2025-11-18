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

    // Convert to lowercase for easier matching
    const name = modelName.toLowerCase();

    // Nested model (has both camera and sky) - check first as it contains "giant" and "large"
    if (name.includes("nested") || name.includes("da3nested")) {
        return "nested";
    }

    // Mono model (no camera support, has sky)
    if (name.includes("mono") || name.includes("da3mono")) {
        return "mono";
    }

    // Metric model (no camera support, has sky)
    if (name.includes("metric") || name.includes("da3metric")) {
        return "metric";
    }

    // Main series models (have camera support, no sky)
    // Match file names like: da3_small.safetensors, da3_base.safetensors, etc.
    if (name.includes("da3_small") || name.includes("da3_base") ||
        name.includes("da3_large") || name.includes("da3_giant") ||
        name.includes("da3-small") || name.includes("da3-base") ||
        name.includes("da3-large") || name.includes("da3-giant")) {
        return "main_series";
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

// Hide an input slot from the node
function hideInputSlot(node, slotName) {
    if (!node.inputs) return null;

    const input = node.inputs.find(i => i.name === slotName);
    if (!input || input._da3_hidden) return input;

    // Store original index for restoration
    input._da3_originalIndex = node.inputs.indexOf(input);
    input._da3_hidden = true;

    // Remove from inputs array
    node.inputs.splice(input._da3_originalIndex, 1);

    // Store reference for later
    if (!node._da3_hiddenInputs) {
        node._da3_hiddenInputs = {};
    }
    node._da3_hiddenInputs[slotName] = input;

    forceUIUpdate(node);
    return input;
}

// Show a hidden input slot
function showInputSlot(node, slotName) {
    if (!node._da3_hiddenInputs || !node._da3_hiddenInputs[slotName]) return;

    const input = node._da3_hiddenInputs[slotName];
    if (!input._da3_hidden) return;

    // Re-insert at original position
    const targetIndex = input._da3_originalIndex || node.inputs.length;
    const insertIndex = Math.min(targetIndex, node.inputs.length);

    // Check if already in array
    if (node.inputs.indexOf(input) === -1) {
        node.inputs.splice(insertIndex, 0, input);
    }

    input._da3_hidden = false;

    forceUIUpdate(node);
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
    // Store hidden widgets and inputs
    node._da3_hiddenWidgets = {};
    node._da3_hiddenInputs = {};

    // Store reference to camera_params input before any hiding
    const cameraParamsInput = node.inputs?.find(i => i.name === "camera_params");
    if (cameraParamsInput) {
        node._da3_cameraParamsInput = cameraParamsInput;
    }

    let lastModelType = null;

    const updateVisibility = () => {
        const modelType = getConnectedModelType(node);

        // If no model connected, show all inputs (default state)
        if (!modelType || modelType === "unknown") {
            // Show camera_params if it was hidden
            if (node._da3_hiddenInputs["camera_params"]) {
                showInputSlot(node, "camera_params");
                console.log(`[DA3] ${node.title || node.type}: No model connected, showing camera_params`);
            }
            lastModelType = null;
            return;
        }

        // Skip if model type hasn't changed
        if (modelType === lastModelType) return;
        lastModelType = modelType;

        const capabilities = getModelCapabilities(modelType);
        const nodeTitle = node.title || node.type;

        // Hide or show camera_params based on model capabilities
        if (capabilities.has_camera_conditioning) {
            // Model supports camera conditioning - show the input
            if (node._da3_hiddenInputs["camera_params"]) {
                showInputSlot(node, "camera_params");
                console.log(`[DA3] ${nodeTitle}: Model supports camera conditioning, showing camera_params`);
            }
        } else {
            // Model does NOT support camera conditioning - hide the input
            const input = node.inputs?.find(i => i.name === "camera_params");
            if (input && !input._da3_hidden) {
                hideInputSlot(node, "camera_params");
                console.log(`[DA3] ${nodeTitle}: Model does NOT support camera conditioning, hiding camera_params`);
            }
        }

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

    // Poll for changes (in case connection change event is missed or model selection changes)
    const pollInterval = setInterval(() => {
        if (!node.graph) {
            clearInterval(pollInterval);
            return;
        }
        updateVisibility();
    }, 2000);

    // Clean up on node removal
    const origOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        clearInterval(pollInterval);
        if (origOnRemoved) {
            origOnRemoved.apply(this, arguments);
        }
    };
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
