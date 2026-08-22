/**
 * PostMessage utilities for iframe communication
 */

/**
 * Create a message sender function for an iframe
 * @param {HTMLIFrameElement} iframe - The iframe element
 * @param {string} logPrefix - Logging prefix (e.g., "[GeomPack VTK]")
 * @returns {Function} Message sending function
 */
export function createMessageSender(iframe, logPrefix = "[Viewer]") {
    return (data) => {
        if (iframe.contentWindow) {
            iframe.contentWindow.postMessage(data, "*");
        } else {
            console.error(`${logPrefix} Iframe contentWindow not available`);
        }
    };
}

/**
 * Create a mesh load message
 * @param {string} filepath - URL path to the mesh file
 * @returns {Object} Message data object
 */
export function createLoadMeshMessage(filepath) {
    return {
        type: "LOAD_MESH",
        filepath: filepath,
        timestamp: Date.now()
    };
}

/**
 * Create a dual mesh load message
 * @param {Object} options - Message options
 * @returns {Object} Message data object
 */
export function createLoadDualMeshMessage(options) {
    const {
        layout,
        mesh1Filepath,
        mesh2Filepath,
        meshFilepath,  // For overlay mode
        opacity1 = 1.0,
        opacity2 = 1.0,
        showEdges = false,
        cameraState = "",
        selectedField = "",
        selectedChannel = "magnitude",
        selectedColormap = "erdc_rainbow_bright",
    } = options;

    return {
        type: 'LOAD_DUAL_MESH',
        layout: layout,
        mesh1Filepath,
        mesh2Filepath,
        meshFilepath,
        opacity1,
        opacity2,
        showEdges,
        cameraState,
        selectedField,
        selectedChannel,
        selectedColormap,
        timestamp: Date.now()
    };
}

/**
 * Build ComfyUI view API URL for a file.
 * Handles absolute paths (/home/.../output/sub/file.glb),
 * relative paths (output/sub/file.glb), and bare filenames (file.glb).
 * @param {string} filename - The filename or path
 * @param {string} defaultType - Default file type (default: "output")
 * @param {string} defaultSubfolder - Default subfolder path (default: "")
 * @returns {string} API URL
 */
export function buildViewUrl(filename, defaultType = "output", defaultSubfolder = "") {
    const normalized = filename.replace(/\\/g, '/');
    const pathMatch = normalized.match(/(?:^|\/)(output|input|temp)\/(.+)$/);
    if (pathMatch) {
        const [, type, relPath] = pathMatch;
        const parts = relPath.split('/');
        const fname = parts.pop();
        const subfolder = parts.join('/');
        return `/view?filename=${encodeURIComponent(fname)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
    }
    // Bare filename fallback
    const basename = normalized.split('/').pop();
    return `/view?filename=${encodeURIComponent(basename)}&type=${defaultType}&subfolder=${defaultSubfolder}`;
}

/**
 * Create an iframe viewer manager for handling viewer switching
 * @param {HTMLIFrameElement} iframe - The iframe element
 * @param {string} logPrefix - Logging prefix
 * @returns {Object} Viewer manager object
 */
export function createViewerManager(iframe, logPrefix = "[Viewer]") {
    let currentViewerType = null;
    let iframeLoaded = false;
    let pendingMessage = null;

    // Track iframe load state
    iframe.addEventListener('load', () => {
        iframeLoaded = true;
        // Send pending message if any
        if (pendingMessage && iframe.contentWindow) {
            iframe.contentWindow.postMessage(pendingMessage, "*");
            pendingMessage = null;
        }
    });

    return {
        /**
         * Get current viewer type
         * @returns {string|null}
         */
        getCurrentType() {
            return currentViewerType;
        },

        /**
         * Check if iframe is loaded
         * @returns {boolean}
         */
        isLoaded() {
            return iframeLoaded;
        },

        /**
         * Send a message to the iframe viewer
         * If iframe is not loaded yet, queue the message
         * @param {Object} data - Message data
         */
        sendMessage(data) {
            if (iframeLoaded && iframe.contentWindow) {
                iframe.contentWindow.postMessage(data, "*");
            } else {
                pendingMessage = data;
            }
        },

        /**
         * Switch to a different viewer type if needed
         * @param {string} viewerType - New viewer type
         * @param {string} viewerUrl - URL of the new viewer
         * @param {Object} messageData - Message to send after viewer loads
         * @returns {boolean} True if viewer was switched, false if already on correct viewer
         */
        switchViewer(viewerType, viewerUrl, messageData) {
            if (viewerType === currentViewerType) {
                // No switch needed, just send message
                this.sendMessage(messageData);
                return false;
            }

            // Viewer switch needed
            currentViewerType = viewerType;
            iframeLoaded = false;
            pendingMessage = messageData;

            // Change iframe src to trigger reload
            iframe.src = viewerUrl + "?v=" + Date.now();
            return true;
        },

        /**
         * Reset the manager state
         */
        reset() {
            currentViewerType = null;
            iframeLoaded = false;
            pendingMessage = null;
        }
    };
}

/**
 * Create an error message handler for iframe errors
 * @param {HTMLElement} infoPanel - The info panel to display errors
 * @param {string} logPrefix - Logging prefix
 * @returns {Function} Event handler function
 */
export function createErrorHandler(infoPanel, logPrefix = "[Viewer]", iframe = null) {
    return (event) => {
        // Provenance guard: without it this fires for EVERY pack's iframe
        // messages in the shared page, not just ours.
        if (iframe && event.source !== iframe.contentWindow) return;
        if (event.data.type === 'MESH_ERROR' && event.data.error) {
            console.error(`${logPrefix} Error from viewer:`, event.data.error);
            if (infoPanel) {
                infoPanel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error: ${event.data.error}</div>`;
            }
        }
    };
}
