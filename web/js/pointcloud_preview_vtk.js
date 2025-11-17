/**
 * VTK.js Point Cloud Preview for ComfyUI
 * Uses VTK.js for scientific 3D visualization
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[DepthAnythingV3] Loading VTK point cloud preview extension...");

// Register the extension
app.registerExtension({
    name: "DepthAnythingV3.PointCloudPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DA3_PreviewPointCloud") {
            console.log("[DepthAnythingV3] Registering Preview Point Cloud node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log('[DepthAnythingV3] Creating VTK point cloud preview widget');

                // Create container div
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.position = "relative";
                container.style.background = "#1a1a1a";
                container.style.borderRadius = "4px";
                container.style.overflow = "hidden";

                // Create iframe for VTK viewer
                const iframe = document.createElement("iframe");
                iframe.src = "/extensions/ComfyUI-DepthAnythingV3/viewer_vtk.html";
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.display = "block";
                container.appendChild(iframe);

                // Store iframe reference
                this._vtkIframe = iframe;

                // Add widget using ComfyUI's addDOMWidget API
                const widget = this.addDOMWidget("preview", "POINTCLOUD_PREVIEW_VTK", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Set widget size
                widget.computeSize = function(width) {
                    return [width || 512, (width || 512) + 60];  // Extra height for controls
                };

                widget.element = container;

                // Set initial node size
                this.setSize([512, 572]);

                console.log("[DepthAnythingV3] VTK Widget created successfully");

                return r;
            };

            // Handle execution
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                console.log('[DepthAnythingV3] VTK Preview node executed with message:', message);

                if (message?.file_path && this._vtkIframe) {
                    console.log('[DepthAnythingV3] Loading point cloud in VTK viewer from:', message.file_path);

                    // Handle file_path as array
                    const filePath = Array.isArray(message.file_path) ? message.file_path[0] : message.file_path;

                    // Construct URL to view the file
                    const url = `/view?filename=${encodeURIComponent(filePath.split('/').pop())}&type=output&subfolder=`;

                    console.log('[DepthAnythingV3] Sending URL to VTK iframe:', url);

                    // Send message to iframe once it's loaded
                    const sendMessage = () => {
                        if (this._vtkIframe && this._vtkIframe.contentWindow) {
                            this._vtkIframe.contentWindow.postMessage({
                                type: 'loadPointCloud',
                                url: url
                            }, '*');
                        }
                    };

                    // If iframe is already loaded, send immediately
                    if (this._vtkIframe.contentWindow) {
                        sendMessage();
                    }

                    // Also send on load in case it wasn't ready
                    this._vtkIframe.addEventListener('load', sendMessage, { once: true });
                }
            };
        }
    }
});

console.log('[DepthAnythingV3] VTK Point Cloud Preview extension loaded');
