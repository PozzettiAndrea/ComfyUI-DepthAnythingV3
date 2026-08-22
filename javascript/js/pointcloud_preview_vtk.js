/**
 * VTK.js Point Cloud Preview Widget
 * For point cloud visualization nodes (SAM3, DepthAnythingV3, etc.)
 */

import { app } from "../../../scripts/app.js";
import { EXTENSION_FOLDER, getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel, createWidgetOptions } from "./utils/uiComponents.js";
import { createViewerManager, createErrorHandler, buildViewUrl } from "./utils/postMessage.js";

const POINTCLOUD_NODES = ["SAM3D_PreviewPointCloud", "DA3_PreviewPointCloud"];

app.registerExtension({
    name: "depthanythingv3.pointcloud.vtk",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (POINTCLOUD_NODES.includes(nodeData.name)) {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = createContainer();

                // Create iframe for VTK.js viewer
                const iframe = createIframe(getViewerUrl("viewer_vtk"));

                // Create info panel
                const infoPanel = createInfoPanel("Point cloud will appear after execution");

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget
                const widget = this.addDOMWidget("preview_vtk", "POINTCLOUD_PREVIEW_VTK", container, createWidgetOptions());
                widget.computeSize = () => [512, 580];

                // Store references
                this._vtkIframe = iframe;
                this._infoPanel = infoPanel;

                // Create viewer manager
                const viewerManager = createViewerManager(iframe, "[PointCloud VTK]");

                // Listen for error messages
                window.addEventListener('message', createErrorHandler(infoPanel, "[PointCloud VTK]", iframe));

                // Set initial node size
                this.setSize([512, 580]);

                return r;
            };

            // Handle execution
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Get file path from ui.file_path
                let filePath = null;
                if (message?.ui?.file_path) {
                    filePath = Array.isArray(message.ui.file_path) ? message.ui.file_path[0] : message.ui.file_path;
                } else if (message?.file_path) {
                    filePath = Array.isArray(message.file_path) ? message.file_path[0] : message.file_path;
                }

                if (filePath && filePath.trim() !== '') {
                    // Normalize path
                    filePath = filePath.replace(/\\/g, '/').trim();

                    // Build URL for ComfyUI view endpoint
                    const outputMatch = filePath.match(/(?:^|\/)(output|input)\/(.+)$/);
                    let url;
                    if (outputMatch) {
                        const [, type, relativePath] = outputMatch;
                        const pathParts = relativePath.split('/');
                        const filename = pathParts.pop();
                        const subfolder = pathParts.join('/');
                        url = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                    } else {
                        const filename = filePath.split('/').pop();
                        url = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                    }

                    // Update info panel
                    if (this._infoPanel) {
                        const filename = filePath.split('/').pop();
                        this._infoPanel.innerHTML = `<strong>File:</strong> ${filename}`;
                    }

                    // Send message to iframe
                    if (this._vtkIframe?.contentWindow) {
                        const sendMessage = () => {
                            this._vtkIframe.contentWindow.postMessage({
                                type: 'LOAD_MESH',
                                filepath: url
                            }, '*');
                        };

                        if (this._vtkIframe.contentDocument?.readyState === 'complete') {
                            sendMessage();
                        } else {
                            this._vtkIframe.addEventListener('load', () => setTimeout(sendMessage, 100), { once: true });
                        }
                    }
                }
            };
        }
    }
});
