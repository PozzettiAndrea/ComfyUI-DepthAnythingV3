/**
 * Auto-detect extension folder name from import.meta.url
 * Handles both ComfyUI-GeometryPack and comfyui-geometrypack (lowercase)
 */

export const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

/**
 * Build a viewer URL with cache buster
 * @param {string} viewerName - Name of the viewer HTML file (e.g., "viewer_vtk")
 * @returns {string} Full URL with cache buster
 */
export function getViewerUrl(viewerName) {
    return `/extensions/${EXTENSION_FOLDER}/${viewerName}.html?v=` + Date.now();
}
