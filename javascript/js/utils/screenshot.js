/**
 * Screenshot capture and upload utility for ComfyUI viewers
 */

/**
 * Convert base64 data URL to Blob
 * @param {string} dataUrl - Base64 data URL (e.g., "data:image/png;base64,...")
 * @returns {Blob} PNG image blob
 */
export function dataUrlToBlob(dataUrl) {
    const base64Data = dataUrl.split(',')[1];
    const byteString = atob(base64Data);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);

    for (let i = 0; i < byteString.length; i++) {
        uint8Array[i] = byteString.charCodeAt(i);
    }

    return new Blob([uint8Array], { type: 'image/png' });
}

/**
 * Upload screenshot to ComfyUI backend
 * @param {string} dataUrl - Base64 data URL of the screenshot
 * @param {string} prefix - Filename prefix (e.g., "vtk-screenshot")
 * @returns {Promise<{name: string, subfolder: string, type: string}>} Upload result
 */
export async function uploadScreenshot(dataUrl, prefix = 'screenshot') {
    const blob = dataUrlToBlob(dataUrl);

    // Generate filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${prefix}-${timestamp}.png`;

    // Create FormData for upload
    const formData = new FormData();
    formData.append('image', blob, filename);
    formData.append('type', 'output');  // Save to output directory
    formData.append('subfolder', '');   // Root of output folder

    // Upload to ComfyUI backend
    const response = await fetch('/upload/image', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

/**
 * Create a message handler for screenshot events from iframe
 * @param {string} prefix - Filename prefix for screenshots
 * @param {Function} onError - Error callback
 * @returns {Function} Event handler function
 */
export function createScreenshotHandler(prefix = 'screenshot', iframe = null, onError = console.error) {
    return async (event) => {
        // Provenance guard: without it this fires for EVERY pack's iframe
        // messages in the shared page, not just ours.
        if (iframe && event.source !== iframe.contentWindow) return;
        if (event.data.type === 'SCREENSHOT' && event.data.image) {
            try {
                await uploadScreenshot(event.data.image, prefix);
            } catch (error) {
                onError(`Error saving screenshot: ${error}`);
            }
        }
    };
}
