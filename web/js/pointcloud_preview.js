/**
 * Point Cloud & Gaussian Splatting Preview for ComfyUI
 * Using Three.js for 3D visualization
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[DepthAnythingV3] Loading point cloud preview extension...");

// Helper class to manage Three.js scene
class PointCloudViewer {
    constructor(container) {
        this.container = container;
        this.canvas = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.pointCloud = null;
        this.controls = null;

        this.setupUI();
    }

    setupUI() {
        // Create canvas
        this.canvas = document.createElement("canvas");
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.container.appendChild(this.canvas);

        // Create controls overlay
        this.controls = document.createElement("div");
        this.controls.style.position = "absolute";
        this.controls.style.top = "10px";
        this.controls.style.right = "10px";
        this.controls.style.background = "rgba(0,0,0,0.7)";
        this.controls.style.padding = "8px";
        this.controls.style.borderRadius = "4px";
        this.controls.style.color = "white";
        this.controls.style.fontSize = "12px";
        this.controls.style.fontFamily = "monospace";
        this.controls.innerHTML = `
            <div>Left: Rotate | Right: Pan</div>
            <div>Scroll: Zoom</div>
            <div id="point-count">Points: 0</div>
        `;
        this.container.appendChild(this.controls);

        // Initialize Three.js
        this.initThreeJS();
    }

    async initThreeJS() {
        // Load Three.js if not available
        if (!window.THREE) {
            await this.loadThreeJS();
        }

        const THREE = window.THREE;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.canvas.clientWidth / this.canvas.clientHeight,
            0.001,
            1000
        );
        this.camera.position.set(0, 0, 2);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Grid
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // Axes
        const axesHelper = new THREE.AxesHelper(0.5);
        this.scene.add(axesHelper);

        // Mouse controls
        this.setupControls();

        // Animation loop
        this.animate();

        console.log("[DepthAnythingV3] Three.js initialized");
    }

    async loadThreeJS() {
        if (window.THREE) {
            return;
        }

        try {
            // Try to load from local bundled Three.js ES module
            // Get the current script path to calculate relative path
            const currentScript = document.currentScript ||
                Array.from(document.scripts).find(s => s.src.includes('pointcloud_preview'));

            let basePath = '';
            if (currentScript && currentScript.src) {
                const scriptUrl = new URL(currentScript.src);
                basePath = scriptUrl.pathname.substring(0, scriptUrl.pathname.lastIndexOf('/'));
            } else {
                // Fallback: find the extension path from existing ComfyUI extensions
                basePath = '/extensions/ComfyUI-DepthAnythingV3/js';
            }

            const threeModulePath = `${basePath}/lib/three.module.min.js`;
            console.log("[DepthAnythingV3] Loading Three.js from:", threeModulePath);

            // Use dynamic import for ES module
            const THREE = await import(threeModulePath);
            window.THREE = THREE;
            console.log("[DepthAnythingV3] Three.js loaded from local ES module");
        } catch (error) {
            console.warn("[DepthAnythingV3] Failed to load local Three.js, falling back to CDN:", error);
            // Fallback to CDN if local loading fails
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
                script.onload = () => {
                    console.log("[DepthAnythingV3] Three.js loaded from CDN (fallback)");
                    resolve();
                };
                script.onerror = () => reject(new Error('Failed to load Three.js'));
                document.head.appendChild(script);
            });
        }
    }

    setupControls() {
        let isDragging = false;
        let isPanning = false;
        let previousMousePosition = { x: 0, y: 0 };

        this.canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            isPanning = e.button === 2; // Right click for pan
            previousMousePosition = { x: e.clientX, y: e.clientY };
            e.preventDefault();
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            if (isPanning) {
                // Pan camera
                const panSpeed = 0.001;
                this.camera.position.x -= deltaX * panSpeed;
                this.camera.position.y += deltaY * panSpeed;
            } else {
                // Rotate camera around center
                const rotateSpeed = 0.005;
                const theta = deltaX * rotateSpeed;
                const phi = deltaY * rotateSpeed;

                const radius = this.camera.position.length();
                const spherical = {
                    radius: radius,
                    theta: Math.atan2(this.camera.position.x, this.camera.position.z) + theta,
                    phi: Math.acos(this.camera.position.y / radius) + phi
                };

                // Clamp phi to avoid gimbal lock
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                this.camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
                this.camera.position.y = spherical.radius * Math.cos(spherical.phi);
                this.camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
                this.camera.lookAt(0, 0, 0);
            }

            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        this.canvas.addEventListener('mouseup', () => {
            isDragging = false;
            isPanning = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.001;
            const delta = e.deltaY * zoomSpeed;
            const newDistance = this.camera.position.length() * (1 + delta);
            this.camera.position.multiplyScalar(newDistance / this.camera.position.length());
        });

        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    animate() {
        if (!this.renderer) return;

        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    async loadPointCloud(plyPath) {
        if (!plyPath) return;

        try {
            console.log('[DepthAnythingV3] Fetching PLY file:', plyPath);
            const response = await fetch(plyPath);
            const text = await response.text();

            // Parse PLY
            const pointData = this.parsePLY(text);

            // Create Three.js point cloud
            this.displayPointCloud(pointData);

        } catch (error) {
            console.error('[DepthAnythingV3] Error loading point cloud:', error);
        }
    }

    parsePLY(plyText) {
        const lines = plyText.split('\n');
        let vertexCount = 0;
        let headerEnded = false;
        let hasColor = false;

        const positions = [];
        const colors = [];

        // Parse header
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            if (line.startsWith('element vertex')) {
                vertexCount = parseInt(line.split(' ')[2]);
            } else if (line.includes('property') && (line.includes('red') || line.includes('green') || line.includes('blue'))) {
                hasColor = true;
            } else if (line === 'end_header') {
                headerEnded = true;

                // Parse vertex data
                for (let j = i + 1; j < lines.length && positions.length / 3 < vertexCount; j++) {
                    const vertexLine = lines[j].trim();
                    if (!vertexLine) continue;

                    const parts = vertexLine.split(/\s+/);
                    if (parts.length < 3) continue;

                    // Position (x, y, z)
                    positions.push(parseFloat(parts[0]));
                    positions.push(parseFloat(parts[1]));
                    positions.push(parseFloat(parts[2]));

                    // Color (r, g, b) - if available
                    if (hasColor && parts.length >= 6) {
                        colors.push(parseInt(parts[3]) / 255);
                        colors.push(parseInt(parts[4]) / 255);
                        colors.push(parseInt(parts[5]) / 255);
                    } else {
                        // Default white color
                        colors.push(1.0);
                        colors.push(1.0);
                        colors.push(1.0);
                    }
                }
                break;
            }
        }

        console.log(`[DepthAnythingV3] Parsed ${positions.length / 3} points, hasColor: ${hasColor}`);

        return {
            positions: new Float32Array(positions),
            colors: new Float32Array(colors),
            count: positions.length / 3
        };
    }

    displayPointCloud(pointData) {
        const THREE = window.THREE;

        // Remove existing point cloud
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(pointData.positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(pointData.colors, 3));

        // Center and normalize the point cloud
        geometry.computeBoundingSphere();
        const center = geometry.boundingSphere.center;
        const radius = geometry.boundingSphere.radius;

        geometry.translate(-center.x, -center.y, -center.z);

        if (radius > 0) {
            const scale = 1.0 / radius;
            geometry.scale(scale, scale, scale);
        }

        // Create material
        const material = new THREE.PointsMaterial({
            size: 0.005,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Create point cloud
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        // Update point count
        const pointCountDiv = this.controls.querySelector('#point-count');
        if (pointCountDiv) {
            pointCountDiv.textContent = `Points: ${pointData.count.toLocaleString()}`;
        }

        // Reset camera
        this.camera.position.set(0, 0, 2);
        this.camera.lookAt(0, 0, 0);

        console.log('[DepthAnythingV3] Point cloud displayed');
    }

    cleanup() {
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.pointCloud) {
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }
    }
}

// Register the extension
app.registerExtension({
    name: "DepthAnythingV3.PointCloudPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DA3_PreviewPointCloud") {
            console.log("[DepthAnythingV3] Registering Preview Point Cloud node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log('[DepthAnythingV3] Creating point cloud preview widget');

                // Create container div
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.position = "relative";
                container.style.background = "#1a1a1a";
                container.style.borderRadius = "4px";
                container.style.overflow = "hidden";

                // Add widget using ComfyUI's addDOMWidget API
                const widget = this.addDOMWidget("preview", "POINTCLOUD_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Set widget size
                widget.computeSize = function(width) {
                    return [width || 512, width || 512];
                };

                widget.element = container;

                // Initialize Three.js viewer
                const viewer = new PointCloudViewer(container);
                this._pointCloudViewer = viewer;

                // Set initial node size
                this.setSize([512, 512]);

                console.log("[DepthAnythingV3] Widget created successfully");

                return r;
            };

            // Handle execution
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                console.log('[DepthAnythingV3] Preview node executed with message:', message);

                if (message?.file_path && this._pointCloudViewer) {
                    console.log('[DepthAnythingV3] Loading point cloud from:', message.file_path);

                    // Handle file_path as array (ComfyUI returns arrays from ui outputs)
                    const filePath = Array.isArray(message.file_path) ? message.file_path[0] : message.file_path;

                    // Construct URL to view the file
                    const url = `/view?filename=${encodeURIComponent(filePath.split('/').pop())}&type=output&subfolder=`;

                    console.log('[DepthAnythingV3] Fetching from URL:', url);
                    this._pointCloudViewer.loadPointCloud(url);
                }
            };
        }
    }
});

console.log('[DepthAnythingV3] Point Cloud Preview extension loaded');
