/**
 * Formatting utilities for mesh metadata display
 */

/**
 * Format bounds array as string
 * @param {number[]} boundsMin - [x, y, z] minimum bounds
 * @param {number[]} boundsMax - [x, y, z] maximum bounds
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted bounds string or 'N/A'
 */
export function formatBounds(boundsMin, boundsMax, decimals = 2) {
    if (!boundsMin || !boundsMax || boundsMin.length !== 3 || boundsMax.length !== 3) {
        return 'N/A';
    }
    const min = boundsMin.map(v => v.toFixed(decimals)).join(', ');
    const max = boundsMax.map(v => v.toFixed(decimals)).join(', ');
    return `[${min}] to [${max}]`;
}

/**
 * Format extents array as string
 * @param {number[]} extents - [x, y, z] extents
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted extents string or 'N/A'
 */
export function formatExtents(extents, decimals = 2) {
    if (!extents || extents.length !== 3) {
        return 'N/A';
    }
    return extents.map(v => v.toFixed(decimals)).join(' x ');
}

/**
 * Format a {typeName: count} map as a sorted, human-readable string, e.g.
 * "3 Planes, 2 Cylinders, 1 BSpline".
 * @param {Object<string, number>} counts
 * @returns {string} Formatted string (empty string if counts is empty/missing)
 */
export function formatTypeCounts(counts) {
    if (!counts) return '';
    return Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => `${count} ${name}${count === 1 ? '' : 's'}`)
        .join(', ');
}

/**
 * Get color for mode label
 * @param {string} mode - Viewer mode (fields, texture, pbr)
 * @returns {string} CSS color
 */
export function getModeColor(mode) {
    switch (mode) {
        case "texture (PBR)":
        case "pbr":
            return '#fc6';
        case "texture":
            return '#c8c';
        default:
            return '#6cc';
    }
}

/**
 * Get color for watertight status
 * @param {boolean} isWatertight
 * @returns {string} CSS color
 */
export function getWatertightColor(isWatertight) {
    return isWatertight ? '#6c6' : '#c66';
}

/**
 * Build mesh info HTML for single mesh viewer
 * @param {Object} data - Mesh metadata
 * @returns {string} HTML string
 */
export function buildMeshInfoHTML(data) {
    const {
        mode = "fields",
        vertices = 'N/A',
        faces = 'N/A',
        boundsMin = [],
        boundsMax = [],
        extents = [],
        avgEdge,
        isWatertight,
        fieldNames = [],
        fieldRanges = [],
        hasTexture,
        hasVertexColors,
        visualKind
    } = data;

    const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
    const modeColor = getModeColor(mode);
    const boundsStr = formatBounds(boundsMin, boundsMax);
    const extentsStr = formatExtents(extents);

    let html = `
        <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
            <span style="color: #888;">Mode:</span>
            <span style="color: ${modeColor}; font-weight: bold;">${modeLabel}</span>

            <span style="color: #888;">Vertices:</span>
            <span>${typeof vertices === 'number' ? vertices.toLocaleString() : vertices}</span>

            <span style="color: #888;">Faces:</span>
            <span>${typeof faces === 'number' ? faces.toLocaleString() : faces}</span>

            <span style="color: #888;">Bounds:</span>
            <span style="font-size: 9px;">${boundsStr}</span>

            <span style="color: #888;">Extents:</span>
            <span>${extentsStr}</span>
    `;

    // Average edge length (mesh resolution)
    if (avgEdge !== undefined && avgEdge !== null) {
        const avgStr = (typeof avgEdge === 'number')
            ? avgEdge.toLocaleString(undefined, { maximumSignificantDigits: 4 })
            : avgEdge;
        html += `
            <span style="color: #888;">Avg edge:</span>
            <span>${avgStr}</span>
        `;
    }

    // Watertight status
    if (isWatertight !== undefined) {
        const watertightText = isWatertight ? 'Yes' : 'No';
        const watertightColor = getWatertightColor(isWatertight);
        html += `
            <span style="color: #888;">Watertight:</span>
            <span style="color: ${watertightColor};">${watertightText}</span>
        `;
    }

    // Mode-specific info
    if (mode === "texture" || mode === "pbr") {
        if (visualKind !== undefined) {
            html += `
                <span style="color: #888;">Visual Kind:</span>
                <span>${visualKind}</span>
            `;
        }
        if (hasTexture !== undefined) {
            const texColor = hasTexture ? '#c8c' : '#888';
            html += `
                <span style="color: #888;">Textures:</span>
                <span style="color: ${texColor};">${hasTexture ? 'Yes' : 'No'}</span>
            `;
        }
        if (hasVertexColors !== undefined) {
            html += `
                <span style="color: #888;">Vertex Colors:</span>
                <span>${hasVertexColors ? 'Yes' : 'No'}</span>
            `;
        }
    } else {
        // Fields mode — annotate each field with its [min, max] value range
        if (fieldNames && fieldNames.length > 0) {
            const fmt = (n) => {
                if (n == null || !isFinite(n)) return '?';
                if (Number.isInteger(n) && Math.abs(n) < 1e7) return String(n);
                return Number(Number(n).toPrecision(4)).toString();
            };
            const ext = (i, which, v) =>
                `<span class="field-extreme" data-fidx="${i}" data-which="${which}" ` +
                `style="cursor:pointer;text-decoration:underline dotted;" ` +
                `title="focus camera on ${which}">${fmt(v)}</span>`;
            const fields = fieldNames.map((name, i) => {
                const r = fieldRanges && fieldRanges[i];
                return (r && r.length === 2)
                    ? `${name} [${ext(i, 'min', r[0])}, ${ext(i, 'max', r[1])}]`
                    : name;
            }).join(', ');
            html += `
                <span style="color: #888;">Fields:</span>
                <span style="font-size: 9px; color: #6cc;">${fields}</span>
            `;
        } else {
            html += `
                <span style="color: #888;">Fields:</span>
                <span style="color: #888;">None</span>
            `;
        }
    }

    html += '</div>';
    return html;
}

/**
 * Build dual mesh info HTML for side-by-side comparison
 * @param {Object} data - Dual mesh metadata
 * @returns {string} HTML string
 */
export function buildDualMeshInfoHTML(data) {
    const {
        mode = "fields",
        layout = "side_by_side",
        mesh1 = {},
        mesh2 = {},
        commonFields = []
    } = data;

    const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
    const modeColor = getModeColor(mode);

    const extentsStr1 = formatExtents(mesh1.extents);
    const extentsStr2 = formatExtents(mesh2.extents);

    const layoutSuffix = layout === "overlay" ? " (Overlay)" : "";

    let html = `
        <div style="display: grid; grid-template-columns: auto 1fr 1fr; gap: 2px 12px;">
            <span style="color: #888;">Mode:</span>
            <span colspan="2" style="grid-column: 2 / 4; color: ${modeColor}; font-weight: bold;">${modeLabel}${layoutSuffix}</span>

            <span style="color: #888;"></span>
            <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 1</span>
            <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 2</span>

            <span style="color: #888;">Vertices:</span>
            <span>${(mesh1.vertices || 'N/A').toLocaleString()}</span>
            <span>${(mesh2.vertices || 'N/A').toLocaleString()}</span>

            <span style="color: #888;">Faces:</span>
            <span>${(mesh1.faces || 'N/A').toLocaleString()}</span>
            <span>${(mesh2.faces || 'N/A').toLocaleString()}</span>
    `;

    if (mesh1.faceTypes !== undefined && mesh2.faceTypes !== undefined) {
        html += `
            <span style="color: #888;">Face types:</span>
            <span style="font-size: 9px;">${mesh1.faceTypes || 'n/a'}</span>
            <span style="font-size: 9px;">${mesh2.faceTypes || 'n/a'}</span>
        `;
    }
    if (mesh1.edgeTypes !== undefined && mesh2.edgeTypes !== undefined) {
        html += `
            <span style="color: #888;">Edge types:</span>
            <span style="font-size: 9px;">${mesh1.edgeTypes || 'n/a'}</span>
            <span style="font-size: 9px;">${mesh2.edgeTypes || 'n/a'}</span>
        `;
    }

    html += `
            <span style="color: #888;">Extents:</span>
            <span style="font-size: 9px;">${extentsStr1}</span>
            <span style="font-size: 9px;">${extentsStr2}</span>
    `;

    // Watertight info
    if (mesh1.isWatertight !== undefined && mesh2.isWatertight !== undefined) {
        const color1 = getWatertightColor(mesh1.isWatertight);
        const color2 = getWatertightColor(mesh2.isWatertight);
        html += `
            <span style="color: #888;">Watertight:</span>
            <span style="color: ${color1};">${mesh1.isWatertight ? 'Yes' : 'No'}</span>
            <span style="color: ${color2};">${mesh2.isWatertight ? 'Yes' : 'No'}</span>
        `;
    }

    // Mode-specific info
    if (mode === "texture") {
        if (mesh1.hasTexture !== undefined) {
            const tex1 = mesh1.hasTexture ? 'Yes' : 'No';
            const tex2 = mesh2.hasTexture ? 'Yes' : 'No';
            html += `
                <span style="color: #888;">Textures:</span>
                <span style="color: ${mesh1.hasTexture ? '#c8c' : '#888'};">${tex1}</span>
                <span style="color: ${mesh2.hasTexture ? '#c8c' : '#888'};">${tex2}</span>
            `;
        }
    } else {
        // Fields mode
        if (commonFields && commonFields.length > 0) {
            html += `
                <span style="color: #888;">Fields:</span>
                <span colspan="2" style="grid-column: 2 / 4; color: #9c9;">${commonFields.length} shared field(s)</span>
            `;
        }
    }

    html += '</div>';
    return html;
}
