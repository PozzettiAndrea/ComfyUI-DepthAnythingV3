/**
 * Analysis panel utilities for mesh analysis nodes
 * (Connected components, degenerate faces, open edges, self-intersections)
 */

/**
 * Build table header HTML
 * @param {string[]} columns - Column headers
 * @returns {string} HTML string
 */
export function buildTableHeader(columns) {
    let html = '<tr style="color: #888; border-bottom: 1px solid #333;">';
    for (const col of columns) {
        const align = col.align || 'left';
        html += `<th style="text-align: ${align}; padding: 2px 4px;">${col.label}</th>`;
    }
    html += '</tr>';
    return html;
}

/**
 * Build table row HTML
 * @param {Object[]} cells - Cell data [{value, color, align}]
 * @returns {string} HTML string
 */
export function buildTableRow(cells) {
    let html = '<tr style="border-bottom: 1px solid #222;">';
    for (const cell of cells) {
        const color = cell.color || '#ccc';
        const align = cell.align || 'left';
        const value = cell.value;
        html += `<td style="text-align: ${align}; padding: 2px 4px; color: ${color};">${value}</td>`;
    }
    html += '</tr>';
    return html;
}

/**
 * Build "and N more" row
 * @param {number} remaining - Number of remaining items
 * @param {number} colSpan - Number of columns to span
 * @param {string} itemName - Name of items (e.g., "components", "faces")
 * @returns {string} HTML string
 */
export function buildMoreRow(remaining, colSpan, itemName = "items") {
    return `
        <tr><td colspan="${colSpan}" style="padding: 4px; color: #888; text-align: center;">
            ... and ${remaining} more ${itemName}
        </td></tr>
    `;
}

/**
 * Build connected components HTML
 * @param {Object} meshData - Mesh component data
 * @param {number} maxDisplay - Max components to display (default: 20)
 * @returns {string} HTML string
 */
export function buildConnectedComponentsHTML(meshData, maxDisplay = 20) {
    const { mesh_name, num_components, total_faces, total_vertices, components } = meshData;

    let html = '<div style="margin-bottom: 8px;">';

    // Header
    html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
    html += `${mesh_name}: <span style="color: #6cf;">${num_components}</span> component(s)`;
    html += `</div>`;

    html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
    html += `Total: ${total_vertices.toLocaleString()} verts, ${total_faces.toLocaleString()} faces`;
    html += `</div>`;

    // Table
    html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
    html += buildTableHeader([
        { label: '#', align: 'left' },
        { label: 'Faces', align: 'right' },
        { label: 'Vertices', align: 'right' }
    ]);

    const displayComponents = components.slice(0, maxDisplay);
    const maxFaces = Math.max(...components.map(c => c.faces));

    for (const comp of displayComponents) {
        const color = comp.faces === maxFaces ? '#6f6' : '#ccc';

        html += buildTableRow([
            { value: comp.id, color: '#888', align: 'left' },
            { value: comp.faces.toLocaleString(), color: color, align: 'right' },
            { value: comp.vertices.toLocaleString(), color: '#ccc', align: 'right' }
        ]);

        // Face indices for small components
        if (comp.face_indices && comp.face_indices.length > 0) {
            html += `<tr style="border-bottom: 1px solid #222;">`;
            html += `<td colspan="3" style="padding: 2px 4px 4px 16px; color: #888; font-size: 8px;">`;
            html += `faces: ${comp.face_indices.join(', ')}`;
            html += `</td></tr>`;
        }
    }

    if (components.length > maxDisplay) {
        html += buildMoreRow(components.length - maxDisplay, 3, "components");
    }

    html += '</table></div>';
    return html;
}

/**
 * Build self-intersections HTML
 * @param {Object} data - Self-intersection data
 * @param {number} maxDisplay - Max faces to display (default: 20)
 * @returns {string} HTML string
 */
export function buildSelfIntersectionsHTML(data, maxDisplay = 20) {
    const { mesh_name, num_intersecting_faces, num_intersection_pairs, total_faces, total_vertices, has_cgal, faces } = data;

    let html = '<div style="margin-bottom: 8px;">';

    const statusColor = num_intersecting_faces === 0 ? '#6f6' : '#f66';
    const statusText = num_intersecting_faces === 0 ? 'Clean' : `${num_intersecting_faces} intersecting face(s)`;

    html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
    html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
    html += `</div>`;

    if (!has_cgal) {
        html += `<div style="color: #f96; font-size: 9px; margin-bottom: 4px;">`;
        html += `CGAL not available - results may be incomplete`;
        html += `</div>`;
    }

    if (num_intersecting_faces > 0) {
        html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
        html += `${num_intersection_pairs} intersection pair(s) detected`;
        html += `</div>`;

        html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
        html += buildTableHeader([
            { label: 'Face', align: 'left' },
            { label: 'Vertices', align: 'left' }
        ]);

        const displayFaces = faces.slice(0, maxDisplay);
        for (const face of displayFaces) {
            html += buildTableRow([
                { value: face.id, color: '#f66', align: 'left' },
                { value: `[${face.vertices.join(', ')}]`, color: '#888', align: 'left' }
            ]);
        }

        if (faces.length > maxDisplay) {
            html += buildMoreRow(faces.length - maxDisplay, 2, "faces");
        }

        html += '</table>';
    } else {
        html += `<div style="color: #888; font-size: 9px;">`;
        html += `Total: ${total_vertices.toLocaleString()} vertices, ${total_faces.toLocaleString()} faces`;
        html += `</div>`;
    }

    html += '</div>';
    return html;
}

/**
 * Build degenerate faces HTML
 * @param {Object} data - Degenerate faces data
 * @param {number} maxDisplay - Max faces to display (default: 20)
 * @returns {string} HTML string
 */
export function buildDegenerateFacesHTML(data, maxDisplay = 20) {
    const { mesh_name, num_degenerate, total_faces, total_vertices, faces } = data;

    let html = '<div style="margin-bottom: 8px;">';

    const statusColor = num_degenerate === 0 ? '#6f6' : '#f66';
    const statusText = num_degenerate === 0 ? 'Clean' : `${num_degenerate} degenerate face(s)`;

    html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
    html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
    html += `</div>`;

    if (num_degenerate > 0 && faces && faces.length > 0) {
        html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
        html += buildTableHeader([
            { label: 'Face', align: 'left' },
            { label: 'Type', align: 'left' },
            { label: 'Vertices', align: 'left' }
        ]);

        const displayFaces = faces.slice(0, maxDisplay);
        for (const face of displayFaces) {
            html += buildTableRow([
                { value: face.id, color: '#f66', align: 'left' },
                { value: face.type || 'degenerate', color: '#888', align: 'left' },
                { value: `[${face.vertices.join(', ')}]`, color: '#888', align: 'left' }
            ]);
        }

        if (faces.length > maxDisplay) {
            html += buildMoreRow(faces.length - maxDisplay, 3, "faces");
        }

        html += '</table>';
    } else {
        html += `<div style="color: #888; font-size: 9px;">`;
        html += `Total: ${total_vertices.toLocaleString()} vertices, ${total_faces.toLocaleString()} faces`;
        html += `</div>`;
    }

    html += '</div>';
    return html;
}

/**
 * Build open edges HTML
 * @param {Object} data - Open edges data
 * @param {number} maxDisplay - Max edges to display (default: 20)
 * @returns {string} HTML string
 */
export function buildOpenEdgesHTML(data, maxDisplay = 20) {
    const { mesh_name, num_open_edges, total_faces, total_vertices, edges } = data;

    let html = '<div style="margin-bottom: 8px;">';

    const statusColor = num_open_edges === 0 ? '#6f6' : '#fc6';
    const statusText = num_open_edges === 0 ? 'Watertight' : `${num_open_edges} open edge(s)`;

    html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
    html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
    html += `</div>`;

    if (num_open_edges > 0 && edges && edges.length > 0) {
        html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
        html += buildTableHeader([
            { label: 'Edge', align: 'left' },
            { label: 'Vertices', align: 'left' }
        ]);

        const displayEdges = edges.slice(0, maxDisplay);
        for (let i = 0; i < displayEdges.length; i++) {
            const edge = displayEdges[i];
            html += buildTableRow([
                { value: i, color: '#fc6', align: 'left' },
                { value: `[${edge.vertices.join(', ')}]`, color: '#888', align: 'left' }
            ]);
        }

        if (edges.length > maxDisplay) {
            html += buildMoreRow(edges.length - maxDisplay, 2, "edges");
        }

        html += '</table>';
    } else {
        html += `<div style="color: #888; font-size: 9px;">`;
        html += `Total: ${total_vertices.toLocaleString()} vertices, ${total_faces.toLocaleString()} faces`;
        html += `</div>`;
    }

    html += '</div>';
    return html;
}

/**
 * Calculate dynamic height for analysis panel based on content
 * @param {number} numRows - Number of data rows
 * @param {number} maxRows - Maximum rows before truncation (default: 20)
 * @param {number} baseHeight - Base height for headers (default: 40)
 * @param {number} rowHeight - Height per row (default: 16)
 * @param {number} minHeight - Minimum panel height (default: 80)
 * @param {number} maxHeight - Maximum panel height (default: 250)
 * @returns {number} Panel height in pixels
 */
export function calculatePanelHeight(numRows, maxRows = 20, baseHeight = 40, rowHeight = 16, minHeight = 80, maxHeight = 250) {
    const displayRows = Math.min(numRows, maxRows) + 3; // +3 for header rows
    return Math.min(Math.max(minHeight, displayRows * rowHeight + baseHeight), maxHeight);
}

/**
 * Build text report HTML (for nodes that output plain text info/reports)
 * @param {string} text - Plain text report
 * @returns {string} HTML string with styling
 */
export function buildTextReportHTML(text) {
    // Convert newlines to <br>, escape HTML
    const escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');

    return `<div style="white-space: pre-wrap; font-size: 10px; line-height: 1.4;">${escaped}</div>`;
}
