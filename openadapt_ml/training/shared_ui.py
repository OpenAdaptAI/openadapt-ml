"""Shared UI components for dashboards and viewers.

This module contains CSS and HTML generation functions used by both
the Training Dashboard and the Viewer for visual consistency.
"""

from __future__ import annotations


def get_shared_header_css() -> str:
    """Generate CSS for the shared dashboard header.

    This CSS is used by both the Training Dashboard and the Viewer.
    Any changes here will affect all dashboards consistently.
    """
    return '''
    .unified-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: linear-gradient(180deg, rgba(18,18,26,0.98) 0%, rgba(26,26,36,0.98) 100%);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 20px;
        gap: 16px;
        flex-wrap: wrap;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .unified-header .nav-tabs {
        display: flex;
        align-items: center;
        gap: 4px;
        background: rgba(0,0,0,0.3);
        padding: 4px;
        border-radius: 8px;
    }
    .unified-header .nav-tab {
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        text-decoration: none;
        color: var(--text-secondary);
        background: transparent;
        border: none;
        transition: all 0.2s;
        cursor: pointer;
    }
    .unified-header .nav-tab:hover {
        color: var(--text-primary);
        background: rgba(255,255,255,0.05);
    }
    .unified-header .nav-tab.active {
        color: var(--bg-primary);
        background: var(--accent);
        font-weight: 600;
    }
    .unified-header .controls-section {
        display: flex;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    }
    .unified-header .control-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .unified-header .control-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .unified-header select {
        padding: 8px 32px 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        background: rgba(0,0,0,0.4);
        color: var(--text-primary);
        border: 1px solid rgba(255,255,255,0.1);
        cursor: pointer;
        appearance: none;
        background-image: url('data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2712%27 height=%278%27%3E%3Cpath fill=%27%23888%27 d=%27M0 0l6 8 6-8z%27/%3E%3C/svg%3E');
        background-repeat: no-repeat;
        background-position: right 10px center;
        transition: all 0.2s;
    }
    .unified-header select:hover {
        border-color: var(--accent);
        background-color: rgba(0,212,170,0.1);
    }
    .unified-header select:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(0,212,170,0.2);
    }
    .unified-header .header-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: "SF Mono", Monaco, monospace;
    }
    '''


def generate_shared_header_html(
    active_page: str,
    controls_html: str = "",
    meta_html: str = "",
) -> str:
    """Generate the shared header HTML.

    Args:
        active_page: Either "training", "viewer", or "benchmarks" to highlight the active tab
        controls_html: Optional HTML for control groups (dropdowns, etc.)
        meta_html: Optional HTML for metadata display (job ID, capture ID, etc.)

    Returns:
        HTML string for the header
    """
    training_active = "active" if active_page == "training" else ""
    viewer_active = "active" if active_page == "viewer" else ""
    benchmarks_active = "active" if active_page == "benchmarks" else ""

    controls_section = ""
    if controls_html or meta_html:
        controls_section = f'''
        <div class="controls-section">
            {controls_html}
            {f'<span class="header-meta">{meta_html}</span>' if meta_html else ''}
        </div>
        '''

    return f'''
    <div class="unified-header">
        <div class="nav-tabs">
            <a href="dashboard.html" class="nav-tab {training_active}">Training</a>
            <a href="viewer.html" class="nav-tab {viewer_active}">Viewer</a>
            <a href="benchmark.html" class="nav-tab {benchmarks_active}">Benchmarks</a>
        </div>
        {controls_section}
    </div>
    '''


def build_nav_links() -> list[tuple[str, str]]:
    """Build navigation links for multi-capture dashboards.

    Returns:
        List of (filename, label) tuples
    """
    return [
        ("dashboard.html", "Training"),
        ("viewer.html", "Viewer"),
        ("benchmark.html", "Benchmarks"),
    ]
