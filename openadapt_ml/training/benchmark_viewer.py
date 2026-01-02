"""Benchmark viewer generation functions.

This module provides functions to generate HTML viewers for benchmark evaluation results.
It is imported and used by trainer.py to maintain consistency with other viewer components.
"""

from __future__ import annotations

import json
from pathlib import Path


def _get_background_tasks_panel_css() -> str:
    """Return CSS for background tasks panel."""
    return '''
        .tasks-panel {
            background: linear-gradient(135deg, rgba(100, 100, 255, 0.1) 0%, rgba(100, 100, 255, 0.05) 100%);
            border: 1px solid rgba(100, 100, 255, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .tasks-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .tasks-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            font-weight: 600;
            color: #6366f1;
        }
        .tasks-title svg {
            width: 20px;
            height: 20px;
        }
        .tasks-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .task-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .task-card:last-child {
            margin-bottom: 0;
        }
        .task-card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .task-status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .task-status-indicator.running {
            background: #3b82f6;
            animation: pulse-task 2s infinite;
        }
        .task-status-indicator.completed {
            background: #10b981;
        }
        .task-status-indicator.failed {
            background: #ef4444;
        }
        .task-status-indicator.pending {
            background: #f59e0b;
        }
        @keyframes pulse-task {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5); }
            50% { opacity: 0.8; box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
        }
        .task-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-primary);
        }
        .task-description {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }
        .task-progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .task-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .task-progress-fill.completed {
            background: linear-gradient(90deg, #10b981, #059669);
        }
        .task-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .task-link {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            background: rgba(99, 102, 241, 0.2);
            border: 1px solid rgba(99, 102, 241, 0.4);
            border-radius: 4px;
            color: #818cf8;
            text-decoration: none;
            font-size: 0.75rem;
            margin-top: 8px;
            transition: all 0.2s;
        }
        .task-link:hover {
            background: rgba(99, 102, 241, 0.3);
            transform: translateY(-1px);
        }
        .task-credentials {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 6px;
            margin: 8px 0;
            font-size: 0.85rem;
        }
        .task-credentials .cred-label {
            color: #fbbf24;
        }
        .task-credentials code {
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            color: #fcd34d;
        }
        .no-tasks {
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .task-phase-badge {
            margin-left: auto;
            padding: 2px 8px;
            background: rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            font-size: 0.75rem;
            color: #a5b4fc;
        }
        .task-logs-details {
            margin-top: 12px;
            border-top: 1px solid var(--border-color);
            padding-top: 8px;
        }
        .task-logs-summary {
            cursor: pointer;
            font-size: 0.75rem;
            color: var(--text-muted);
            user-select: none;
        }
        .task-logs-summary:hover {
            color: var(--text-secondary);
        }
        .task-logs-content {
            margin-top: 8px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 4px;
            font-size: 0.7rem;
            line-height: 1.4;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
            color: #10b981;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        }
        .vm-details-section {
            margin-top: 12px;
            border-top: 1px solid var(--border-color);
            padding-top: 12px;
        }
        .vm-details-toggle {
            cursor: pointer;
            font-size: 0.75rem;
            color: var(--text-muted);
            user-select: none;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 0;
        }
        .vm-details-toggle:hover {
            color: var(--text-secondary);
        }
        .vm-details-toggle-icon {
            transition: transform 0.2s;
        }
        .vm-details-toggle.expanded .vm-details-toggle-icon {
            transform: rotate(90deg);
        }
        .vm-details-content {
            display: none;
            margin-top: 8px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            font-size: 0.75rem;
        }
        .vm-details-toggle.expanded + .vm-details-content {
            display: block;
        }
        .vm-detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .vm-detail-row:last-child {
            border-bottom: none;
        }
        .vm-detail-label {
            color: var(--text-muted);
            font-weight: 500;
        }
        .vm-detail-value {
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, monospace;
        }
        .vm-detail-value.success {
            color: #10b981;
        }
        .vm-detail-value.warning {
            color: #f59e0b;
        }
        .vm-detail-value.error {
            color: #ef4444;
        }
        .vm-dependencies-list {
            margin-top: 8px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }
        .vm-dependency-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 0.7rem;
        }
        .vm-dependency-icon {
            font-size: 1rem;
        }
        .vm-progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin: 8px 0;
        }
        .vm-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            border-radius: 3px;
            transition: width 0.5s ease;
        }
    '''


def _get_background_tasks_panel_html() -> str:
    """Return HTML for background tasks panel with JS polling and improved styling."""
    return '''
    <div class="tasks-panel" id="tasks-panel">
        <div class="tasks-header">
            <div class="tasks-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                </svg>
                Background Tasks
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="tasks-refresh" id="tasks-refresh-time">Checking...</span>
                <button class="refresh-btn" onclick="refreshBackgroundTasks()" title="Refresh tasks" id="tasks-refresh-btn" style="background: rgba(99, 102, 241, 0.2); border-color: rgba(99, 102, 241, 0.4);">
                    <span class="refresh-icon">&#8635;</span>
                    <span class="spinner" style="border-top-color: #6366f1;"></span>
                    Refresh
                </button>
            </div>
        </div>

        <!-- API Error Banner -->
        <div class="api-error-banner" id="tasks-api-error" style="display: none;">
            <span class="error-icon">!</span>
            <span class="error-message" id="tasks-error-msg">Failed to fetch tasks</span>
            <button class="retry-btn" onclick="refreshBackgroundTasks()">Retry</button>
        </div>

        <!-- Loading state -->
        <div id="tasks-loading" style="display: none; text-align: center; padding: 30px;">
            <div style="display: inline-block; width: 24px; height: 24px; border: 3px solid rgba(99,102,241,0.3); border-top-color: #6366f1; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="margin-top: 12px; color: var(--text-muted); font-size: 0.85rem;">Loading tasks...</div>
        </div>

        <div id="tasks-list">
            <div class="no-tasks">
                <div style="font-size: 2rem; margin-bottom: 12px; opacity: 0.5;">&#128203;</div>
                Checking for active tasks...
            </div>
        </div>
    </div>

    <script>
        let isTasksRefreshing = false;
        let tasksErrorCount = 0;

        function setTasksLoadingState(loading) {
            const loadingEl = document.getElementById('tasks-loading');
            const listEl = document.getElementById('tasks-list');
            const btn = document.getElementById('tasks-refresh-btn');

            if (loading) {
                loadingEl.style.display = 'block';
                listEl.style.display = 'none';
                if (btn) btn.classList.add('loading');
            } else {
                loadingEl.style.display = 'none';
                listEl.style.display = 'block';
                if (btn) btn.classList.remove('loading');
            }
        }

        function showTasksError(msg) {
            const errorEl = document.getElementById('tasks-api-error');
            const errorMsgEl = document.getElementById('tasks-error-msg');
            if (errorEl && errorMsgEl) {
                errorMsgEl.textContent = msg;
                errorEl.style.display = 'flex';
            }
        }

        function hideTasksError() {
            const errorEl = document.getElementById('tasks-api-error');
            if (errorEl) errorEl.style.display = 'none';
        }

        async function refreshBackgroundTasks() {
            if (isTasksRefreshing) return;
            isTasksRefreshing = true;
            setTasksLoadingState(true);
            hideTasksError();

            try {
                const response = await fetch('/api/tasks?' + Date.now());
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const tasks = await response.json();
                if (tasks.error) throw new Error(tasks.error);

                renderBackgroundTasks(tasks);
                tasksErrorCount = 0;
                document.getElementById('tasks-refresh-time').textContent =
                    'Updated ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Tasks refresh failed:', e);
                tasksErrorCount++;
                showTasksError(e.message || 'Connection failed');
            } finally {
                isTasksRefreshing = false;
                setTasksLoadingState(false);
            }
        }

        async function fetchBackgroundTasks() {
            if (isTasksRefreshing) return;
            if (tasksErrorCount >= 3) {
                document.getElementById('tasks-refresh-time').textContent = 'Polling paused';
                return;
            }

            try {
                const response = await fetch('/api/tasks?' + Date.now());
                if (response.ok) {
                    const tasks = await response.json();
                    if (!tasks.error) {
                        renderBackgroundTasks(tasks);
                        hideTasksError();
                        tasksErrorCount = 0;
                        document.getElementById('tasks-refresh-time').textContent =
                            'Updated ' + new Date().toLocaleTimeString();
                    }
                }
            } catch (e) {
                console.log('Tasks API unavailable:', e);
                tasksErrorCount++;
            }
        }

        function renderVMDetails(metadata) {
            if (!metadata) return '';

            const statusClass = (value, type = 'default') => {
                if (type === 'probe') {
                    return value && value !== 'Not responding' && value !== 'Connection failed' ? 'success' : 'error';
                } else if (type === 'qmp') {
                    return value ? 'success' : 'warning';
                }
                return '';
            };

            const renderDependencies = (deps) => {
                if (!deps || deps.length === 0) return '';

                const statusIcons = {
                    'complete': '✓',
                    'installing': '⏳',
                    'pending': '○'
                };

                return `
                    <div class="vm-detail-row">
                        <div class="vm-detail-label">Dependencies</div>
                    </div>
                    <div class="vm-dependencies-list">
                        ${deps.map(dep => `
                            <div class="vm-dependency-item">
                                <span class="vm-dependency-icon">${dep.icon || '📦'}</span>
                                <span>${statusIcons[dep.status] || '○'} ${dep.name}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            };

            return `
                <div class="vm-details-section">
                    <div class="vm-details-toggle" onclick="this.classList.toggle('expanded')">
                        <span class="vm-details-toggle-icon">▶</span>
                        <span>VM Details</span>
                    </div>
                    <div class="vm-details-content">
                        ${metadata.setup_script_phase ? `
                            <div class="vm-detail-row">
                                <div class="vm-detail-label">Setup Phase</div>
                                <div class="vm-detail-value">${metadata.setup_script_phase}</div>
                            </div>
                        ` : ''}
                        ${metadata.disk_usage_gb ? `
                            <div class="vm-detail-row">
                                <div class="vm-detail-label">Disk Usage</div>
                                <div class="vm-detail-value">${metadata.disk_usage_gb}</div>
                            </div>
                        ` : ''}
                        ${metadata.memory_usage_mb ? `
                            <div class="vm-detail-row">
                                <div class="vm-detail-label">Memory Usage</div>
                                <div class="vm-detail-value">${metadata.memory_usage_mb}</div>
                            </div>
                        ` : ''}
                        ${metadata.probe_response !== undefined ? `
                            <div class="vm-detail-row">
                                <div class="vm-detail-label">WAA Server (/probe)</div>
                                <div class="vm-detail-value ${statusClass(metadata.probe_response, 'probe')}">
                                    ${metadata.probe_response}
                                </div>
                            </div>
                        ` : ''}
                        ${metadata.qmp_connected !== undefined ? `
                            <div class="vm-detail-row">
                                <div class="vm-detail-label">QMP (port 7200)</div>
                                <div class="vm-detail-value ${statusClass(metadata.qmp_connected, 'qmp')}">
                                    ${metadata.qmp_connected ? 'Connected ✓' : 'Not connected'}
                                </div>
                            </div>
                        ` : ''}
                        ${renderDependencies(metadata.dependencies)}
                    </div>
                </div>
            `;
        }

        function renderBackgroundTasks(tasks) {
            const container = document.getElementById('tasks-list');

            // Debug: Log incoming tasks data
            console.log('[SSE Debug] renderBackgroundTasks called with:', JSON.stringify(tasks, null, 2));

            if (!tasks || tasks.length === 0) {
                container.innerHTML = '<div class="no-tasks">No active background tasks</div>';
                return;
            }

            const phaseLabels = {
                'downloading': '⬇️ Downloading',
                'extracting': '📦 Extracting',
                'configuring': '⚙️ Configuring',
                'building': '🔨 Building',
                'booting': '🚀 Booting',
                'oobe': '🪟 Windows Setup',
                'ready': '✅ Ready',
                'unknown': '⏳ Starting'
            };

            const html = tasks.map(task => {
                const statusClass = task.status || 'pending';
                const progressPercent = task.progress_percent || 0;
                const progressClass = task.status === 'completed' ? 'completed' : '';

                // Determine phase: use task.phase, fall back to metadata.phase,
                // then if status is 'completed' use 'ready', otherwise 'unknown'
                let phase = task.phase || task.metadata?.phase;
                if (!phase) {
                    // If no phase specified, infer from status to prevent "Starting" + "completed" conflict
                    phase = (task.status === 'completed') ? 'ready' : 'unknown';
                }
                const phaseLabel = phaseLabels[phase] || phase;

                // Debug: Log per-task phase/status mapping
                console.log(`[SSE Debug] Task ${task.task_id}: status=${task.status}, phase=${task.phase}, resolvedPhase=${phase}, phaseLabel=${phaseLabel}`);

                // Build link if VNC URL available
                let linkHtml = '';
                if (task.metadata && task.metadata.vnc_url) {
                    linkHtml = `<a href="${task.metadata.vnc_url}" target="_blank" class="task-link">
                        Open VNC →
                    </a>`;
                }

                // Show Windows credentials if available
                let credentialsHtml = '';
                if (task.metadata && task.metadata.windows_username) {
                    credentialsHtml = `
                        <div class="task-credentials">
                            <span class="cred-label">🔑 Login:</span>
                            <code>${task.metadata.windows_username}</code> /
                            <code>${task.metadata.windows_password || '(empty)'}</code>
                        </div>
                    `;
                }

                // Add expandable logs if available
                let logsHtml = '';
                if (task.metadata && task.metadata.recent_logs) {
                    const taskId = task.task_id.replace(/[^a-z0-9]/gi, '_');
                    logsHtml = `
                        <details class="task-logs-details">
                            <summary class="task-logs-summary">Show recent logs</summary>
                            <pre class="task-logs-content">${task.metadata.recent_logs.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
                        </details>
                    `;
                }

                // Add VM Details expandable section for Windows containers
                let vmDetailsHtml = '';
                if (task.task_type === 'docker_container' && task.metadata) {
                    vmDetailsHtml = renderVMDetails(task.metadata);
                }

                // Progress label clarifies what % means
                // Use a single unified status display to avoid showing conflicting states
                let progressLabel;
                if (task.status === 'completed' || phase === 'ready') {
                    progressLabel = 'Complete';
                } else {
                    progressLabel = `Setup phase progress: ${progressPercent.toFixed(0)}%`;
                }

                return `
                    <div class="task-card">
                        <div class="task-card-header">
                            <div class="task-status-indicator ${statusClass}"></div>
                            <span class="task-title">${task.title || 'Unknown Task'}</span>
                            <span class="task-phase-badge">${phaseLabel}</span>
                        </div>
                        <div class="task-description">${task.description || ''}</div>
                        <div class="task-progress-bar">
                            <div class="task-progress-fill ${progressClass}" style="width: ${progressPercent}%"></div>
                        </div>
                        <div class="task-meta">
                            <span>${progressLabel}</span>
                        </div>
                        ${credentialsHtml}
                        ${linkHtml}
                        ${vmDetailsHtml}
                        ${logsHtml}
                    </div>
                `;
            }).join('');

            container.innerHTML = html;
        }

        // Initial fetch and poll every 10 seconds
        fetchBackgroundTasks();
        setInterval(fetchBackgroundTasks, 10000);
    </script>
    '''


def _get_live_evaluation_panel_css() -> str:
    """Return CSS for live evaluation progress panel."""
    return '''
        .live-eval-panel {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(139, 92, 246, 0.05) 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .live-eval-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .live-eval-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            font-weight: 600;
            color: #8b5cf6;
        }
        .live-eval-title svg {
            width: 20px;
            height: 20px;
        }
        .live-eval-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .live-eval-status {
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .live-eval-progress {
            font-size: 0.95rem;
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: 8px;
        }
        .live-eval-task-name {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .live-eval-step {
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .live-eval-step-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .live-eval-step-number {
            font-weight: 600;
            color: var(--accent);
            min-width: 60px;
        }
        .live-eval-action {
            flex: 1;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.85rem;
            color: var(--text-primary);
        }
        .live-eval-screenshot {
            max-width: 300px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            margin: 8px 0;
        }
        .live-eval-reasoning {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-style: italic;
            margin-top: 8px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }
        .live-eval-result {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .live-eval-result.success {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }
        .live-eval-result.failure {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        .live-eval-idle {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .live-eval-steps-container {
            max-height: 400px;
            overflow-y: auto;
        }
        /* SSE Connection Status Indicator */
        .sse-connection-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 12px;
        }
        .sse-connection-status.connected {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }
        .sse-connection-status.connecting {
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
        }
        .sse-connection-status.disconnected {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        .sse-connection-status.fallback {
            background: rgba(156, 163, 175, 0.2);
            color: #9ca3af;
        }
        .sse-connection-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }
        .sse-connection-status.connecting .sse-connection-dot {
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    '''


def _get_live_evaluation_panel_html() -> str:
    """Return HTML for live evaluation panel with SSE and polling fallback."""
    return '''
    <div class="live-eval-panel" id="live-eval-panel">
        <div class="live-eval-header">
            <div class="live-eval-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
                </svg>
                Live Evaluation
                <span class="sse-connection-status connecting" id="sse-status">
                    <span class="sse-connection-dot"></span>
                    <span id="sse-status-text">Connecting</span>
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="live-eval-refresh" id="live-eval-refresh-time">Checking...</span>
                <button class="refresh-btn" onclick="if(window.sseManager) { window.sseManager.disconnect(); window.sseManager.connect(); }" title="Reconnect to live updates" style="background: rgba(245, 158, 11, 0.2); border-color: rgba(245, 158, 11, 0.4);">
                    <span class="refresh-icon">&#8635;</span>
                    <span class="spinner" style="border-top-color: #f59e0b;"></span>
                    Reconnect
                </button>
            </div>
        </div>
        <div id="live-eval-content">
            <div class="live-eval-idle">
                <div style="font-size: 2rem; margin-bottom: 12px; opacity: 0.5;">&#9889;</div>
                No evaluation running
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 8px;">
                    Start an evaluation to see real-time progress
                </div>
            </div>
        </div>
    </div>

    <script>
        // SSE Manager for real-time benchmark updates
        class BenchmarkSSEManager {
            constructor() {
                this.eventSource = null;
                this.pollingInterval = null;
                this.staleCheckInterval = null;  // Track stale connection check interval
                this.usePolling = false;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 2000;
                this.lastHeartbeat = Date.now();
                this.state = {
                    status: 'idle',
                    tasks_completed: 0,
                    total_tasks: 0,
                    current_task: null,
                    results: []
                };
            }

            // Clear all intervals to prevent memory leaks
            clearAllIntervals() {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                }
                if (this.staleCheckInterval) {
                    clearInterval(this.staleCheckInterval);
                    this.staleCheckInterval = null;
                }
            }

            connect() {
                // Check if EventSource is supported
                if (!window.EventSource) {
                    console.log('SSE not supported, falling back to polling');
                    this.startPolling();
                    return;
                }

                // Clear any existing intervals before reconnecting
                this.clearAllIntervals();

                this.updateConnectionStatus('connecting');

                try {
                    this.eventSource = new EventSource('/api/benchmark-sse?interval=2');

                    this.eventSource.addEventListener('connected', (e) => {
                        console.log('SSE connected:', e.data);
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus('connected');
                    });

                    this.eventSource.addEventListener('status', (e) => {
                        const data = JSON.parse(e.data);
                        this.handleStatusEvent(data);
                        this.updateTimestamp();
                    });

                    this.eventSource.addEventListener('progress', (e) => {
                        const data = JSON.parse(e.data);
                        this.handleProgressEvent(data);
                        this.updateTimestamp();
                    });

                    this.eventSource.addEventListener('task_complete', (e) => {
                        const data = JSON.parse(e.data);
                        this.handleTaskCompleteEvent(data);
                        this.updateTimestamp();
                    });

                    this.eventSource.addEventListener('heartbeat', (e) => {
                        this.lastHeartbeat = Date.now();
                        // Heartbeats keep connection alive, no UI update needed
                    });

                    this.eventSource.addEventListener('error', (e) => {
                        const data = JSON.parse(e.data);
                        console.error('SSE error event:', data);
                    });

                    this.eventSource.onerror = (e) => {
                        console.error('SSE connection error:', e);
                        this.handleConnectionError();
                    };

                    // Check for stale connection (no heartbeat in 60 seconds)
                    // Store interval ID to clear on reconnect
                    this.staleCheckInterval = setInterval(() => {
                        if (this.eventSource && (Date.now() - this.lastHeartbeat > 60000)) {
                            console.log('SSE connection stale, reconnecting...');
                            this.reconnect();
                        }
                    }, 30000);

                } catch (e) {
                    console.error('SSE connection failed:', e);
                    this.startPolling();
                }
            }

            handleStatusEvent(data) {
                console.log('[SSE Debug] handleStatusEvent:', JSON.stringify(data));
                // Clear previous vmStatus to prevent stale state accumulation
                this.state.vmStatus = data;
                if (data.waa_ready) {
                    this.state.status = 'ready';
                }
                console.log('[SSE Debug] Updated state after status event:', JSON.stringify(this.state));
                this.render();
            }

            handleProgressEvent(data) {
                console.log('[SSE Debug] handleProgressEvent:', JSON.stringify(data));
                this.state.status = 'running';
                this.state.tasks_completed = data.tasks_completed;
                this.state.total_tasks = data.total_tasks;
                this.state.current_task = {
                    task_id: data.current_task,
                    instruction: `Task ${data.current_task}`,
                    domain: 'waa'
                };
                console.log('[SSE Debug] Updated state after progress event:', JSON.stringify(this.state));
                this.render();
            }

            handleTaskCompleteEvent(data) {
                this.state.results.push({
                    task_id: data.task_id,
                    success: data.success,
                    score: data.score
                });
                this.render();
            }

            handleConnectionError() {
                this.updateConnectionStatus('disconnected');

                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`SSE reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                    setTimeout(() => this.reconnect(), this.reconnectDelay * this.reconnectAttempts);
                } else {
                    console.log('Max SSE reconnect attempts reached, falling back to polling');
                    this.startPolling();
                }
            }

            reconnect() {
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                this.connect();
            }

            startPolling() {
                this.usePolling = true;
                this.updateConnectionStatus('fallback');

                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }

                // Clear any existing intervals before starting new polling
                this.clearAllIntervals();

                // Use existing polling function
                fetchLiveEvaluationPolling();
                this.pollingInterval = setInterval(fetchLiveEvaluationPolling, 2000);
            }

            updateConnectionStatus(status) {
                const el = document.getElementById('sse-status');
                const textEl = document.getElementById('sse-status-text');
                if (!el || !textEl) return;

                el.className = 'sse-connection-status ' + status;
                const statusText = {
                    'connected': 'Live',
                    'connecting': 'Connecting',
                    'disconnected': 'Disconnected',
                    'fallback': 'Polling'
                };
                textEl.textContent = statusText[status] || status;
            }

            updateTimestamp() {
                const el = document.getElementById('live-eval-refresh-time');
                if (el) {
                    el.textContent = 'Updated ' + new Date().toLocaleTimeString();
                }
            }

            render() {
                renderLiveEvaluation(this.state);
            }

            disconnect() {
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                // Clear all intervals using centralized cleanup
                this.clearAllIntervals();
            }
        }

        // Polling fallback function
        async function fetchLiveEvaluationPolling() {
            try {
                const response = await fetch('/api/benchmark-live?' + Date.now());
                if (response.ok) {
                    const state = await response.json();
                    console.log('[SSE Debug] Polling received state:', JSON.stringify(state));
                    renderLiveEvaluation(state);
                    document.getElementById('live-eval-refresh-time').textContent =
                        'Updated ' + new Date().toLocaleTimeString();
                }
            } catch (e) {
                console.log('Live evaluation API unavailable:', e);
                document.getElementById('live-eval-content').innerHTML =
                    '<div class="live-eval-idle">Live evaluation API not available</div>';
            }
        }

        function renderLiveEvaluation(state) {
            const container = document.getElementById('live-eval-content');

            if (!state || state.status === 'idle' || !state.current_task) {
                container.innerHTML = '<div class="live-eval-idle">No evaluation running</div>';
                return;
            }

            const task = state.current_task;
            const progress = `${state.tasks_completed || 0}/${state.total_tasks || 0}`;

            // Build status section
            let statusHtml = `
                <div class="live-eval-status">
                    <div class="live-eval-progress">Evaluating task ${progress}: ${task.task_id}</div>
                    <div class="live-eval-task-name">${task.instruction || 'No instruction'}</div>
                    <div class="live-eval-task-name">Domain: ${task.domain || 'unknown'}</div>
                </div>
            `;

            // Build steps section
            let stepsHtml = '';
            if (task.steps && task.steps.length > 0) {
                stepsHtml = '<div class="live-eval-steps-container">';

                // Show last 5 steps
                const recentSteps = task.steps.slice(-5);
                recentSteps.forEach(step => {
                    const actionText = formatAction(step.action);
                    const screenshotHtml = step.screenshot_url
                        ? `<img src="${step.screenshot_url}" class="live-eval-screenshot" alt="Step ${step.step_idx}" />`
                        : '';
                    const reasoningHtml = step.reasoning
                        ? `<div class="live-eval-reasoning">"${step.reasoning}"</div>`
                        : '';

                    stepsHtml += `
                        <div class="live-eval-step">
                            <div class="live-eval-step-header">
                                <div class="live-eval-step-number">Step ${step.step_idx}</div>
                                <div class="live-eval-action">${actionText}</div>
                            </div>
                            ${screenshotHtml}
                            ${reasoningHtml}
                        </div>
                    `;
                });

                stepsHtml += '</div>';
            }

            // Show result if task completed
            let resultHtml = '';
            if (task.result) {
                const resultClass = task.result.success ? 'success' : 'failure';
                const resultIcon = task.result.success ? '✓' : '✗';
                resultHtml = `
                    <div class="live-eval-status">
                        <div class="live-eval-result ${resultClass}">
                            ${resultIcon} ${task.result.success ? 'Success' : 'Failure'}
                            (${task.result.num_steps} steps in ${task.result.total_time_seconds.toFixed(2)}s)
                        </div>
                    </div>
                `;
            }

            // Show recent results summary
            if (state.results && state.results.length > 0) {
                const successCount = state.results.filter(r => r.success).length;
                resultHtml += `
                    <div class="live-eval-status" style="margin-top: 8px;">
                        <small>Results: ${successCount}/${state.results.length} passed</small>
                    </div>
                `;
            }

            container.innerHTML = statusHtml + stepsHtml + resultHtml;
        }

        function formatAction(action) {
            if (!action) return 'No action';

            const type = action.type || 'unknown';
            const parts = [type.toUpperCase()];

            if (action.x !== null && action.y !== null) {
                parts.push(`(x=${action.x.toFixed(3)}, y=${action.y.toFixed(3)})`);
            } else if (action.target_node_id) {
                parts.push(`[${action.target_node_id}]`);
            }

            if (action.text) {
                parts.push(`"${action.text}"`);
            }

            if (action.key) {
                parts.push(`key=${action.key}`);
            }

            return parts.join(' ');
        }

        // Initialize SSE manager and store on window for reconnect button
        window.sseManager = new BenchmarkSSEManager();
        window.sseManager.connect();

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (window.sseManager) window.sseManager.disconnect();
        });
    </script>
    '''


def _get_azure_jobs_panel_css() -> str:
    """Return CSS for the Azure jobs status panel with color-coded status indicators."""
    return '''
        .azure-jobs-panel {
            background: linear-gradient(135deg, rgba(0, 120, 212, 0.15) 0%, rgba(0, 120, 212, 0.05) 100%);
            border: 1px solid rgba(0, 120, 212, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .azure-jobs-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .azure-jobs-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            font-weight: 600;
            color: #0078d4;
        }
        .azure-jobs-title svg {
            width: 20px;
            height: 20px;
        }
        .azure-jobs-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .azure-jobs-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
            transition: color 0.2s;
        }
        .azure-jobs-refresh.error {
            color: #ef4444;
        }
        .azure-jobs-refresh.success {
            color: #10b981;
        }
        /* API Error Banner */
        .api-error-banner {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.4);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: none;
            align-items: center;
            gap: 12px;
            font-size: 0.85rem;
            color: #fca5a5;
        }
        .api-error-banner.show {
            display: flex;
        }
        .api-error-banner .error-icon {
            font-size: 1.2rem;
            flex-shrink: 0;
        }
        .api-error-banner .error-message {
            flex: 1;
        }
        .api-error-banner .retry-btn {
            padding: 4px 10px;
            background: rgba(239, 68, 68, 0.3);
            border: 1px solid rgba(239, 68, 68, 0.5);
            border-radius: 4px;
            color: #fca5a5;
            cursor: pointer;
            font-size: 0.75rem;
            transition: background 0.2s;
        }
        .api-error-banner .retry-btn:hover {
            background: rgba(239, 68, 68, 0.4);
        }
        /* Job items with color-coded borders */
        .azure-job-item {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 14px 18px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid transparent;
            transition: all 0.2s ease;
        }
        .azure-job-item:last-child {
            margin-bottom: 0;
        }
        .azure-job-item:hover {
            background: rgba(0, 0, 0, 0.4);
        }
        /* Color-coded left border based on status - Running=Yellow, Completed=Green, Failed=Red */
        .azure-job-item.status-running {
            border-left-color: #f59e0b;
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, rgba(0, 0, 0, 0.3) 20%);
        }
        .azure-job-item.status-completed {
            border-left-color: #10b981;
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(0, 0, 0, 0.3) 20%);
        }
        .azure-job-item.status-failed,
        .azure-job-item.status-canceled {
            border-left-color: #ef4444;
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(0, 0, 0, 0.3) 20%);
        }
        .azure-job-item.status-provisioning,
        .azure-job-item.status-preparing,
        .azure-job-item.status-queued,
        .azure-job-item.status-starting {
            border-left-color: #3b82f6;
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(0, 0, 0, 0.3) 20%);
        }
        .azure-job-status {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 130px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .status-dot.provisioning,
        .status-dot.preparing,
        .status-dot.queued,
        .status-dot.starting {
            background: #3b82f6;
            animation: pulse-status 2s infinite;
        }
        .status-dot.running {
            background: #f59e0b;
            animation: pulse-status 1.5s infinite;
        }
        .status-dot.completed {
            background: #10b981;
            animation: none;
        }
        .status-dot.failed,
        .status-dot.canceled {
            background: #ef4444;
            animation: none;
        }
        .status-dot.unknown {
            background: #6b7280;
            animation: none;
        }
        @keyframes pulse-status {
            0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 currentColor; }
            50% { opacity: 0.6; transform: scale(0.9); }
        }
        .status-text {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-text.running { color: #f59e0b; }
        .status-text.completed { color: #10b981; }
        .status-text.failed, .status-text.canceled { color: #ef4444; }
        .status-text.provisioning, .status-text.preparing, .status-text.queued, .status-text.starting { color: #3b82f6; }
        .azure-job-info {
            flex: 1;
            min-width: 0;
        }
        .azure-job-id {
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.85rem;
            color: var(--text-primary);
            font-weight: 500;
        }
        .azure-job-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .azure-job-meta-item {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .azure-job-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            background: #0078d4;
            color: white;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .azure-job-link:hover {
            background: #106ebe;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 120, 212, 0.3);
        }
        .no-jobs {
            text-align: center;
            padding: 30px 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .no-jobs code {
            display: block;
            margin-top: 12px;
            padding: 10px 14px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 6px;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        /* Refresh button with loading spinner */
        .refresh-btn {
            background: rgba(0, 120, 212, 0.2);
            border: 1px solid rgba(0, 120, 212, 0.4);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            padding: 6px 12px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .refresh-btn:hover:not(:disabled) {
            background: rgba(0, 120, 212, 0.3);
            transform: translateY(-1px);
        }
        .refresh-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .refresh-btn .spinner {
            display: none;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #0078d4;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        .refresh-btn.loading .spinner {
            display: inline-block;
        }
        .refresh-btn.loading .refresh-icon {
            display: none;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    '''


def _get_azure_jobs_panel_html() -> str:
    """Return HTML for the Azure jobs status panel with JS polling, error handling, and loading states."""
    return '''
    <div class="azure-jobs-panel" id="azure-jobs-panel">
        <div class="azure-jobs-header">
            <div class="azure-jobs-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                </svg>
                Azure ML Jobs
            </div>
            <div class="azure-jobs-controls">
                <span class="azure-jobs-refresh" id="jobs-refresh-time">Checking...</span>
                <button id="azure-jobs-refresh-btn" class="refresh-btn" onclick="refreshAzureJobs()" title="Refresh job status from Azure">
                    <span class="refresh-icon">&#8635;</span>
                    <span class="spinner"></span>
                    Refresh
                </button>
            </div>
        </div>

        <!-- API Error Banner (hidden by default) -->
        <div class="api-error-banner" id="azure-jobs-error">
            <span class="error-icon">!</span>
            <span class="error-message" id="azure-jobs-error-msg">Failed to fetch Azure jobs</span>
            <button class="retry-btn" onclick="refreshAzureJobs()">Retry</button>
        </div>

        <!-- Loading state -->
        <div id="azure-jobs-loading" style="display: none; text-align: center; padding: 30px;">
            <div style="display: inline-block; width: 24px; height: 24px; border: 3px solid rgba(0,120,212,0.3); border-top-color: #0078d4; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="margin-top: 12px; color: var(--text-muted); font-size: 0.85rem;">Loading Azure jobs...</div>
        </div>

        <div id="azure-jobs-list">
            <div class="no-jobs">
                <div style="font-size: 2rem; margin-bottom: 12px; opacity: 0.5;">&#9729;</div>
                Checking Azure ML for jobs...
            </div>
        </div>

        <button id="toggle-logs-btn" onclick="toggleLogs()" style="
            margin-top: 12px;
            padding: 8px 14px;
            background: rgba(0, 120, 212, 0.2);
            border: 1px solid rgba(0, 120, 212, 0.4);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        ">
            <span id="logs-icon">&#9660;</span>
            <span id="logs-btn-text">Show Logs</span>
        </button>
        <div id="job-logs-panel" style="display: none; margin-top: 12px;">
            <div id="log-job-status" style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;"></div>
            <pre id="job-logs-content" style="
                background: #1a1a1a;
                color: #10b981;
                padding: 14px;
                border-radius: 6px;
                font-size: 0.75rem;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'SF Mono', Monaco, monospace;
                border: 1px solid rgba(255,255,255,0.1);
            ">Loading logs...</pre>
        </div>
    </div>

    <script>
        // Track refresh state
        let isAzureJobsRefreshing = false;
        let azureJobsErrorCount = 0;

        // Show/hide loading state and error banner
        function setAzureJobsState(state, errorMsg = '') {
            const loadingEl = document.getElementById('azure-jobs-loading');
            const listEl = document.getElementById('azure-jobs-list');
            const errorEl = document.getElementById('azure-jobs-error');
            const errorMsgEl = document.getElementById('azure-jobs-error-msg');
            const refreshTimeEl = document.getElementById('jobs-refresh-time');
            const refreshBtn = document.getElementById('azure-jobs-refresh-btn');

            // Reset states
            loadingEl.style.display = 'none';
            errorEl.classList.remove('show');

            if (state === 'loading') {
                loadingEl.style.display = 'block';
                listEl.style.display = 'none';
                refreshBtn.classList.add('loading');
                refreshBtn.disabled = true;
            } else if (state === 'error') {
                listEl.style.display = 'block';
                errorEl.classList.add('show');
                errorMsgEl.textContent = errorMsg || 'Failed to fetch Azure jobs. Check Azure CLI login.';
                refreshTimeEl.textContent = 'Error';
                refreshTimeEl.classList.add('error');
                refreshTimeEl.classList.remove('success');
                refreshBtn.classList.remove('loading');
                refreshBtn.disabled = false;
            } else if (state === 'success') {
                listEl.style.display = 'block';
                refreshTimeEl.classList.remove('error');
                refreshTimeEl.classList.add('success');
                refreshBtn.classList.remove('loading');
                refreshBtn.disabled = false;
                azureJobsErrorCount = 0;  // Reset error count on success
            } else {
                listEl.style.display = 'block';
                refreshBtn.classList.remove('loading');
                refreshBtn.disabled = false;
            }
        }

        // Force refresh from Azure (bypasses cache)
        async function refreshAzureJobs() {
            if (isAzureJobsRefreshing) return;
            isAzureJobsRefreshing = true;
            setAzureJobsState('loading');
            document.getElementById('jobs-refresh-time').textContent = 'Refreshing...';

            try {
                const response = await fetch('/api/azure-jobs?force=true&t=' + Date.now());
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const jobs = await response.json();
                if (jobs.error) {
                    throw new Error(jobs.error);
                }
                renderAzureJobs(jobs, true);
                setAzureJobsState('success');
                document.getElementById('jobs-refresh-time').textContent =
                    'Live from Azure - ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Azure jobs refresh failed:', e);
                azureJobsErrorCount++;
                setAzureJobsState('error', e.message || 'Connection failed');
            } finally {
                isAzureJobsRefreshing = false;
            }
        }

        // Fetch Azure job status from API (normal polling)
        async function fetchAzureJobs() {
            if (isAzureJobsRefreshing) return;

            // If we've had multiple errors, slow down polling
            if (azureJobsErrorCount >= 3) {
                document.getElementById('jobs-refresh-time').textContent =
                    'Polling paused (too many errors). Click Refresh.';
                return;
            }

            try {
                const response = await fetch('/api/azure-jobs?t=' + Date.now());
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const jobs = await response.json();
                if (jobs.error) {
                    throw new Error(jobs.error);
                }
                renderAzureJobs(jobs, true);
                setAzureJobsState('success');
                document.getElementById('jobs-refresh-time').textContent =
                    'Live - ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.log('Azure API error:', e);
                azureJobsErrorCount++;

                // Try cached fallback
                try {
                    const fallbackResponse = await fetch('benchmark_results/azure_jobs.json?t=' + Date.now());
                    if (fallbackResponse.ok) {
                        const jobs = await fallbackResponse.json();
                        renderAzureJobs(jobs, false);
                        document.getElementById('jobs-refresh-time').textContent =
                            'Cached - ' + new Date().toLocaleTimeString();
                        document.getElementById('jobs-refresh-time').classList.remove('error');
                        return;
                    }
                } catch (fallbackError) {
                    // Fallback also failed
                }

                // Show empty state with guidance
                document.getElementById('azure-jobs-list').innerHTML =
                    '<div class="no-jobs">' +
                    '<div style="font-size: 2rem; margin-bottom: 12px; opacity: 0.5;">&#9729;</div>' +
                    'No Azure jobs found<code>uv run python -m openadapt_ml.benchmarks.cli run-azure</code>' +
                    '</div>';
            }
        }

        function renderAzureJobs(jobs, isLive) {
            if (!jobs || jobs.length === 0) {
                document.getElementById('azure-jobs-list').innerHTML =
                    '<div class="no-jobs">' +
                    '<div style="font-size: 2rem; margin-bottom: 12px; opacity: 0.5;">&#9729;</div>' +
                    'No Azure jobs found<code>uv run python -m openadapt_ml.benchmarks.cli run-azure</code>' +
                    '</div>';
                return;
            }

            const html = jobs.slice(0, 5).map(job => {
                const status = (job.status || 'unknown').toLowerCase();
                const statusClass = status;
                let statusText = job.status ? job.status.charAt(0).toUpperCase() + job.status.slice(1) : 'Unknown';

                // Show display_name if available (live data), otherwise job_id
                const displayName = job.display_name || job.job_id;

                // Calculate elapsed time for running jobs
                let elapsedMins = 0;
                let elapsedText = '';
                let isStuck = false;
                if (job.started_at) {
                    const start = new Date(job.started_at);
                    elapsedMins = (Date.now() - start.getTime()) / 60000;
                    if (status === 'running') {
                        elapsedText = elapsedMins < 60
                            ? Math.round(elapsedMins) + 'm'
                            : Math.round(elapsedMins / 60) + 'h ' + Math.round(elapsedMins % 60) + 'm';
                        // Warn if running > 30 mins
                        if (elapsedMins > 30) {
                            isStuck = true;
                        }
                    }
                }

                // Build metadata items
                const metaItems = [];
                if (elapsedText && status === 'running') {
                    metaItems.push('<span class="azure-job-meta-item">&#128337; ' + elapsedText + '</span>');
                }
                if (!isLive && job.num_tasks) {
                    metaItems.push('<span class="azure-job-meta-item">~' + job.num_tasks + ' tasks</span>');
                }
                if (job.results?.success_rate !== undefined) {
                    metaItems.push('<span class="azure-job-meta-item">' + (job.results.success_rate * 100).toFixed(1) + '% success</span>');
                }
                if (job.started_at && status !== 'running') {
                    const date = new Date(job.started_at);
                    metaItems.push('<span class="azure-job-meta-item">' + date.toLocaleString() + '</span>');
                }
                const metaHtml = metaItems.join('');

                // Add warning for stuck jobs
                const stuckWarning = isStuck
                    ? '<div style="color: #ff9800; font-size: 0.7rem; margin-top: 6px; display: flex; align-items: center; gap: 4px;"><span>&#9888;</span> Running > 30min. May be stuck. Consider canceling.</div>'
                    : '';

                return '<div class="azure-job-item status-' + statusClass + '">' +
                    '<div class="azure-job-status">' +
                        '<span class="status-dot ' + statusClass + '"></span>' +
                        '<span class="status-text ' + statusClass + '">' + statusText + '</span>' +
                    '</div>' +
                    '<div class="azure-job-info">' +
                        '<div class="azure-job-id">' + displayName + '</div>' +
                        '<div class="azure-job-meta">' + metaHtml + '</div>' +
                        stuckWarning +
                    '</div>' +
                    '<a href="' + (job.azure_dashboard_url || '#') + '" target="_blank" class="azure-job-link">' +
                        'Open in Azure &#8594;' +
                    '</a>' +
                '</div>';
            }).join('');

            document.getElementById('azure-jobs-list').innerHTML = html;
        }

        // Log viewer state
        let showLogs = false;
        let currentLogJobId = null;

        async function fetchJobLogs() {
            if (!showLogs) return;

            const logEl = document.getElementById('job-logs-content');
            const statusEl = document.getElementById('log-job-status');

            try {
                const url = currentLogJobId
                    ? '/api/azure-job-logs?job_id=' + currentLogJobId
                    : '/api/azure-job-logs';
                const response = await fetch(url + '&t=' + Date.now());
                if (response.ok) {
                    const data = await response.json();
                    if (logEl) {
                        logEl.textContent = data.logs || 'No logs available';
                        if (data.command) {
                            logEl.textContent = 'Command: ' + data.command + '\\n\\n' + (data.logs || '');
                        }
                        // Color code based on status
                        logEl.style.color = data.status === 'running' ? '#f59e0b' :
                                           data.status === 'completed' ? '#10b981' :
                                           data.status === 'failed' ? '#ef4444' : '#10b981';
                    }
                    if (statusEl && data.job_id) {
                        statusEl.textContent = 'Job: ' + data.job_id + ' (' + data.status + ')';
                    }
                } else {
                    if (logEl) logEl.textContent = 'Failed to fetch logs (HTTP ' + response.status + ')';
                }
            } catch (e) {
                console.log('Error fetching logs:', e);
                if (logEl) logEl.textContent = 'Error fetching logs: ' + e.message;
            }
        }

        function toggleLogs() {
            showLogs = !showLogs;
            const panel = document.getElementById('job-logs-panel');
            const icon = document.getElementById('logs-icon');
            const btnText = document.getElementById('logs-btn-text');

            if (panel) {
                panel.style.display = showLogs ? 'block' : 'none';
            }
            if (icon) {
                icon.innerHTML = showLogs ? '&#9650;' : '&#9660;';
            }
            if (btnText) {
                btnText.textContent = showLogs ? 'Hide Logs' : 'Show Logs';
            }
            if (showLogs) fetchJobLogs();
        }

        // Initial fetch and poll every 30 seconds (use Refresh button for immediate updates)
        fetchAzureJobs();
        setInterval(fetchAzureJobs, 30000);
        setInterval(fetchJobLogs, 5000);  // Poll logs every 5 seconds
    </script>
    '''


def _get_vm_discovery_panel_css() -> str:
    """Return CSS for VM Discovery panel with prominent VNC button."""
    return '''
        .vm-discovery-panel {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }
        .vm-discovery-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .vm-discovery-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            font-weight: 600;
            color: #10b981;
        }
        .vm-discovery-title svg {
            width: 20px;
            height: 20px;
        }
        .vm-discovery-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .vm-discovery-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .vm-item {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 12px;
            transition: all 0.2s;
        }
        .vm-item:last-child {
            margin-bottom: 0;
        }
        .vm-item:hover {
            border-color: rgba(16, 185, 129, 0.5);
        }
        .vm-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .vm-name {
            font-weight: 600;
            font-size: 1rem;
            color: var(--text-primary);
        }
        .vm-status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.2);
        }
        .vm-status-indicator.online {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }
        .vm-status-indicator.offline {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        .vm-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .vm-status-dot.online {
            background: #10b981;
            box-shadow: 0 0 6px #10b981;
        }
        .vm-status-dot.offline {
            background: #ef4444;
        }
        .vm-status-dot.unknown {
            background: #6b7280;
        }
        /* IP Address display - prominent */
        .vm-ip-display {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 8px;
            margin-bottom: 14px;
        }
        .vm-ip-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .vm-ip-value {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: #10b981;
            letter-spacing: 0.5px;
        }
        .vm-ip-copy {
            margin-left: auto;
            padding: 4px 8px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 4px;
            color: #10b981;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
        }
        .vm-ip-copy:hover {
            background: rgba(16, 185, 129, 0.3);
        }
        .vm-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-bottom: 14px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .vm-info-item {
            display: flex;
            gap: 6px;
        }
        .vm-info-label {
            color: var(--text-muted);
        }
        .vm-info-value {
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, monospace;
        }
        .vm-actions {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        /* VNC Button - Large and Prominent */
        .vm-vnc-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 20px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            border-radius: 8px;
            color: white;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        .vm-vnc-link:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }
        .vm-vnc-link .vnc-icon {
            font-size: 1.1rem;
        }
        .vm-vnc-link .vnc-ip {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.8rem;
            opacity: 0.9;
            margin-left: 4px;
        }
        .vm-waa-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .vm-waa-status.ready {
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
            background: rgba(16, 185, 129, 0.1);
        }
        .vm-waa-status.not-ready {
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
            background: rgba(239, 68, 68, 0.1);
        }
        .vm-waa-status.checking {
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.4);
            background: rgba(245, 158, 11, 0.1);
        }
        .vm-last-checked {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .no-vms {
            text-align: center;
            padding: 30px 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .no-vms-icon {
            font-size: 2rem;
            margin-bottom: 12px;
            opacity: 0.5;
        }
        .vm-add-button {
            margin-top: 12px;
            padding: 10px 18px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 6px;
            color: #10b981;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .vm-add-button:hover {
            background: rgba(16, 185, 129, 0.3);
            transform: translateY(-1px);
        }
        .vm-add-form {
            display: none;
            margin-top: 12px;
            padding: 18px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }
        .vm-add-form.show {
            display: block;
        }
        .vm-form-row {
            margin-bottom: 14px;
        }
        .vm-form-row label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 500;
        }
        .vm-form-row input {
            width: 100%;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
            transition: border-color 0.2s;
        }
        .vm-form-row input:focus {
            outline: none;
            border-color: #10b981;
        }
        .vm-form-actions {
            display: flex;
            gap: 10px;
            margin-top: 18px;
        }
        .vm-form-submit {
            padding: 10px 18px;
            background: #10b981;
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .vm-form-cancel {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.85rem;
        }
    '''


def _get_vm_discovery_panel_html() -> str:
    """Return HTML for VM Discovery panel with prominent VNC button and loading states."""
    return '''
    <div class="vm-discovery-panel" id="vm-discovery-panel">
        <div class="vm-discovery-header">
            <div class="vm-discovery-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 3h18v4H3V3zm0 6h18v12H3V9zm2 2v8h14v-8H5zm2 2h4v4H7v-4z"/>
                </svg>
                Windows VMs
            </div>
            <div class="vm-discovery-controls">
                <span class="vm-discovery-refresh" id="vm-refresh-time">Checking...</span>
                <button class="refresh-btn" onclick="refreshVMs()" title="Refresh VM status" id="vm-refresh-btn">
                    <span class="refresh-icon">&#8635;</span>
                    <span class="spinner"></span>
                    Refresh
                </button>
            </div>
        </div>

        <!-- API Error Banner -->
        <div class="api-error-banner" id="vm-api-error">
            <span class="error-icon">!</span>
            <span class="error-message" id="vm-error-msg">Failed to fetch VMs</span>
            <button class="retry-btn" onclick="refreshVMs()">Retry</button>
        </div>

        <!-- Loading state -->
        <div id="vm-loading" style="display: none; text-align: center; padding: 30px;">
            <div style="display: inline-block; width: 24px; height: 24px; border: 3px solid rgba(16,185,129,0.3); border-top-color: #10b981; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="margin-top: 12px; color: var(--text-muted); font-size: 0.85rem;">Checking VM status...</div>
        </div>

        <div id="vm-list">
            <div class="no-vms">
                <div class="no-vms-icon">&#128187;</div>
                Checking for registered VMs...
            </div>
        </div>
        <button id="vm-add-button" class="vm-add-button" onclick="toggleVMAddForm()">
            <span>+</span> Add VM
        </button>
        <div id="vm-add-form" class="vm-add-form">
            <div class="vm-form-row">
                <label>VM Name:</label>
                <input type="text" id="vm-name" placeholder="e.g., azure-waa-vm" />
            </div>
            <div class="vm-form-row">
                <label>SSH Host (IP):</label>
                <input type="text" id="vm-ssh-host" placeholder="e.g., 172.171.112.41" />
            </div>
            <div class="vm-form-row">
                <label>SSH User:</label>
                <input type="text" id="vm-ssh-user" value="azureuser" />
            </div>
            <div class="vm-form-row">
                <label>VNC Port:</label>
                <input type="number" id="vm-vnc-port" value="8006" />
            </div>
            <div class="vm-form-row">
                <label>WAA Port:</label>
                <input type="number" id="vm-waa-port" value="5000" />
            </div>
            <div class="vm-form-row">
                <label>Docker Container:</label>
                <input type="text" id="vm-docker-container" value="win11-waa" />
            </div>
            <div class="vm-form-row">
                <label>Internal IP:</label>
                <input type="text" id="vm-internal-ip" value="20.20.20.21" />
            </div>
            <div class="vm-form-actions">
                <button class="vm-form-submit" onclick="submitVMRegistration()">Register VM</button>
                <button class="vm-form-cancel" onclick="toggleVMAddForm()">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let isVMRefreshing = false;
        let vmErrorCount = 0;

        function setVMLoadingState(loading) {
            const loadingEl = document.getElementById('vm-loading');
            const listEl = document.getElementById('vm-list');
            const btn = document.getElementById('vm-refresh-btn');

            if (loading) {
                loadingEl.style.display = 'block';
                listEl.style.display = 'none';
                if (btn) btn.classList.add('loading');
            } else {
                loadingEl.style.display = 'none';
                listEl.style.display = 'block';
                if (btn) btn.classList.remove('loading');
            }
        }

        function showVMError(msg) {
            const errorEl = document.getElementById('vm-api-error');
            const errorMsgEl = document.getElementById('vm-error-msg');
            if (errorEl && errorMsgEl) {
                errorMsgEl.textContent = msg;
                errorEl.classList.add('show');
            }
        }

        function hideVMError() {
            const errorEl = document.getElementById('vm-api-error');
            if (errorEl) errorEl.classList.remove('show');
        }

        async function refreshVMs() {
            if (isVMRefreshing) return;
            isVMRefreshing = true;
            setVMLoadingState(true);
            hideVMError();

            try {
                const response = await fetch('/api/vms?' + Date.now());
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const vms = await response.json();
                if (vms.error) throw new Error(vms.error);

                renderVMs(vms);
                vmErrorCount = 0;
                document.getElementById('vm-refresh-time').textContent =
                    'Updated ' + new Date().toLocaleTimeString();
            } catch (e) {
                console.error('VM refresh failed:', e);
                vmErrorCount++;
                showVMError(e.message || 'Connection failed');
            } finally {
                isVMRefreshing = false;
                setVMLoadingState(false);
            }
        }

        async function fetchVMs() {
            if (isVMRefreshing) return;
            if (vmErrorCount >= 3) {
                document.getElementById('vm-refresh-time').textContent = 'Polling paused';
                return;
            }

            try {
                const response = await fetch('/api/vms?' + Date.now());
                if (response.ok) {
                    const vms = await response.json();
                    if (!vms.error) {
                        renderVMs(vms);
                        hideVMError();
                        vmErrorCount = 0;
                        document.getElementById('vm-refresh-time').textContent =
                            'Updated ' + new Date().toLocaleTimeString();
                    }
                }
            } catch (e) {
                console.log('VM API unavailable:', e);
                vmErrorCount++;
            }
        }

        function copyToClipboard(text, btn) {
            navigator.clipboard.writeText(text).then(() => {
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = originalText; }, 1500);
            });
        }

        function renderVMs(vms) {
            const container = document.getElementById('vm-list');

            if (!vms || vms.length === 0) {
                container.innerHTML = '<div class="no-vms"><div class="no-vms-icon">&#128187;</div>No VMs registered. Click "Add VM" to register one.</div>';
                return;
            }

            const html = vms.map(vm => {
                const statusClass = vm.status || 'unknown';
                const statusText = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);
                const waaStatusClass = vm.waa_probe_status === 'ready' ? 'ready' :
                                       vm.waa_probe_status === 'checking' ? 'checking' : 'not-ready';
                const waaStatusIcon = vm.waa_probe_status === 'ready' ? '&#10003;' :
                                      vm.waa_probe_status === 'checking' ? '&#8987;' : '&#10007;';
                const waaStatusText = vm.waa_probe_status === 'ready' ? 'WAA Server Ready' :
                                     vm.waa_probe_status === 'not responding' ? 'WAA Not Responding' :
                                     vm.waa_probe_status === 'checking' ? 'Checking...' :
                                     vm.waa_probe_status === 'ssh failed' ? 'SSH Failed' : 'Unknown';

                const vncUrl = 'http://' + vm.ssh_host + ':' + (vm.vnc_port || 8006);
                const vmIp = vm.ssh_host;

                return '<div class="vm-item">' +
                    '<div class="vm-item-header">' +
                        '<span class="vm-name">' + (vm.name || 'Unnamed VM') + '</span>' +
                        '<div class="vm-status-indicator ' + statusClass + '">' +
                            '<div class="vm-status-dot ' + statusClass + '"></div>' +
                            '<span>' + statusText + '</span>' +
                        '</div>' +
                    '</div>' +

                    // Prominent IP display
                    '<div class="vm-ip-display">' +
                        '<span class="vm-ip-label">IP Address:</span>' +
                        '<span class="vm-ip-value">' + vmIp + '</span>' +
                        '<button class="vm-ip-copy" onclick="copyToClipboard(\\\'' + vmIp + '\\\', this)">Copy</button>' +
                    '</div>' +

                    '<div class="vm-info">' +
                        '<div class="vm-info-item">' +
                            '<span class="vm-info-label">SSH:</span>' +
                            '<span class="vm-info-value">' + (vm.ssh_user || 'azureuser') + '@' + vmIp + '</span>' +
                        '</div>' +
                        '<div class="vm-info-item">' +
                            '<span class="vm-info-label">Container:</span>' +
                            '<span class="vm-info-value">' + (vm.docker_container || 'win11-waa') + '</span>' +
                        '</div>' +
                    '</div>' +

                    '<div class="vm-actions">' +
                        // Large prominent VNC button
                        '<a href="' + vncUrl + '" target="_blank" class="vm-vnc-link">' +
                            '<span class="vnc-icon">&#128424;</span>' +
                            'Open VNC' +
                            '<span class="vnc-ip">:' + (vm.vnc_port || 8006) + '</span>' +
                        '</a>' +
                        '<div class="vm-waa-status ' + waaStatusClass + '">' +
                            waaStatusIcon + ' ' + waaStatusText +
                        '</div>' +
                    '</div>' +

                    '<div class="vm-last-checked">' +
                        '<span>&#128337;</span> Last checked: ' + (vm.last_checked ? new Date(vm.last_checked).toLocaleString() : 'Never') +
                    '</div>' +
                '</div>';
            }).join('');

            container.innerHTML = html;
        }

        function toggleVMAddForm() {
            const form = document.getElementById('vm-add-form');
            form.classList.toggle('show');
        }

        async function submitVMRegistration() {
            const vmData = {
                name: document.getElementById('vm-name').value,
                ssh_host: document.getElementById('vm-ssh-host').value,
                ssh_user: document.getElementById('vm-ssh-user').value,
                vnc_port: parseInt(document.getElementById('vm-vnc-port').value),
                waa_port: parseInt(document.getElementById('vm-waa-port').value),
                docker_container: document.getElementById('vm-docker-container').value,
                internal_ip: document.getElementById('vm-internal-ip').value
            };

            // Basic validation
            if (!vmData.name || !vmData.ssh_host) {
                alert('Please fill in VM Name and SSH Host');
                return;
            }

            try {
                const response = await fetch('/api/vms/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(vmData)
                });

                if (response.ok) {
                    const result = await response.json();
                    if (result.status === 'success') {
                        toggleVMAddForm();
                        fetchVMs();
                        // Clear form
                        document.getElementById('vm-name').value = '';
                        document.getElementById('vm-ssh-host').value = '';
                    } else {
                        alert('Failed to register VM: ' + (result.message || 'Unknown error'));
                    }
                } else {
                    alert('Failed to register VM: Server error (HTTP ' + response.status + ')');
                }
            } catch (e) {
                alert('Failed to register VM: ' + e.message);
            }
        }

        // Initial fetch and poll every 10 seconds
        fetchVMs();
        setInterval(fetchVMs, 10000);
    </script>
    '''


def generate_benchmark_viewer(
    benchmark_dir: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Generate benchmark viewer HTML from benchmark results directory.

    Args:
        benchmark_dir: Path to benchmark results directory (e.g., benchmark_results/waa_eval_20241214/)
        output_path: Optional path for output benchmark.html (default: benchmark_dir/benchmark.html)

    Returns:
        Path to generated benchmark.html file

    Example:
        from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

        viewer_path = generate_benchmark_viewer("benchmark_results/test_run_phase1")
        print(f"Generated: {viewer_path}")
    """
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    if output_path is None:
        output_path = benchmark_dir / "benchmark.html"
    else:
        output_path = Path(output_path)

    # Load metadata
    metadata_path = benchmark_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {benchmark_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load summary
    summary_path = benchmark_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Load all task results
    tasks_dir = benchmark_dir / "tasks"
    task_results = []

    if tasks_dir.exists():
        for task_dir in sorted(tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_json = task_dir / "task.json"
            execution_json = task_dir / "execution.json"

            if not task_json.exists() or not execution_json.exists():
                continue

            with open(task_json) as f:
                task_data = json.load(f)

            with open(execution_json) as f:
                execution_data = json.load(f)

            # Combine task and execution data
            task_result = {
                "task_id": task_data["task_id"],
                "instruction": task_data["instruction"],
                "domain": task_data.get("domain", "unknown"),
                "success": execution_data["success"],
                "score": execution_data.get("score", 0.0),
                "num_steps": execution_data["num_steps"],
                "total_time_seconds": execution_data.get("total_time_seconds", 0.0),
                "error": execution_data.get("error"),
                "reason": execution_data.get("reason"),
                "steps": execution_data.get("steps", []),
                "screenshots_dir": str(task_dir / "screenshots"),
            }
            task_results.append(task_result)

    # Import shared header components from trainer
    from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html

    # Generate HTML
    html = _generate_benchmark_viewer_html(
        metadata=metadata,
        summary=summary,
        tasks=task_results,
        benchmark_dir=benchmark_dir,
        shared_header_css=_get_shared_header_css(),
        shared_header_html=_generate_shared_header_html("benchmarks"),
    )

    output_path.write_text(html)
    print(f"Generated benchmark viewer: {output_path}")
    return output_path


def generate_multi_run_benchmark_viewer(
    benchmark_dirs: list[Path],
    output_path: Path | str,
) -> Path:
    """Generate benchmark viewer HTML supporting multiple benchmark runs.

    Args:
        benchmark_dirs: List of benchmark result directories (sorted most recent first)
        output_path: Path for output benchmark.html

    Returns:
        Path to generated benchmark.html file
    """
    output_path = Path(output_path)

    # Load metadata and summary for all runs
    all_runs = []
    for benchmark_dir in benchmark_dirs:
        metadata_path = benchmark_dir / "metadata.json"
        summary_path = benchmark_dir / "summary.json"

        if not metadata_path.exists() or not summary_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)
        with open(summary_path) as f:
            summary = json.load(f)

        # Load all task results for this run
        tasks_dir = benchmark_dir / "tasks"
        task_results = []

        if tasks_dir.exists():
            for task_dir in sorted(tasks_dir.iterdir()):
                if not task_dir.is_dir():
                    continue

                task_json = task_dir / "task.json"
                execution_json = task_dir / "execution.json"

                if not task_json.exists() or not execution_json.exists():
                    continue

                with open(task_json) as f:
                    task_data = json.load(f)

                with open(execution_json) as f:
                    execution_data = json.load(f)

                # Combine task and execution data
                task_result = {
                    "task_id": task_data["task_id"],
                    "instruction": task_data["instruction"],
                    "domain": task_data.get("domain", "unknown"),
                    "success": execution_data["success"],
                    "score": execution_data.get("score", 0.0),
                    "num_steps": execution_data["num_steps"],
                    "total_time_seconds": execution_data.get("total_time_seconds", 0.0),
                    "error": execution_data.get("error"),
                    "reason": execution_data.get("reason"),
                    "steps": execution_data.get("steps", []),
                }
                task_results.append(task_result)

        all_runs.append({
            "run_name": metadata.get("run_name", benchmark_dir.name),
            "model_id": metadata.get("model_id", "unknown"),
            "created_at": metadata.get("created_at", ""),
            "benchmark_name": metadata.get("benchmark_name", ""),
            "dir_name": benchmark_dir.name,  # For screenshot paths
            "summary": summary,
            "tasks": task_results,
        })

    if not all_runs:
        return generate_empty_benchmark_viewer(output_path)

    # Import shared header components from trainer
    from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html

    # Generate HTML
    html = _generate_multi_run_benchmark_viewer_html(
        runs=all_runs,
        shared_header_css=_get_shared_header_css(),
        shared_header_html=_generate_shared_header_html("benchmarks"),
    )

    output_path.write_text(html)
    print(f"Generated multi-run benchmark viewer: {output_path}")
    return output_path


def generate_empty_benchmark_viewer(output_path: Path | str) -> Path:
    """Generate an empty benchmark viewer with guidance when no real data exists.

    Args:
        output_path: Path to output benchmark.html

    Returns:
        Path to generated file
    """
    output_path = Path(output_path)

    # Import shared header components from trainer
    from openadapt_ml.training.trainer import _get_shared_header_css, _generate_shared_header_html

    shared_header_css = _get_shared_header_css()
    shared_header_html = _generate_shared_header_html("benchmarks")
    azure_jobs_css = _get_azure_jobs_panel_css()
    azure_jobs_html = _get_azure_jobs_panel_html()
    tasks_css = _get_background_tasks_panel_css()
    tasks_html = _get_background_tasks_panel_html()
    live_eval_css = _get_live_evaluation_panel_css()
    live_eval_html = _get_live_evaluation_panel_html()
    vm_discovery_css = _get_vm_discovery_panel_css()
    vm_discovery_html = _get_vm_discovery_panel_html()

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Viewer - No Data</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        {shared_header_css}
        {tasks_css}
        {azure_jobs_css}
        {live_eval_css}
        {vm_discovery_css}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 24px;
        }}
        .empty-state {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: calc(100vh - 60px);
            padding: 40px;
            text-align: center;
        }}
        .empty-icon {{
            font-size: 64px;
            margin-bottom: 24px;
            opacity: 0.5;
        }}
        .empty-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        .empty-description {{
            color: var(--text-secondary);
            margin-bottom: 32px;
            max-width: 500px;
            line-height: 1.6;
        }}
        .guide-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
            max-width: 600px;
            text-align: left;
        }}
        .guide-card h3 {{
            color: var(--accent);
            margin-bottom: 12px;
            font-size: 16px;
        }}
        .guide-card code {{
            background: var(--bg-tertiary);
            padding: 12px 16px;
            border-radius: 8px;
            display: block;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: var(--text-primary);
            white-space: pre-wrap;
            margin-bottom: 12px;
        }}
        .guide-card p {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
        }}
        a {{
            color: var(--accent);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        {live_eval_html}
        {tasks_html}
        {azure_jobs_html}
        {vm_discovery_html}
    </div>

    <div class="empty-state">
        <div class="empty-icon">🚧</div>
        <h1 class="empty-title">Windows Agent Arena Integration</h1>
        <p class="empty-description">
            This tab will display results from <strong>WAA benchmark</strong> evaluations (154 real Windows tasks).<br>
            <span style="color: var(--text-muted);">Status: Work in Progress - requires Windows VM or Azure setup</span>
        </p>

        <div class="guide-card" style="background: var(--bg-tertiary); border-color: var(--accent);">
            <h3 style="color: var(--text-primary);">Looking for synthetic benchmark results?</h3>
            <code>uv run python -m openadapt_ml.scripts.eval_policy \\
  --config configs/qwen3vl_synthetic_som.yaml \\
  --backend qwen3 --dsl-mode som</code>
            <p>The synthetic login benchmark (with SoM mode achieving 100%) uses eval_policy.py, not this viewer.</p>
        </div>

        <div class="guide-card">
            <h3>WAA Local Setup (Windows Required)</h3>
            <code># Clone WAA repository
git clone https://github.com/anthropics/WindowsAgentArena

# Run evaluation
uv run python -m openadapt_ml.benchmarks.cli run-local \\
  --waa-path /path/to/WindowsAgentArena</code>
            <p>Requires Windows environment. See <a href="https://github.com/anthropics/WindowsAgentArena" style="color: var(--accent);">WAA repo</a> for setup.</p>
        </div>

        <div class="guide-card">
            <h3>WAA on Azure (Parallel VMs)</h3>
            <code># Setup Azure resources
python scripts/setup_azure.py

# Run evaluation on Azure VMs
uv run python -m openadapt_ml.benchmarks.cli run-azure --workers 4</code>
            <p>Runs WAA tasks in parallel on Azure Windows VMs. See docs/azure_waa_setup.md</p>
        </div>
    </div>
</body>
</html>'''

    output_path.write_text(html)
    return output_path


def _generate_benchmark_viewer_html(
    metadata: dict,
    summary: dict,
    tasks: list[dict],
    benchmark_dir: Path,
    shared_header_css: str,
    shared_header_html: str,
) -> str:
    """Generate the benchmark viewer HTML content.

    Args:
        metadata: Benchmark metadata (run name, model ID, etc.)
        summary: Summary statistics (success rate, avg steps, etc.)
        tasks: List of task results with execution data
        benchmark_dir: Path to benchmark directory (for relative paths)
        shared_header_css: CSS for shared header
        shared_header_html: HTML for shared header

    Returns:
        Complete HTML string
    """
    # Prepare data as JSON
    tasks_json = json.dumps(tasks)
    summary_json = json.dumps(summary)
    metadata_json = json.dumps(metadata)

    # Calculate unique domains for filter
    domains = sorted(set(task["domain"] for task in tasks))
    domains_json = json.dumps(domains)

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Viewer - {metadata.get("run_name", "Unknown")}</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
            --success: #00d4aa;
            --failure: #ff4444;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}

        {shared_header_css}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s;
        }}

        .summary-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
        }}

        .summary-card .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .summary-card .subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .filters {{
            display: flex;
            gap: 12px;
            padding: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .filter-select {{
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
        }}

        .filter-select:hover {{
            border-color: var(--accent);
            background-color: rgba(0,212,170,0.1);
        }}

        .task-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .task-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s;
        }}

        .task-item:hover {{
            border-color: var(--accent);
        }}

        .task-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            cursor: pointer;
            user-select: none;
        }}

        .task-header:hover {{
            background: var(--bg-tertiary);
        }}

        .task-status {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
            flex-shrink: 0;
        }}

        .task-status.success {{
            background: var(--success);
            color: var(--bg-primary);
        }}

        .task-status.failure {{
            background: var(--failure);
            color: var(--bg-primary);
        }}

        .task-info {{
            flex: 1;
            min-width: 0;
        }}

        .task-id {{
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }}

        .task-instruction {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .task-meta {{
            display: flex;
            gap: 20px;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .task-domain {{
            padding: 4px 10px;
            background: rgba(0,212,170,0.15);
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
        }}

        .task-expand-icon {{
            color: var(--text-muted);
            transition: transform 0.2s;
        }}

        .task-item.expanded .task-expand-icon {{
            transform: rotate(90deg);
        }}

        .task-details {{
            display: none;
            padding: 0 20px 20px;
            border-top: 1px solid var(--border-color);
        }}

        .task-item.expanded .task-details {{
            display: block;
        }}

        .steps-list {{
            margin-top: 16px;
        }}

        .step-item {{
            display: flex;
            gap: 16px;
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 8px;
        }}

        .step-number {{
            font-weight: 600;
            color: var(--accent);
            min-width: 60px;
        }}

        .step-screenshot {{
            max-width: 200px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}

        .step-action {{
            flex: 1;
        }}

        .action-type {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            color: var(--accent);
            margin-bottom: 4px;
        }}

        .action-details {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .no-tasks {{
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }}

        .no-tasks-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
            opacity: 0.5;
        }}

        .mock-banner {{
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 87, 34, 0.2) 100%);
            border: 2px solid #ff9800;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .mock-banner-icon {{
            font-size: 2rem;
            flex-shrink: 0;
        }}

        .mock-banner-content {{
            flex: 1;
        }}

        .mock-banner-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #ff9800;
            margin-bottom: 6px;
        }}

        .mock-banner-text {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        .run-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 24px;
        }}

        .run-badge.mock {{
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 87, 34, 0.2) 100%);
            border: 1px solid #ff9800;
            color: #ffb74d;
        }}

        .run-badge.real {{
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(0, 150, 136, 0.2) 100%);
            border: 1px solid var(--success);
            color: var(--success);
        }}

        .run-badge-icon {{
            font-size: 1rem;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        <div id="mock-banner" class="mock-banner" style="display: none;">
            <div class="mock-banner-icon">WARNING</div>
            <div class="mock-banner-content">
                <div class="mock-banner-title">Mock Data - Simulated Results Only</div>
                <div class="mock-banner-text">
                    This benchmark run uses simulated mock data for pipeline testing and development.
                    These results do NOT represent actual Windows Agent Arena evaluation performance.
                    To run real WAA evaluation, use: <code>uv run python -m openadapt_ml.benchmarks.cli run-local</code> or <code>run-azure</code>
                </div>
            </div>
        </div>

        <div id="run-badge" class="run-badge" style="display: none;">
            <span class="run-badge-icon"></span>
            <span class="run-badge-text"></span>
        </div>

        <div class="summary-cards">
            <div class="summary-card">
                <div class="label">Total Tasks</div>
                <div class="value" id="total-tasks">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Success Rate</div>
                <div class="value" id="success-rate">0%</div>
                <div class="subtitle" id="success-count">0 / 0 passed</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Steps</div>
                <div class="value" id="avg-steps">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Time</div>
                <div class="value" id="avg-time">0s</div>
            </div>
        </div>

        <div class="filters">
            <span class="filter-label">Status:</span>
            <select class="filter-select" id="filter-status">
                <option value="all">All Tasks</option>
                <option value="success">Success Only</option>
                <option value="failure">Failure Only</option>
            </select>

            <span class="filter-label">Domain:</span>
            <select class="filter-select" id="filter-domain">
                <option value="all">All Domains</option>
            </select>
        </div>

        <div class="task-list" id="task-list"></div>

        <div class="no-tasks" id="no-tasks" style="display: none;">
            <div class="no-tasks-icon">📋</div>
            <div>No tasks match the current filters</div>
        </div>
    </div>

    <script>
        // Data from backend
        const tasks = {tasks_json};
        const summary = {summary_json};
        const metadata = {metadata_json};
        const domains = {domains_json};

        // State
        let currentFilters = {{
            status: 'all',
            domain: 'all'
        }};

        // Detect mock vs real run and show appropriate badges
        function detectAndShowRunType() {{
            const isMock = metadata.benchmark_name && metadata.benchmark_name.includes('mock');
            const badge = document.getElementById('run-badge');
            const banner = document.getElementById('mock-banner');
            const badgeIcon = badge.querySelector('.run-badge-icon');
            const badgeText = badge.querySelector('.run-badge-text');

            if (isMock) {{
                // Show mock warning badge
                badge.classList.add('mock');
                badge.classList.remove('real');
                badgeIcon.textContent = '⚠️';
                badgeText.textContent = 'MOCK DATA - Simulated results for pipeline testing';
                badge.style.display = 'inline-flex';

                // Show mock banner
                banner.style.display = 'flex';
            }} else {{
                // Show real evaluation badge
                badge.classList.add('real');
                badge.classList.remove('mock');
                badgeIcon.textContent = '✓';
                badgeText.textContent = 'REAL - Actual Windows Agent Arena evaluation';
                badge.style.display = 'inline-flex';

                // Hide mock banner
                banner.style.display = 'none';
            }}
        }}

        // Initialize
        function init() {{
            detectAndShowRunType();
            updateSummaryCards();
            populateDomainFilter();
            renderTaskList();

            // Event listeners
            document.getElementById('filter-status').addEventListener('change', (e) => {{
                currentFilters.status = e.target.value;
                renderTaskList();
            }});

            document.getElementById('filter-domain').addEventListener('change', (e) => {{
                currentFilters.domain = e.target.value;
                renderTaskList();
            }});
        }}

        function updateSummaryCards() {{
            document.getElementById('total-tasks').textContent = summary.num_tasks || tasks.length;

            const successRate = (summary.success_rate || 0) * 100;
            document.getElementById('success-rate').textContent = successRate.toFixed(1) + '%';
            document.getElementById('success-count').textContent =
                `${{summary.num_success || 0}} / ${{summary.num_tasks || tasks.length}} passed`;

            const avgSteps = summary.avg_steps || 0;
            document.getElementById('avg-steps').textContent = avgSteps.toFixed(1);

            const avgTime = summary.avg_time_seconds || 0;
            document.getElementById('avg-time').textContent = avgTime.toFixed(2) + 's';
        }}

        function populateDomainFilter() {{
            const select = document.getElementById('filter-domain');
            domains.forEach(domain => {{
                const option = document.createElement('option');
                option.value = domain;
                option.textContent = domain.charAt(0).toUpperCase() + domain.slice(1);
                select.appendChild(option);
            }});
        }}

        function filterTasks() {{
            return tasks.filter(task => {{
                if (currentFilters.status !== 'all') {{
                    const isSuccess = task.success;
                    if (currentFilters.status === 'success' && !isSuccess) return false;
                    if (currentFilters.status === 'failure' && isSuccess) return false;
                }}

                if (currentFilters.domain !== 'all' && task.domain !== currentFilters.domain) {{
                    return false;
                }}

                return true;
            }});
        }}

        function renderTaskList() {{
            const filteredTasks = filterTasks();
            const container = document.getElementById('task-list');
            const noTasks = document.getElementById('no-tasks');

            if (filteredTasks.length === 0) {{
                container.innerHTML = '';
                noTasks.style.display = 'block';
                return;
            }}

            noTasks.style.display = 'none';
            container.innerHTML = filteredTasks.map(task => renderTaskItem(task)).join('');

            // Add click handlers
            document.querySelectorAll('.task-header').forEach(header => {{
                header.addEventListener('click', () => {{
                    const item = header.closest('.task-item');
                    item.classList.toggle('expanded');
                }});
            }});
        }}

        function renderTaskItem(task) {{
            const statusClass = task.success ? 'success' : 'failure';
            const statusIcon = task.success ? '✓' : '✗';

            const stepsHtml = task.steps && task.steps.length > 0
                ? task.steps.map(step => renderStep(step, task)).join('')
                : '<div style="padding: 12px; color: var(--text-muted);">No step details available</div>';

            return `
                <div class="task-item" data-task-id="${{task.task_id}}">
                    <div class="task-header">
                        <div class="task-status ${{statusClass}}">${{statusIcon}}</div>
                        <div class="task-info">
                            <div class="task-id">${{task.task_id}}</div>
                            <div class="task-instruction">${{task.instruction}}</div>
                        </div>
                        <div class="task-domain">${{task.domain}}</div>
                        <div class="task-meta">
                            <span>${{task.num_steps}} steps</span>
                            <span>${{task.total_time_seconds.toFixed(2)}}s</span>
                        </div>
                        <div class="task-expand-icon">▶</div>
                    </div>
                    <div class="task-details">
                        <div class="steps-list">
                            ${{stepsHtml}}
                        </div>
                    </div>
                </div>
            `;
        }}

        function renderStep(step, task) {{
            const actionType = step.action.type || 'unknown';
            const actionDetails = formatActionDetails(step.action);

            // Build screenshot path relative to benchmark.html
            const screenshotPath = step.screenshot_path
                ? `tasks/${{task.task_id}}/${{step.screenshot_path}}`
                : '';

            const screenshotHtml = screenshotPath
                ? `<img src="${{screenshotPath}}" class="step-screenshot" alt="Step ${{step.step_idx}}" />`
                : '';

            return `
                <div class="step-item">
                    <div class="step-number">Step ${{step.step_idx}}</div>
                    ${{screenshotHtml}}
                    <div class="step-action">
                        <div class="action-type">${{actionType}}</div>
                        <div class="action-details">${{actionDetails}}</div>
                        ${{step.reasoning ? `<div style="margin-top: 8px; font-style: italic; color: var(--text-secondary);">${{step.reasoning}}</div>` : ''}}
                    </div>
                </div>
            `;
        }}

        function formatActionDetails(action) {{
            const parts = [];

            if (action.x !== null && action.y !== null) {{
                parts.push(`x: ${{action.x.toFixed(3)}}, y: ${{action.y.toFixed(3)}}`);
            }}

            if (action.text) {{
                parts.push(`text: "${{action.text}}"`);
            }}

            if (action.key) {{
                parts.push(`key: ${{action.key}}`);
            }}

            if (action.target_name) {{
                parts.push(`target: ${{action.target_name}}`);
            }}

            return parts.length > 0 ? parts.join(', ') : 'No details';
        }}

        // Initialize on page load
        init();
    </script>
</body>
</html>'''

    return html


def _generate_multi_run_benchmark_viewer_html(
    runs: list[dict],
    shared_header_css: str,
    shared_header_html: str,
) -> str:
    """Generate HTML for multi-run benchmark viewer with run selector.

    Args:
        runs: List of run dictionaries with metadata, summary, and tasks
        shared_header_css: CSS for shared header
        shared_header_html: HTML for shared header

    Returns:
        Complete HTML string
    """
    # Get Azure jobs panel components
    azure_jobs_css = _get_azure_jobs_panel_css()
    azure_jobs_html = _get_azure_jobs_panel_html()
    tasks_css = _get_background_tasks_panel_css()
    tasks_html = _get_background_tasks_panel_html()
    live_eval_css = _get_live_evaluation_panel_css()
    live_eval_html = _get_live_evaluation_panel_html()
    vm_discovery_css = _get_vm_discovery_panel_css()
    vm_discovery_html = _get_vm_discovery_panel_html()

    # Prepare runs data as JSON
    runs_json = json.dumps(runs)

    # Calculate unique domains across all runs
    all_domains = set()
    for run in runs:
        for task in run["tasks"]:
            all_domains.add(task["domain"])
    domains = sorted(all_domains)
    domains_json = json.dumps(domains)

    # Build run selector options
    run_options = []
    for i, run in enumerate(runs):
        success_rate = run["summary"].get("success_rate", 0) * 100
        label = f"{run['model_id']} - {success_rate:.0f}% ({run['run_name']})"
        run_options.append(f'<option value="{i}">{label}</option>')
    run_options_html = "\n".join(run_options)

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Viewer - Multiple Runs</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
            --success: #00d4aa;
            --failure: #ff4444;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}

        {shared_header_css}
        {tasks_css}
        {azure_jobs_css}
        {live_eval_css}
        {vm_discovery_css}

        .run-selector-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .run-selector-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        #run-selector {{
            flex: 1;
            max-width: 600px;
            padding: 10px 36px 10px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            background: rgba(0,0,0,0.4);
            color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.1);
            cursor: pointer;
            appearance: none;
            background-image: url('data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2712%27 height=%278%27%3E%3Cpath fill=%27%23888%27 d=%27M0 0l6 8 6-8z%27/%3E%3C/svg%3E');
            background-repeat: no-repeat;
            background-position: right 12px center;
            transition: all 0.2s;
        }}

        #run-selector:hover {{
            border-color: var(--accent);
            background-color: rgba(0,212,170,0.1);
        }}

        #run-selector:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(0,212,170,0.2);
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s;
        }}

        .summary-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
        }}

        .summary-card .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .summary-card .subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .filters {{
            display: flex;
            gap: 12px;
            padding: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .filter-select {{
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
        }}

        .filter-select:hover {{
            border-color: var(--accent);
            background-color: rgba(0,212,170,0.1);
        }}

        .task-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .task-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s;
        }}

        .task-item:hover {{
            border-color: var(--accent);
        }}

        .task-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            cursor: pointer;
            user-select: none;
        }}

        .task-header:hover {{
            background: var(--bg-tertiary);
        }}

        .task-status {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
            flex-shrink: 0;
        }}

        .task-status.success {{
            background: var(--success);
            color: var(--bg-primary);
        }}

        .task-status.failure {{
            background: var(--failure);
            color: var(--bg-primary);
        }}

        .task-info {{
            flex: 1;
            min-width: 0;
        }}

        .task-id {{
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }}

        .task-instruction {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .task-meta {{
            display: flex;
            gap: 20px;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .task-domain {{
            padding: 4px 10px;
            background: rgba(0,212,170,0.15);
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
        }}

        .task-expand-icon {{
            color: var(--text-muted);
            transition: transform 0.2s;
        }}

        .task-item.expanded .task-expand-icon {{
            transform: rotate(90deg);
        }}

        .task-details {{
            display: none;
            padding: 0 20px 20px;
            border-top: 1px solid var(--border-color);
        }}

        .task-item.expanded .task-details {{
            display: block;
        }}

        .steps-list {{
            margin-top: 16px;
        }}

        .step-item {{
            display: flex;
            gap: 16px;
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 8px;
        }}

        .step-number {{
            font-weight: 600;
            color: var(--accent);
            min-width: 60px;
        }}

        .step-screenshot {{
            max-width: 200px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}

        .step-action {{
            flex: 1;
        }}

        .action-type {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            color: var(--accent);
            margin-bottom: 4px;
        }}

        .action-details {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: "SF Mono", Monaco, monospace;
        }}

        .no-tasks {{
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }}

        .no-tasks-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
            opacity: 0.5;
        }}

        .mock-banner {{
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 87, 34, 0.2) 100%);
            border: 2px solid #ff9800;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .mock-banner-icon {{
            font-size: 2rem;
            flex-shrink: 0;
        }}

        .mock-banner-content {{
            flex: 1;
        }}

        .mock-banner-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #ff9800;
            margin-bottom: 6px;
        }}

        .mock-banner-text {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        .run-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 24px;
        }}

        .run-badge.mock {{
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 87, 34, 0.2) 100%);
            border: 1px solid #ff9800;
            color: #ffb74d;
        }}

        .run-badge.real {{
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(0, 150, 136, 0.2) 100%);
            border: 1px solid var(--success);
            color: var(--success);
        }}

        .run-badge-icon {{
            font-size: 1rem;
        }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        {live_eval_html}
        {tasks_html}
        {azure_jobs_html}
        {vm_discovery_html}

        <div id="mock-banner" class="mock-banner" style="display: none;">
            <div class="mock-banner-icon">WARNING</div>
            <div class="mock-banner-content">
                <div class="mock-banner-title">Mock Data - Simulated Results Only</div>
                <div class="mock-banner-text">
                    This benchmark run uses simulated mock data for pipeline testing and development.
                    These results do NOT represent actual Windows Agent Arena evaluation performance.
                    To run real WAA evaluation, use: <code>uv run python -m openadapt_ml.benchmarks.cli run-local</code> or <code>run-azure</code>
                </div>
            </div>
        </div>

        <div class="run-selector-section">
            <span class="run-selector-label">Benchmark Run:</span>
            <select id="run-selector">
                {run_options_html}
            </select>
        </div>

        <div id="run-badge" class="run-badge" style="display: none;">
            <span class="run-badge-icon"></span>
            <span class="run-badge-text"></span>
        </div>

        <div class="summary-cards">
            <div class="summary-card">
                <div class="label">Total Tasks</div>
                <div class="value" id="total-tasks">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Success Rate</div>
                <div class="value" id="success-rate">0%</div>
                <div class="subtitle" id="success-count">0 / 0 passed</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Steps</div>
                <div class="value" id="avg-steps">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Avg Time</div>
                <div class="value" id="avg-time">0s</div>
            </div>
        </div>

        <div class="filters">
            <span class="filter-label">Status:</span>
            <select class="filter-select" id="filter-status">
                <option value="all">All Tasks</option>
                <option value="success">Success Only</option>
                <option value="failure">Failure Only</option>
            </select>

            <span class="filter-label">Domain:</span>
            <select class="filter-select" id="filter-domain">
                <option value="all">All Domains</option>
            </select>
        </div>

        <div class="task-list" id="task-list"></div>

        <div class="no-tasks" id="no-tasks" style="display: none;">
            <div class="no-tasks-icon">📋</div>
            <div>No tasks match the current filters</div>
        </div>
    </div>

    <script>
        // Data from backend
        const allRuns = {runs_json};
        const allDomains = {domains_json};

        // State
        let currentRunIndex = 0;
        let currentFilters = {{
            status: 'all',
            domain: 'all'
        }};

        // Get current run data
        function getCurrentRun() {{
            return allRuns[currentRunIndex];
        }}

        function getCurrentTasks() {{
            return getCurrentRun().tasks;
        }}

        function getCurrentSummary() {{
            return getCurrentRun().summary;
        }}

        // Detect mock vs real run and show appropriate badges
        function detectAndShowRunType() {{
            const currentRun = getCurrentRun();
            const isMock = currentRun.benchmark_name && currentRun.benchmark_name.includes('mock');
            const badge = document.getElementById('run-badge');
            const banner = document.getElementById('mock-banner');
            const badgeIcon = badge.querySelector('.run-badge-icon');
            const badgeText = badge.querySelector('.run-badge-text');

            if (isMock) {{
                // Show mock warning badge
                badge.classList.add('mock');
                badge.classList.remove('real');
                badgeIcon.textContent = '⚠️';
                badgeText.textContent = 'MOCK DATA - Simulated results for pipeline testing';
                badge.style.display = 'inline-flex';

                // Show mock banner
                banner.style.display = 'flex';
            }} else {{
                // Show real evaluation badge
                badge.classList.add('real');
                badge.classList.remove('mock');
                badgeIcon.textContent = '✓';
                badgeText.textContent = 'REAL - Actual Windows Agent Arena evaluation';
                badge.style.display = 'inline-flex';

                // Hide mock banner
                banner.style.display = 'none';
            }}
        }}

        // Initialize
        function init() {{
            populateDomainFilter();
            updateDisplay();

            // Event listeners
            document.getElementById('run-selector').addEventListener('change', (e) => {{
                currentRunIndex = parseInt(e.target.value);
                updateDisplay();
            }});

            document.getElementById('filter-status').addEventListener('change', (e) => {{
                currentFilters.status = e.target.value;
                renderTaskList();
            }});

            document.getElementById('filter-domain').addEventListener('change', (e) => {{
                currentFilters.domain = e.target.value;
                renderTaskList();
            }});
        }}

        function updateDisplay() {{
            detectAndShowRunType();
            updateSummaryCards();
            renderTaskList();
        }}

        function updateSummaryCards() {{
            const summary = getCurrentSummary();
            const tasks = getCurrentTasks();

            document.getElementById('total-tasks').textContent = summary.num_tasks || tasks.length;

            const successRate = (summary.success_rate || 0) * 100;
            document.getElementById('success-rate').textContent = successRate.toFixed(1) + '%';
            document.getElementById('success-count').textContent =
                `${{summary.num_success || 0}} / ${{summary.num_tasks || tasks.length}} passed`;

            const avgSteps = summary.avg_steps || 0;
            document.getElementById('avg-steps').textContent = avgSteps.toFixed(1);

            const avgTime = summary.avg_time_seconds || 0;
            document.getElementById('avg-time').textContent = avgTime.toFixed(2) + 's';
        }}

        function populateDomainFilter() {{
            const select = document.getElementById('filter-domain');
            // Clear existing options except "All Domains"
            select.innerHTML = '<option value="all">All Domains</option>';

            allDomains.forEach(domain => {{
                const option = document.createElement('option');
                option.value = domain;
                option.textContent = domain.charAt(0).toUpperCase() + domain.slice(1);
                select.appendChild(option);
            }});
        }}

        function filterTasks() {{
            const tasks = getCurrentTasks();
            return tasks.filter(task => {{
                if (currentFilters.status !== 'all') {{
                    const isSuccess = task.success;
                    if (currentFilters.status === 'success' && !isSuccess) return false;
                    if (currentFilters.status === 'failure' && isSuccess) return false;
                }}

                if (currentFilters.domain !== 'all' && task.domain !== currentFilters.domain) {{
                    return false;
                }}

                return true;
            }});
        }}

        function renderTaskList() {{
            const filteredTasks = filterTasks();
            const container = document.getElementById('task-list');
            const noTasks = document.getElementById('no-tasks');

            if (filteredTasks.length === 0) {{
                container.innerHTML = '';
                noTasks.style.display = 'block';
                return;
            }}

            noTasks.style.display = 'none';
            container.innerHTML = filteredTasks.map(task => renderTaskItem(task)).join('');

            // Add click handlers
            document.querySelectorAll('.task-header').forEach(header => {{
                header.addEventListener('click', () => {{
                    const item = header.closest('.task-item');
                    item.classList.toggle('expanded');
                }});
            }});
        }}

        function renderTaskItem(task) {{
            const statusClass = task.success ? 'success' : 'failure';
            const statusIcon = task.success ? '✓' : '✗';

            const stepsHtml = task.steps && task.steps.length > 0
                ? task.steps.map(step => renderStep(step, task)).join('')
                : '<div style="padding: 12px; color: var(--text-muted);">No step details available</div>';

            return `
                <div class="task-item" data-task-id="${{task.task_id}}">
                    <div class="task-header">
                        <div class="task-status ${{statusClass}}">${{statusIcon}}</div>
                        <div class="task-info">
                            <div class="task-id">${{task.task_id}}</div>
                            <div class="task-instruction">${{task.instruction}}</div>
                        </div>
                        <div class="task-domain">${{task.domain}}</div>
                        <div class="task-meta">
                            <span>${{task.num_steps}} steps</span>
                            <span>${{task.total_time_seconds.toFixed(2)}}s</span>
                        </div>
                        <div class="task-expand-icon">▶</div>
                    </div>
                    <div class="task-details">
                        <div class="steps-list">
                            ${{stepsHtml}}
                        </div>
                    </div>
                </div>
            `;
        }}

        function renderStep(step, task) {{
            const actionType = step.action.type || 'unknown';
            const actionDetails = formatActionDetails(step.action);
            const runDirName = getCurrentRun().dir_name;

            // Build screenshot path relative to benchmark.html
            const screenshotPath = step.screenshot_path
                ? `benchmark_tasks/${{runDirName}}/${{task.task_id}}/${{step.screenshot_path}}`
                : '';

            const screenshotHtml = screenshotPath
                ? `<img src="${{screenshotPath}}" class="step-screenshot" alt="Step ${{step.step_idx}}" />`
                : '';

            return `
                <div class="step-item">
                    <div class="step-number">Step ${{step.step_idx}}</div>
                    ${{screenshotHtml}}
                    <div class="step-action">
                        <div class="action-type">${{actionType}}</div>
                        <div class="action-details">${{actionDetails}}</div>
                        ${{step.reasoning ? `<div style="margin-top: 8px; font-style: italic; color: var(--text-secondary);">${{step.reasoning}}</div>` : ''}}
                    </div>
                </div>
            `;
        }}

        function formatActionDetails(action) {{
            const parts = [];

            if (action.x !== null && action.y !== null) {{
                parts.push(`x: ${{action.x.toFixed(3)}}, y: ${{action.y.toFixed(3)}}`);
            }}

            if (action.text) {{
                parts.push(`text: "${{action.text}}"`);
            }}

            if (action.key) {{
                parts.push(`key: ${{action.key}}`);
            }}

            if (action.target_node_id) {{
                parts.push(`element: [${{action.target_node_id}}]`);
            }}

            if (action.target_name) {{
                parts.push(`target: ${{action.target_name}}`);
            }}

            return parts.length > 0 ? parts.join(', ') : 'No details';
        }}

        // Initialize on page load
        init();
    </script>
</body>
</html>'''

    return html
