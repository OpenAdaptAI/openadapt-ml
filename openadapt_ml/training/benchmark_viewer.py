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
    """Return HTML for background tasks panel with JS polling."""
    return '''
    <div class="tasks-panel" id="tasks-panel">
        <div class="tasks-header">
            <div class="tasks-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                </svg>
                Background Tasks
            </div>
            <span class="tasks-refresh" id="tasks-refresh-time">Checking...</span>
        </div>
        <div id="tasks-list">
            <div class="no-tasks">Checking for active tasks...</div>
        </div>
    </div>

    <script>
        async function fetchBackgroundTasks() {
            try {
                const response = await fetch('/api/tasks?' + Date.now());
                if (response.ok) {
                    const tasks = await response.json();
                    renderBackgroundTasks(tasks);
                    document.getElementById('tasks-refresh-time').textContent =
                        'Updated ' + new Date().toLocaleTimeString();
                }
            } catch (e) {
                console.log('Tasks API unavailable:', e);
                document.getElementById('tasks-list').innerHTML =
                    '<div class="no-tasks">Tasks API not available</div>';
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
                const phase = task.phase || task.metadata?.phase || 'unknown';
                const phaseLabel = phaseLabels[phase] || phase;

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
                const progressLabel = task.status === 'completed'
                    ? 'Complete'
                    : `Setup phase progress: ${progressPercent.toFixed(0)}%`;

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
                            <span>${task.status}</span>
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
    '''


def _get_live_evaluation_panel_html() -> str:
    """Return HTML for live evaluation panel with JS polling."""
    return '''
    <div class="live-eval-panel" id="live-eval-panel">
        <div class="live-eval-header">
            <div class="live-eval-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
                </svg>
                Live Evaluation
            </div>
            <span class="live-eval-refresh" id="live-eval-refresh-time">Checking...</span>
        </div>
        <div id="live-eval-content">
            <div class="live-eval-idle">No evaluation running</div>
        </div>
    </div>

    <script>
        async function fetchLiveEvaluation() {
            try {
                const response = await fetch('/api/benchmark-live?' + Date.now());
                if (response.ok) {
                    const state = await response.json();
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

        // Initial fetch and poll every 2 seconds
        fetchLiveEvaluation();
        setInterval(fetchLiveEvaluation, 2000);
    </script>
    '''


def _get_azure_jobs_panel_css() -> str:
    """Return CSS for the Azure jobs status panel."""
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
        .azure-jobs-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .azure-job-item {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .azure-job-item:last-child {
            margin-bottom: 0;
        }
        .azure-job-status {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 120px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.provisioning {
            background: #f59e0b;
        }
        .status-dot.running {
            background: #3b82f6;
        }
        .status-dot.completed {
            background: #10b981;
            animation: none;
        }
        .status-dot.failed {
            background: #ef4444;
            animation: none;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .azure-job-info {
            flex: 1;
            min-width: 0;
        }
        .azure-job-id {
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.85rem;
            color: var(--text-primary);
        }
        .azure-job-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        .azure-job-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
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
        }
        .no-jobs {
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    '''


def _get_azure_jobs_panel_html() -> str:
    """Return HTML for the Azure jobs status panel with JS polling."""
    return '''
    <div class="azure-jobs-panel" id="azure-jobs-panel">
        <div class="azure-jobs-header">
            <div class="azure-jobs-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                </svg>
                Azure Jobs
            </div>
            <span class="azure-jobs-refresh" id="jobs-refresh-time">Checking...</span>
        </div>
        <div id="azure-jobs-list">
            <div class="no-jobs">Loading Azure job status...</div>
        </div>
        <button id="toggle-logs-btn" onclick="toggleLogs()" style="
            margin-top: 12px;
            padding: 6px 12px;
            background: rgba(0, 120, 212, 0.2);
            border: 1px solid rgba(0, 120, 212, 0.4);
            border-radius: 4px;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.8rem;
        ">Show Logs</button>
        <div id="job-logs-panel" style="display: none; margin-top: 12px;">
            <div id="log-job-status" style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 6px;"></div>
            <pre id="job-logs-content" style="
                background: #1a1a1a;
                color: #00ff00;
                padding: 12px;
                border-radius: 4px;
                font-size: 0.7rem;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            ">Click to load logs...</pre>
        </div>
    </div>

    <script>
        // Fetch LIVE Azure job status from API
        async function fetchAzureJobs() {
            try {
                // First try live API endpoint
                const response = await fetch('/api/azure-jobs?' + Date.now());
                if (response.ok) {
                    const jobs = await response.json();
                    if (!jobs.error) {
                        renderAzureJobs(jobs, true);  // isLive=true
                        document.getElementById('jobs-refresh-time').textContent =
                            'Live from Azure • ' + new Date().toLocaleTimeString();
                        return;
                    }
                }
            } catch (e) {
                console.log('Live Azure API unavailable, falling back to cached:', e);
            }

            // Fallback to cached file
            try {
                const response = await fetch('benchmark_results/azure_jobs.json?' + Date.now());
                if (response.ok) {
                    const jobs = await response.json();
                    renderAzureJobs(jobs, false);  // isLive=false
                    document.getElementById('jobs-refresh-time').textContent =
                        'Cached • ' + new Date().toLocaleTimeString();
                    return;
                }
            } catch (e) {
                console.log('Cached file also unavailable:', e);
            }

            document.getElementById('azure-jobs-list').innerHTML =
                '<div class="no-jobs">No Azure jobs found. Run: <code>uv run python -m openadapt_ml.benchmarks.cli run-azure</code></div>';
        }

        function renderAzureJobs(jobs, isLive) {
            if (!jobs || jobs.length === 0) {
                document.getElementById('azure-jobs-list').innerHTML =
                    '<div class="no-jobs">No Azure jobs found. Run: <code>uv run python -m openadapt_ml.benchmarks.cli run-azure</code></div>';
                return;
            }

            const html = jobs.slice(0, 5).map(job => {
                const statusClass = job.status || 'unknown';
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
                    if (job.status === 'running') {
                        elapsedText = elapsedMins < 60
                            ? Math.round(elapsedMins) + 'm'
                            : Math.round(elapsedMins / 60) + 'h ' + Math.round(elapsedMins % 60) + 'm';
                        // Warn if running > 30 mins (WAA tasks typically take 5-10 mins each)
                        if (elapsedMins > 30) {
                            isStuck = true;
                            statusText += ' ⚠️';
                        }
                    }
                }

                // Build metadata
                const meta = [];
                if (elapsedText && job.status === 'running') meta.push(elapsedText);
                if (!isLive && job.num_tasks) meta.push('~' + job.num_tasks + ' tasks');
                if (!isLive && job.workers) meta.push('~' + job.workers + ' workers');
                if (job.results?.success_rate !== undefined) {
                    meta.push((job.results.success_rate * 100).toFixed(1) + '% success');
                }
                if (job.started_at && job.status !== 'running') {
                    const date = new Date(job.started_at);
                    meta.push(date.toLocaleString());
                }
                const metaText = meta.join(' • ');

                // Add warning for stuck jobs
                const stuckWarning = isStuck
                    ? '<div style="color: #ff9800; font-size: 0.7rem; margin-top: 4px;">⚠️ Running > 30min. May be stuck. Consider canceling.</div>'
                    : '';

                return '<div class="azure-job-item" style="' + (isStuck ? 'border-color: #ff9800;' : '') + '">' +
                    '<div class="azure-job-status">' +
                        '<span class="status-dot ' + statusClass + '"></span>' +
                        '<span>' + statusText + '</span>' +
                    '</div>' +
                    '<div class="azure-job-info">' +
                        '<div class="azure-job-id">' + displayName + '</div>' +
                        '<div class="azure-job-meta">' + metaText + '</div>' +
                        stuckWarning +
                    '</div>' +
                    '<a href="' + job.azure_dashboard_url + '" target="_blank" class="azure-job-link">' +
                        'Open in Azure →' +
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

            try {
                const url = currentLogJobId
                    ? '/api/azure-job-logs?job_id=' + currentLogJobId
                    : '/api/azure-job-logs';
                const response = await fetch(url + '&t=' + Date.now());
                if (response.ok) {
                    const data = await response.json();
                    const logEl = document.getElementById('job-logs-content');
                    if (logEl) {
                        logEl.textContent = data.logs || 'No logs available';
                        if (data.command) {
                            logEl.textContent = 'Command: ' + data.command + '\\n\\n' + (data.logs || '');
                        }
                    }
                    const statusEl = document.getElementById('log-job-status');
                    if (statusEl && data.job_id) {
                        statusEl.textContent = data.job_id + ' (' + data.status + ')';
                    }
                }
            } catch (e) {
                console.log('Error fetching logs:', e);
            }
        }

        function toggleLogs() {
            showLogs = !showLogs;
            const panel = document.getElementById('job-logs-panel');
            const btn = document.getElementById('toggle-logs-btn');
            if (panel) {
                panel.style.display = showLogs ? 'block' : 'none';
            }
            if (btn) {
                btn.textContent = showLogs ? 'Hide Logs' : 'Show Logs';
            }
            if (showLogs) fetchJobLogs();
        }

        // Initial fetch and poll every 10 seconds
        fetchAzureJobs();
        setInterval(fetchAzureJobs, 10000);
        setInterval(fetchJobLogs, 5000);  // Poll logs every 5 seconds
    </script>
    '''


def _get_vm_discovery_panel_css() -> str:
    """Return CSS for VM Discovery panel (green/teal themed)."""
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
        .vm-discovery-refresh {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .vm-item {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .vm-item:last-child {
            margin-bottom: 0;
        }
        .vm-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .vm-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-primary);
        }
        .vm-status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
        }
        .vm-status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .vm-status-dot.online {
            background: #10b981;
        }
        .vm-status-dot.offline {
            background: #ef4444;
        }
        .vm-status-dot.unknown {
            background: #6b7280;
        }
        .vm-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
            margin-bottom: 12px;
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
            gap: 8px;
            align-items: center;
        }
        .vm-vnc-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 6px;
            color: #10b981;
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .vm-vnc-link:hover {
            background: rgba(16, 185, 129, 0.3);
            transform: translateY(-1px);
        }
        .vm-waa-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            font-size: 0.8rem;
        }
        .vm-waa-status.ready {
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .vm-waa-status.not-ready {
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        .vm-last-checked {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 8px;
        }
        .no-vms {
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .vm-add-button {
            margin-top: 12px;
            padding: 8px 16px;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 6px;
            color: #10b981;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .vm-add-button:hover {
            background: rgba(16, 185, 129, 0.3);
        }
        .vm-add-form {
            display: none;
            margin-top: 12px;
            padding: 16px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        .vm-add-form.show {
            display: block;
        }
        .vm-form-row {
            margin-bottom: 12px;
        }
        .vm-form-row label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .vm-form-row input {
            width: 100%;
            padding: 6px 10px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }
        .vm-form-actions {
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }
        .vm-form-submit {
            padding: 8px 16px;
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
    """Return HTML for VM Discovery panel with JS polling."""
    return '''
    <div class="vm-discovery-panel" id="vm-discovery-panel">
        <div class="vm-discovery-header">
            <div class="vm-discovery-title">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 3h18v4H3V3zm0 6h18v12H3V9zm2 2v8h14v-8H5zm2 2h4v4H7v-4z"/>
                </svg>
                VM Discovery
            </div>
            <span class="vm-discovery-refresh" id="vm-refresh-time">Checking...</span>
        </div>
        <div id="vm-list">
            <div class="no-vms">Checking for registered VMs...</div>
        </div>
        <button id="vm-add-button" class="vm-add-button" onclick="toggleVMAddForm()">
            + Add VM
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
        async function fetchVMs() {
            try {
                const response = await fetch('/api/vms?' + Date.now());
                if (response.ok) {
                    const vms = await response.json();
                    renderVMs(vms);
                    document.getElementById('vm-refresh-time').textContent =
                        'Updated ' + new Date().toLocaleTimeString();
                }
            } catch (e) {
                console.log('VM API unavailable:', e);
                document.getElementById('vm-list').innerHTML =
                    '<div class="no-vms">VM API not available</div>';
            }
        }

        function renderVMs(vms) {
            const container = document.getElementById('vm-list');

            if (!vms || vms.length === 0) {
                container.innerHTML = '<div class="no-vms">No VMs registered. Click "Add VM" to register one.</div>';
                return;
            }

            const html = vms.map(vm => {
                const statusClass = vm.status || 'unknown';
                const waaStatusClass = vm.waa_probe_status === 'ready' ? 'ready' : 'not-ready';
                const waaStatusIcon = vm.waa_probe_status === 'ready' ? '✓' : '✗';
                const waaStatusText = vm.waa_probe_status === 'ready' ? 'WAA Ready' :
                                     vm.waa_probe_status === 'not responding' ? 'WAA Not Responding' :
                                     vm.waa_probe_status === 'ssh failed' ? 'SSH Failed' : 'Unknown';

                const vncUrl = `http://${vm.ssh_host}:${vm.vnc_port}`;

                return `
                    <div class="vm-item">
                        <div class="vm-item-header">
                            <span class="vm-name">${vm.name}</span>
                            <div class="vm-status-indicator">
                                <div class="vm-status-dot ${statusClass}"></div>
                                <span>${statusClass}</span>
                            </div>
                        </div>
                        <div class="vm-info">
                            <div class="vm-info-item">
                                <span class="vm-info-label">SSH:</span>
                                <span class="vm-info-value">${vm.ssh_user}@${vm.ssh_host}</span>
                            </div>
                            <div class="vm-info-item">
                                <span class="vm-info-label">Container:</span>
                                <span class="vm-info-value">${vm.docker_container}</span>
                            </div>
                        </div>
                        <div class="vm-actions">
                            <a href="${vncUrl}" target="_blank" class="vm-vnc-link">
                                Open VNC →
                            </a>
                            <div class="vm-waa-status ${waaStatusClass}">
                                ${waaStatusIcon} ${waaStatusText}
                            </div>
                        </div>
                        <div class="vm-last-checked">
                            Last checked: ${vm.last_checked ? new Date(vm.last_checked).toLocaleString() : 'Never'}
                        </div>
                    </div>
                `;
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
                    alert('Failed to register VM: Server error');
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
