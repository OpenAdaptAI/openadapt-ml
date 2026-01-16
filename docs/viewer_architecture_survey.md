# ML Dashboard/Viewer Architecture Survey

**Date:** January 2026
**Purpose:** Research best practices for maintainable, LLM-friendly viewer architectures to replace the current ~390K lines of embedded HTML/CSS/JS in Python strings.

## Executive Summary

After surveying popular ML tools, Python web frameworks, and LLM-friendly code patterns, we recommend a **hybrid approach** combining:

1. **Jinja2 templates** for HTML structure (with separate CSS/JS files)
2. **Plotly** for interactive visualizations with standalone HTML export
3. **htmx + Alpine.js** for lightweight interactivity without a build step
4. **Vertical slice architecture** for LLM-friendly code organization

This approach prioritizes: separation of concerns, small focused files, no build step required, and the ability to generate fully self-contained HTML files.

---

## 1. How Popular ML Tools Handle Dashboards/Viewers

### MLflow UI

**Architecture:** FastAPI + Uvicorn backend (as of version 3.6), React frontend

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn (migrated from Flask in late 2024) |
| Frontend | React-based SPA |
| Communication | REST APIs with Protocol Buffer definitions |
| Storage | SQLAlchemy (PostgreSQL, MySQL, SQLite) + Artifact stores (S3, GCS, Azure Blob) |

**Key Insight:** Multi-layered architecture separating HTTP handling from business logic and storage backends. The tracking server serves both REST APIs and the UI.

*Sources: [MLflow Architecture Overview](https://mlflow.org/docs/latest/self-hosting/architecture/overview/), [MLflow GitHub](https://github.com/mlflow/mlflow)*

### Weights & Biases

**Architecture:** Two-component model (client SDK + server)

| Component | Technology |
|-----------|------------|
| Backend | Cloud-hosted server with ClickHouse database |
| Frontend | Interactive dashboards with real-time updates |
| Communication | Client SDK with one-line logging integration |

**2025 Updates:** Browser caching for faster workspace reloads, multi-video sync feature, custom run display names.

*Sources: [W&B Reference Architecture](https://docs.wandb.ai/platform/hosting/self-managed/ref-arch), [W&B Documentation](https://docs.wandb.ai/guides/track/log/)*

### TensorBoard

**Architecture:** Python backend + Polymer 1.7 frontend (legacy)

| Component | Technology |
|-----------|------------|
| Backend | Python REST API |
| Frontend | Polymer v1 Web Components (now legacy, LitElement recommended) |
| Data Storage | Protobuf events files |
| Plugin System | Dashboards as plugins with Python + Polymer |

**Key Insight:** Plugin-based architecture allows extensibility, but Polymer is now deprecated in favor of LitElement.

*Sources: [TensorBoard GitHub](https://github.com/tensorflow/tensorboard), [G-Research TensorBoard Guide](https://www.gresearch.com/news/working-with-tensorflows-tensorboard-tool/)*

### Streamlit

**Architecture:** Python backend + React frontend with WebSocket communication

| Component | Technology |
|-----------|------------|
| Backend | Tornado web server, Python script execution |
| Frontend | React 18 with Vite build system |
| Communication | WebSockets with Protocol Buffers |
| State | Reactive execution model (script re-runs on interaction) |

**Key Components:**
- `ForwardMsg`: Backend-to-frontend (UI deltas)
- `BackMsg`: Frontend-to-backend (user interactions)
- Frontend assets bundled into Python wheel

*Sources: [Streamlit Architecture Docs](https://docs.streamlit.io/develop/concepts/architecture/architecture), [Streamlit DeepWiki](https://deepwiki.com/streamlit/streamlit)*

### Gradio

**Architecture:** Two codebases communicating via HTTP/SSE

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python functions |
| Frontend | Svelte 5 + SvelteKit (pnpm monorepo) |
| Communication | HTTP + Server-Sent Events (SSE) |
| Component Model | 40+ component packages (@gradio/*) |

**Key Features:**
- `preprocess()`: Converts browser data to Python types
- `postprocess()`: Converts Python return values for frontend display
- Configuration JSON drives dynamic UI rendering

*Sources: [Gradio DeepWiki](https://deepwiki.com/gradio-app/gradio), [Gradio Frontend Guide](https://www.gradio.app/guides/frontend)*

### Panel (HoloViz)

**Architecture:** Built on Bokeh + Param

| Component | Technology |
|-----------|------------|
| Backend | Tornado (default), Flask, Django, or FastAPI |
| Frontend | Bokeh widgets + visualization libraries |
| Communication | WebSocket-based Bokeh server |
| Deployment | Server-based or WASM via Pyodide/PyScript |

**Key Insight:** Reactive programming model where UI components auto-update when data changes. Works with 15+ visualization libraries.

*Sources: [Panel Overview](https://panel.holoviz.org/), [Panel GitHub](https://github.com/holoviz/panel)*

---

## 2. Python + Web UI Architecture Patterns

### Pattern Comparison

| Pattern | Description | Pros | Cons |
|---------|-------------|------|------|
| **Server-Rendered (Jinja2)** | HTML generated server-side | Fast TTFB (45ms vs 250ms), SEO-friendly, simpler mental model | Less interactive, page reloads |
| **SPA (React/Vue)** | Client-side rendering with API | Rich interactivity, real-time updates | Larger bundle size, slower initial load, build step required |
| **htmx + Alpine.js** | Progressive enhancement | ~29KB total, no build step, works without JS | Less ecosystem support, newer pattern |
| **Hybrid** | Server-rendered with JS enhancement | Best of both worlds | More complexity to manage |
| **Static Generation** | Pre-built HTML files | Fastest serving, works offline | No server-side computation |

### Performance Comparison (2025 HTTPArchive)

| Metric | Jinja2+Bootstrap | React SPA |
|--------|------------------|-----------|
| TTFB | 45ms | 250ms |
| Client-side bloat | 70% less | Baseline |
| Memory (JS heap) | 50MB peak | 200MB |

*Sources: [InfoWorld htmx/Alpine](https://www.infoworld.com/article/3856520/htmx-and-alpine-js-how-to-combine-two-great-lean-front-ends.html), [Bootstrap Python Components 2025](https://johal.in/bootstrap-python-components-jinja2-templating-for-responsive-design-systems-2025/)*

### htmx + Alpine.js for Python Backends

**htmx** (~14KB): Server-driven UI updates via HTML attributes
- Makes HTTP requests and swaps response HTML into the page
- State lives on the server
- No JavaScript required for basic interactivity

**Alpine.js** (~15KB): Client-side reactivity
- Client-side state management
- Dropdowns, accordions, filters
- Complements htmx perfectly

**Combined benefits:**
- No build step required
- Progressive enhancement (works without JS)
- Django/Flask/FastAPI compatible
- Modern UX with traditional server-rendered simplicity

*Sources: [SaaS Pegasus Django Guide](https://www.saaspegasus.com/guides/modern-javascript-for-django-developers/htmx-alpine/), [Django htmx Alpine GitHub](https://github.com/arcanemachine/django-htmx-alpine)*

---

## 3. LLM/Claude-Friendly Architecture Patterns

### Key Principles

| Principle | Description | Why It Matters for LLMs |
|-----------|-------------|------------------------|
| **Vertical Slice Architecture** | Organize by feature, not layer | Context isolation - AI understands self-contained features |
| **Small, focused files** | Single responsibility per file | Fits in context window, easier to modify |
| **Separation of concerns** | Data, logic, presentation separate | Clear modification targets |
| **Standard patterns** | Avoid custom DSLs | LLMs trained on standard patterns |
| **Type safety** | Type hints, validation | Clearer contracts, fewer errors |
| **Explicit over implicit** | Clear data flow | Easier to trace and modify |

### Architecture Documentation (Critical for LLM Assistance)

Create these files at project root:

1. **`ARCHITECTURE.md`**: Data structures, data flow, module responsibilities
2. **`guidelines.md`**: Coding conventions, naming patterns
3. **`PR.md`**: Current work context (updated per task)

*Sources: [AI Coding Workflow 2026](https://addyosmani.com/blog/ai-coding-workflow/), [JetBrains AI Guidelines](https://blog.jetbrains.com/idea/2025/05/coding-guidelines-for-your-ai-agents/)*

### LLM-Friendly File Structure

```
# BAD: Monolithic files mixing concerns
viewers/
    benchmark_viewer.py  # 15K lines with embedded HTML/CSS/JS

# GOOD: Vertical slices with separation
viewers/
    benchmark/
        __init__.py
        data.py           # Data loading/processing
        templates/
            base.html
            components/
                chart.html
                table.html
        static/
            styles.css
            interactions.js
        generator.py      # HTML generation logic
```

### Key Recommendations from Research

1. **Plan before coding** - Create architecture docs first
2. **Work incrementally** - Small, testable changes
3. **Keep files under 500 lines** - Fits in LLM context
4. **Use consistent naming** - Predictable patterns
5. **Document the "why"** - Comments explain intent, not mechanics
6. **Regular refactoring** - Prevent entropy accumulation

*Sources: [Medium LLM Coding Workflow](https://medium.com/@wojtek.jurkowlaniec/coding-workflow-with-llm-on-larger-projects-87dd2bf6fd2c), [Graphite Best Practices](https://graphite.com/guides/best-practices-ai-coding-assistants)*

---

## 4. Lightweight Standalone HTML Viewer Options

### Plotly Standalone HTML

**Best for:** Interactive charts that work offline

```python
import plotly.express as px

fig = px.scatter(df, x="x", y="y")
fig.write_html(
    "chart.html",
    include_plotlyjs=True,  # Self-contained (~3-5MB)
    full_html=True
)
```

**Options:**
| Parameter | Size | Offline? |
|-----------|------|----------|
| `include_plotlyjs=True` | ~5MB | Yes |
| `include_plotlyjs='cdn'` | ~10KB | No (CDN) |
| `include_plotlyjs='directory'` | ~10KB | Yes (separate file) |

*Sources: [Plotly HTML Export](https://plotly.com/python/interactive-html-export/), [Plotly Community](https://community.plotly.com/t/standalone-html-files-with-interactivity/9072)*

### CDN-Loaded Frameworks for Standalone HTML

For standalone HTML files that need interactivity:

```html
<!-- Alpine.js - Lightweight reactivity -->
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>

<!-- Chart.js - Simple charts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- Vue.js - More complex UIs -->
<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
```

### Data Embedding Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Inline JSON** | Single file, works offline | Large file size |
| **Separate JSON** | Smaller HTML, reusable data | Two files to manage |
| **CDN resources** | Smallest HTML | Requires internet |
| **Base64 assets** | Images embedded | Increases size significantly |

**Recommended: Inline JSON for data, CDN for libraries (with fallback)**

```html
<script>
  const DATA = {{ data | tojson }};
</script>
<script src="https://cdn.example.com/lib.js"></script>
<script>
  // Fallback if CDN fails
  if (typeof Library === 'undefined') {
    document.write('<script src="lib.local.js"><\/script>');
  }
</script>
```

---

## 5. Python Libraries for Dashboard Generation

### HTML Generation Libraries

| Library | Approach | Best For |
|---------|----------|----------|
| **Jinja2** | Template engine | Complex HTML with inheritance |
| **Dominate** | DOM-like Python API | Programmatic HTML building |
| **Yattag** | Tag/text functions | Form handling with defaults |
| **Airium** | Context managers | Has reverse translator (HTML to Python) |

**Recommendation:** Jinja2 for templates + Dominate for programmatic generation

*Sources: [Yattag](https://www.yattag.org/), [Dominate GitHub](https://github.com/Knio/dominate)*

### Full Framework Comparison

| Framework | Backend | Learning Curve | Best For | Standalone HTML? |
|-----------|---------|----------------|----------|------------------|
| **Streamlit** | Tornado + React | Low | Quick prototypes | No |
| **Gradio** | FastAPI + Svelte | Low | ML demos | No |
| **Panel** | Bokeh/Tornado | Medium | Data dashboards | Yes (WASM) |
| **NiceGUI** | FastAPI + Vue | Medium | Internal tools | No |
| **Reflex** | FastAPI + React | Higher | Full web apps | Yes (static export) |
| **Plotly/Dash** | Flask/Dash | Medium | Data viz | Yes (partial) |

### Simplest Solution That Works

For **standalone HTML viewers** (your use case):

1. **Jinja2** - Template structure
2. **Plotly** - Interactive charts (with `write_html()`)
3. **Alpine.js via CDN** - Client-side interactivity
4. **Inline CSS/JS** or CDN - Styling and behavior

No build step, no server required for viewing, fully self-contained files.

---

## 6. Recommended Architecture for openadapt-viewer

### Design Goals

1. **Maintainability** - Easy for humans and LLMs to modify
2. **Separation of concerns** - Data, templates, styles, scripts separate
3. **Standalone capability** - Generate self-contained HTML files
4. **No build step** - CDN-loaded libraries, no webpack/vite
5. **Progressive enhancement** - Works without JS, enhanced with JS

### Recommended Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Data Processing | Pure Python | Type-safe, testable |
| HTML Structure | Jinja2 templates | Industry standard, well-understood |
| Visualization | Plotly | Best standalone export support |
| Styling | Tailwind CSS (CDN) | Utility-first, no build |
| Interactivity | Alpine.js (CDN) | Lightweight, declarative |
| Enhanced UX | htmx (optional) | Server-driven updates if needed |

### Proposed File Structure

```
openadapt-viewer/
├── pyproject.toml
├── ARCHITECTURE.md              # LLM context file
├── guidelines.md                # Coding conventions
│
├── src/
│   └── openadapt_viewer/
│       ├── __init__.py
│       ├── cli.py               # CLI entry points
│       │
│       ├── core/                # Shared utilities
│       │   ├── __init__.py
│       │   ├── types.py         # Pydantic models, type definitions
│       │   ├── data_loader.py   # Common data loading utilities
│       │   └── html_builder.py  # Jinja2 environment setup
│       │
│       ├── templates/           # Jinja2 templates
│       │   ├── base.html        # Base template with CDN imports
│       │   ├── components/      # Reusable HTML components
│       │   │   ├── header.html
│       │   │   ├── chart.html
│       │   │   ├── table.html
│       │   │   ├── image_grid.html
│       │   │   └── navigation.html
│       │   └── layouts/         # Page layouts
│       │       ├── single_page.html
│       │       └── multi_tab.html
│       │
│       ├── static/              # CSS and JS (kept minimal)
│       │   ├── styles/
│       │   │   ├── base.css     # Custom styles beyond Tailwind
│       │   │   └── themes.css   # Light/dark theme variables
│       │   └── scripts/
│       │       ├── interactions.js  # Alpine.js components
│       │       └── charts.js        # Plotly configuration helpers
│       │
│       └── viewers/             # Vertical slices by viewer type
│           ├── __init__.py
│           │
│           ├── benchmark/       # Benchmark viewer
│           │   ├── __init__.py
│           │   ├── data.py      # Data models and loading
│           │   ├── charts.py    # Plotly figure builders
│           │   ├── generator.py # HTML generation logic
│           │   └── templates/   # Benchmark-specific templates
│           │       ├── overview.html
│           │       ├── run_detail.html
│           │       └── comparison.html
│           │
│           ├── recording/       # Recording viewer
│           │   ├── __init__.py
│           │   ├── data.py
│           │   ├── charts.py
│           │   ├── generator.py
│           │   └── templates/
│           │       ├── recording.html
│           │       └── action_detail.html
│           │
│           └── training/        # Training viewer
│               ├── __init__.py
│               ├── data.py
│               ├── charts.py
│               ├── generator.py
│               └── templates/
│                   ├── metrics.html
│                   └── loss_curves.html
│
└── tests/
    ├── conftest.py
    ├── test_core/
    ├── test_viewers/
    └── fixtures/
        └── sample_data/
```

### Key Design Decisions

#### 1. Vertical Slice Organization
Each viewer type is a self-contained module with its own data loading, chart building, and template rendering. This allows:
- LLMs to understand and modify one viewer without loading entire codebase
- Independent development and testing
- Clear ownership of functionality

#### 2. Template Inheritance

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}OpenAdapt Viewer{% endblock %}</title>

    <!-- CDN imports -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>

    {% block head %}{% endblock %}
</head>
<body class="bg-gray-100 dark:bg-gray-900">
    {% include 'components/header.html' %}

    <main class="container mx-auto p-4">
        {% block content %}{% endblock %}
    </main>

    {% block scripts %}{% endblock %}
</body>
</html>
```

#### 3. Data/Presentation Separation

```python
# viewers/benchmark/data.py
from pydantic import BaseModel
from typing import List, Optional

class BenchmarkRun(BaseModel):
    """Strongly typed benchmark run data."""
    run_id: str
    task_name: str
    metrics: dict
    timestamps: List[float]
    # ... other fields

def load_benchmark_data(path: str) -> List[BenchmarkRun]:
    """Load and validate benchmark data."""
    # Pure data loading, no HTML concerns
    ...
```

```python
# viewers/benchmark/generator.py
from jinja2 import Environment, PackageLoader
from .data import load_benchmark_data
from .charts import create_metrics_chart

def generate_benchmark_html(data_path: str, output_path: str) -> None:
    """Generate standalone benchmark HTML file."""
    env = Environment(loader=PackageLoader('openadapt_viewer', 'templates'))

    # Load data
    runs = load_benchmark_data(data_path)

    # Create visualizations
    metrics_chart = create_metrics_chart(runs)

    # Render template
    template = env.get_template('viewers/benchmark/overview.html')
    html = template.render(
        runs=runs,
        metrics_chart=metrics_chart.to_html(include_plotlyjs=False),
    )

    # Write standalone file
    with open(output_path, 'w') as f:
        f.write(html)
```

#### 4. Standalone HTML Generation

For fully self-contained files:

```python
def generate_standalone_html(data_path: str, output_path: str) -> None:
    """Generate a single HTML file that works offline."""
    # ... generate content ...

    # Inline Plotly.js for offline use
    plotly_js = requests.get('https://cdn.plot.ly/plotly-2.32.0.min.js').text

    template = env.get_template('standalone.html')
    html = template.render(
        content=content,
        plotly_js=plotly_js,  # Inlined
        data_json=json.dumps(data),  # Embedded data
    )
```

### Migration Path

1. **Phase 1: Extract templates** (Week 1-2)
   - Move HTML from Python strings to Jinja2 templates
   - Keep existing data loading logic
   - Verify output matches current viewers

2. **Phase 2: Separate CSS/JS** (Week 2-3)
   - Extract inline styles to CSS files
   - Extract inline scripts to JS files
   - Use CDN for libraries

3. **Phase 3: Refactor data layer** (Week 3-4)
   - Create Pydantic models for data
   - Separate data loading from HTML generation
   - Add type hints throughout

4. **Phase 4: Vertical slices** (Week 4-5)
   - Reorganize into feature-based modules
   - Create viewer-specific templates
   - Add comprehensive tests

---

## Pros/Cons Summary Table

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Current (embedded HTML)** | Single file, no dependencies | Unmaintainable, LLM-hostile, duplicated code | Nothing (legacy) |
| **Jinja2 + CDN libs** | Separation of concerns, no build step, LLM-friendly | Limited interactivity, larger HTML files | Standalone viewers |
| **Streamlit** | Rapid development, Python-only | Requires server, no standalone export | Live dashboards |
| **Gradio** | ML-focused, easy sharing | Limited customization, requires server | ML demos |
| **React SPA** | Rich interactivity, large ecosystem | Build step, larger bundle, separate codebase | Complex apps |
| **htmx + Alpine** | Lightweight, progressive enhancement | Newer ecosystem, fewer examples | Enhanced server apps |

---

## Conclusion

For the openadapt-viewer package, we recommend:

1. **Jinja2 templates** with template inheritance for HTML structure
2. **Separate CSS/JS files** with CDN-loaded libraries (Tailwind, Alpine.js, Plotly)
3. **Vertical slice architecture** organizing code by viewer type
4. **Pydantic models** for type-safe data handling
5. **Plotly** for interactive visualizations with standalone export capability

This approach:
- Reduces the ~390K lines of embedded HTML to ~50K lines of properly organized code
- Makes each viewer independently understandable and modifiable
- Supports both server-rendered and standalone HTML output
- Requires no build step or complex toolchain
- Is well-understood by LLMs due to standard patterns

---

## References

### ML Tools
- [MLflow Architecture](https://mlflow.org/docs/latest/self-hosting/architecture/overview/)
- [W&B Reference Architecture](https://docs.wandb.ai/platform/hosting/self-managed/ref-arch)
- [TensorBoard GitHub](https://github.com/tensorflow/tensorboard)
- [Streamlit Architecture](https://docs.streamlit.io/develop/concepts/architecture/architecture)
- [Gradio Documentation](https://www.gradio.app/guides/frontend)
- [Panel HoloViz](https://panel.holoviz.org/)

### Architecture Patterns
- [htmx + Alpine.js Guide](https://www.infoworld.com/article/3856520/htmx-and-alpine-js-how-to-combine-two-great-lean-front-ends.html)
- [Django htmx Alpine](https://www.saaspegasus.com/guides/modern-javascript-for-django-developers/htmx-alpine/)
- [AI Coding Workflow 2026](https://addyosmani.com/blog/ai-coding-workflow/)
- [JetBrains AI Guidelines](https://blog.jetbrains.com/idea/2025/05/coding-guidelines-for-your-ai-agents/)
- [Graphite AI Best Practices](https://graphite.com/guides/best-practices-ai-coding-assistants)

### Libraries
- [Plotly HTML Export](https://plotly.com/python/interactive-html-export/)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/en/stable/templates/)
- [Dominate GitHub](https://github.com/Knio/dominate)
- [Alpine.js](https://alpinejs.dev/)
- [htmx](https://htmx.org/)
