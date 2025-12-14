# openadapt-privacy Implementation Plan

This document outlines the plan to extract privacy/scrubbing functionality from OpenAdapt into a standalone `openadapt-privacy` package.

## Why Privacy Before Recording

1. **Dependency ordering**: `openadapt-privacy` is logically independent (input: text/images → output: scrubbed text/images). Recording is more coupled to OS APIs and client architecture.

2. **Clean boundaries**: If privacy exists first, recording can depend on it cleanly. Otherwise scrubbing gets baked into recording.

3. **Lower risk**: Privacy is "generic infra" - easy to defend as Background IP. Recording overlaps more with client capture stacks.

4. **Quick win**: 80% of the code already exists in OpenAdapt. Extraction is straightforward.

## Repository Structure

```
openadapt-privacy/
├── LICENSE                          # MIT or Apache-2 (MLDSAI Inc.)
├── README.md
├── pyproject.toml                   # name = "openadapt-privacy"
│
├── openadapt_privacy/
│   ├── __init__.py
│   ├── config.py                    # Privacy-specific config (scrub chars, keys, entities)
│   ├── types.py                     # Modality enum, typed helpers
│   ├── base.py                      # ScrubbingProvider, TextScrubbingMixin, Factory
│   │
│   ├── providers/
│   │   ├── __init__.py              # ScrubProvider enum
│   │   └── presidio.py              # PresidioScrubbingProvider
│   │
│   └── pipelines/
│       ├── __init__.py
│       ├── dicts.py                 # Helpers over nested dicts/lists
│       └── media.py                 # Generic mp4/pdf/image scrub helpers
│
├── tests/
│   ├── test_base.py
│   ├── test_presidio_text.py
│   ├── test_presidio_image.py
│   └── test_dict_pipeline.py
│
└── examples/
    ├── scrub_text_dicts.py
    └── scrub_images_from_folder.py
```

## Migration from OpenAdapt

### 1. base.py → openadapt_privacy/base.py

Move these classes wholesale:
- `Modality` - Enum for modality types (TEXT, PIL_IMAGE, PDF, MP4)
- `ScrubbingProvider` - Base class with abstract scrub methods
- `TextScrubbingMixin` - Mixin with `scrub_dict`, `scrub_list_dicts`, etc.
- `ScrubbingProviderFactory` - Factory for getting providers by modality

**Changes needed:**
- Replace `from openadapt.config import config` with local config module
- Remove any OpenAdapt-specific imports

### 2. config.py (new)

```python
# openadapt_privacy/config.py
from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class PrivacyConfig:
    """Configuration for privacy scrubbing."""

    # Character used to replace scrubbed text
    SCRUB_CHAR: str = "█"

    # Keys in dicts that should be scrubbed
    SCRUB_KEYS_HTML: list[str] = field(default_factory=lambda: [
        "text", "canonical_text", "value", "tooltip", "title"
    ])

    # Action text formatting (for text/canonical_text handling)
    ACTION_TEXT_NAME_PREFIX: str = "<"
    ACTION_TEXT_NAME_SUFFIX: str = ">"
    ACTION_TEXT_SEP: str = "-"

    # Presidio-specific config
    SCRUB_PRESIDIO_IGNORE_ENTITIES: Sequence[str] = ()
    SCRUB_CONFIG_TRF: dict | None = None
    SCRUB_LANGUAGE: str = "en"
    SCRUB_FILL_COLOR: int = 0  # BGR black

    # SpaCy model
    SPACY_MODEL_NAME: str = "en_core_web_trf"

# Global default instance
config = PrivacyConfig()
```

### 3. providers/presidio.py

Move `PresidioScrubbingProvider` with these changes:

**Remove OpenAdapt dependencies:**
- `openadapt.build_utils.redirect_stdout_stderr` → drop or reimplement locally
- `openadapt.custom_logger.logger` → `logging.getLogger(__name__)`
- `openadapt.spacy_model_helpers.download_spacy_model` → inline implementation
- `openadapt.config.config` → local config

**Local spacy model helper:**
```python
def _ensure_spacy_model(model_name: str) -> None:
    """Download spaCy model if not installed."""
    if not spacy.util.is_package(model_name):
        logger.info(f"Downloading {model_name} model...")
        spacy.cli.download(model_name)
```

**Simplified provider:**
```python
class PresidioScrubbingProvider(ScrubbingProvider, TextScrubbingMixin):
    """Presidio-based scrubbing provider."""

    name: str = "presidio"
    capabilities: List[str] = [Modality.TEXT, Modality.PIL_IMAGE]

    def scrub_text(self, text: str, is_separated: bool = False) -> str:
        """Scrub PII from text using Presidio."""
        ...

    def scrub_image(self, image: Image.Image, fill_color: int = 0) -> Image.Image:
        """Scrub PII from image using Presidio Image Redactor."""
        ...
```

### 4. pipelines/dicts.py

Thin wrapper around `TextScrubbingMixin` methods:

```python
from typing import Any
from ..base import TextScrubbingMixin, ScrubbingProvider

class DictScrubber(TextScrubbingMixin):
    """Scrub nested dicts using a ScrubbingProvider."""

    def __init__(self, scrubber: ScrubbingProvider):
        self._scrubber = scrubber

    def scrub_text(self, text: str, is_separated: bool = False) -> str:
        return self._scrubber.scrub_text(text, is_separated=is_separated)

def scrub_dict(input_dict: dict[str, Any], scrubber: ScrubbingProvider) -> dict[str, Any]:
    """Convenience function to scrub a dict."""
    helper = DictScrubber(scrubber)
    return helper.scrub_dict(input_dict)

def scrub_list_dicts(input_list: list[dict], scrubber: ScrubbingProvider) -> list[dict]:
    """Convenience function to scrub a list of dicts."""
    helper = DictScrubber(scrubber)
    return helper.scrub_list_dicts(input_list)
```

## What Stays in OpenAdapt

The ORM models keep their `.scrub()` methods but become clients of `openadapt-privacy`:

```python
# openadapt/models.py

from openadapt_privacy.base import ScrubbingProvider
from openadapt_privacy.providers.presidio import PresidioScrubbingProvider

class Recording(db.Base):
    def scrub(self, scrubber: ScrubbingProvider | None = None) -> None:
        scrubber = scrubber or PresidioScrubbingProvider()
        self.task_description = scrubber.scrub_text(self.task_description)

class ActionEvent(db.Base):
    def scrub(self, scrubber: ScrubbingProvider) -> None:
        self.scrubbed_text = scrubber.scrub_text(self.text, is_separated=True)
        self.scrubbed_canonical_text = scrubber.scrub_text(
            self.canonical_text, is_separated=True
        )
        self.key_char = scrubber.scrub_text(self.key_char)
        # ...

class Screenshot(db.Base):
    def scrub(self, scrubber: ScrubbingProvider) -> None:
        scrubbed_image = scrubber.scrub_image(self.image)
        self.png_data = self.convert_png_to_binary(scrubbed_image)
```

## Dependencies

```toml
# pyproject.toml
[project]
name = "openadapt-privacy"
version = "0.1.0"
description = "Privacy scrubbing for GUI automation data"
requires-python = ">=3.10"
license = "MIT"
authors = [
    {name = "MLDSAI Inc."}
]

dependencies = [
    "pillow>=10.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
presidio = [
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
    "presidio-image-redactor>=0.0.50",
    "spacy>=3.7.0",
    "spacy-transformers>=1.3.0",
]

dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
```

## Implementation Phases

### Phase 1: Core extraction (target: 1 day)
- [ ] Create `openadapt-privacy` repository
- [ ] Extract `base.py` with config decoupling
- [ ] Extract `providers/presidio.py`
- [ ] Basic tests for text scrubbing

### Phase 2: Pipelines and integration (target: 1 day)
- [ ] Add `pipelines/dicts.py`
- [ ] Add `pipelines/media.py` (image handling)
- [ ] Update OpenAdapt models to use the new package
- [ ] Integration tests

### Phase 3: Documentation and release (target: 0.5 day)
- [ ] README with usage examples
- [ ] GitHub Actions for CI
- [ ] PyPI release (or internal package registry)

## Usage Example

```python
from openadapt_privacy.providers.presidio import PresidioScrubbingProvider
from openadapt_privacy.pipelines.dicts import scrub_dict

# Initialize scrubber
scrubber = PresidioScrubbingProvider()

# Scrub text
text = "My email is john@example.com and my SSN is 123-45-6789"
scrubbed = scrubber.scrub_text(text)
# Output: "My email is <EMAIL> and my SSN is <US_SSN>"

# Scrub nested dict
event = {
    "text": "Contact me at john@example.com",
    "metadata": {
        "value": "SSN: 123-45-6789",
        "title": "User John Doe"
    }
}
scrubbed_event = scrub_dict(event, scrubber)

# Scrub image
from PIL import Image
image = Image.open("screenshot.png")
scrubbed_image = scrubber.scrub_image(image)
```

## Notes

### Coordinate Normalization (not privacy-related)

The coordinate normalization discussion from the SOM report is separate from privacy:
- Fine-tuned models don't need fallback normalization
- SoM mode avoids coordinates entirely
- This is handled in `openadapt-ml`, not `openadapt-privacy`

### Future Providers

The architecture supports adding more scrubbing providers:
- `PrivateAIScrubbingProvider` (cloud API)
- `AWSComprehendScrubbingProvider` (AWS Comprehend Medical)
- `LocalLLMScrubbingProvider` (Phi-3/Llama for offline PII detection)

Each would implement `ScrubbingProvider` and register via the factory.
