from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from openadapt_ml.schemas.sessions import Action, Episode, Observation, Session, Step


IMG_WIDTH = 800
IMG_HEIGHT = 600


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:  # type: ignore[name-defined]
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = _load_font(16)
FONT_TITLE = _load_font(24)


def _normalize(x_px: int, y_px: int) -> Tuple[float, float]:
    """Normalize pixel coordinates to [0, 1] relative to image size."""

    return x_px / IMG_WIDTH, y_px / IMG_HEIGHT


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Compute text width/height using textbbox for Pillow compatibility."""

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


@dataclass
class LoginUIElements:
    """Absolute pixel bounds for important interactive regions.

    Bounds are (x, y, w, h) in pixels.
    """

    username_box: Tuple[int, int, int, int]
    password_box: Tuple[int, int, int, int]
    login_button: Tuple[int, int, int, int]


def _compute_login_layout(max_offset: int = 10, jitter: bool = True) -> LoginUIElements:
    """Sample a login UI layout, optionally with jitter.

    This computes absolute pixel bounds for all key elements once, so that a
    single layout can be reused across all frames in an episode.
    """

    # Username label and box base geometry
    label_x = 200
    uname_label_y = 160
    box_w, box_h = 360, 40
    uname_box_y = uname_label_y + 24

    def _maybe_jitter(x: int, y: int) -> tuple[int, int]:
        if not jitter:
            return x, y
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        jx = max(0, min(IMG_WIDTH, x + dx))
        jy = max(0, min(IMG_HEIGHT, y + dy))
        return jx, jy

    # Username box position
    uname_x, uname_y = _maybe_jitter(label_x, uname_box_y)
    uname_x = max(20, min(IMG_WIDTH - box_w - 20, uname_x))
    uname_y = max(uname_label_y + 10, min(IMG_HEIGHT - box_h - 100, uname_y))
    username_box = (uname_x, uname_y, box_w, box_h)

    # Password label and box
    pw_label_y = uname_y + box_h + 30
    pw_box_y = pw_label_y + 24
    pw_x, pw_y = _maybe_jitter(label_x, pw_box_y)
    pw_x = max(20, min(IMG_WIDTH - box_w - 20, pw_x))
    pw_y = max(pw_label_y + 10, min(IMG_HEIGHT - box_h - 80, pw_y))
    password_box = (pw_x, pw_y, box_w, box_h)

    # Login button
    btn_w, btn_h = 140, 45
    base_btn_x = (IMG_WIDTH - btn_w) // 2
    base_btn_y = pw_y + box_h + 50
    btn_x, btn_y = _maybe_jitter(base_btn_x, base_btn_y)
    btn_x = max(20, min(IMG_WIDTH - btn_w - 20, btn_x))
    btn_y = max(pw_y + box_h + 20, min(IMG_HEIGHT - btn_h - 40, btn_y))
    login_button = (btn_x, btn_y, btn_w, btn_h)

    return LoginUIElements(
        username_box=username_box,
        password_box=password_box,
        login_button=login_button,
    )


def _draw_login_screen(
    username: str = "",
    password: str = "",
    layout: Optional[LoginUIElements] = None,
    jitter: bool = True,
) -> tuple[Image.Image, LoginUIElements]:
    """Draw a simple login screen with slight layout jitter and a decoy button.

    Returns the image and absolute pixel bounds for key interactive elements.
    Bounds are (x, y, w, h).
    """

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)

    # Title
    title_text = "Welcome Back!"
    tw, th = _text_size(draw, title_text, FONT_TITLE)
    tx = (IMG_WIDTH - tw) // 2
    ty = 80
    draw.text((tx, ty), title_text, fill="black", font=FONT_TITLE)

    # Determine layout once; if not provided, sample it (optionally with jitter).
    if layout is None:
        layout = _compute_login_layout(jitter=jitter)

    # Username label and box
    label_x = 200
    uname_label_y = 160
    box_w, box_h = 360, 40
    draw.text((label_x, uname_label_y), "Username:", fill="black", font=FONT)

    uname_x, uname_y, _, _ = layout.username_box
    draw.rectangle(
        [
            (uname_x, uname_y),
            (uname_x + box_w, uname_y + box_h),
        ],
        outline="black",
        fill="white",
    )
    if username:
        draw.text((uname_x + 8, uname_y + 10), username, fill="black", font=FONT)

    # Password label and box
    pw_x, pw_y, _, _ = layout.password_box
    pw_label_y = pw_y - 24
    draw.text((label_x, pw_label_y), "Password:", fill="black", font=FONT)

    draw.rectangle(
        [
            (pw_x, pw_y),
            (pw_x + box_w, pw_y + box_h),
        ],
        outline="black",
        fill="white",
    )
    if password:
        masked = "*" * len(password)
        draw.text((pw_x + 8, pw_y + 10), masked, fill="black", font=FONT)

    # Login button
    btn_x, btn_y, btn_w, btn_h = layout.login_button

    draw.rectangle(
        [
            (btn_x, btn_y),
            (btn_x + btn_w, btn_y + btn_h),
        ],
        outline="black",
        fill="green",
    )
    btn_text = "Login"
    btw, bth = _text_size(draw, btn_text, FONT)
    draw.text(
        (btn_x + (btn_w - btw) // 2, btn_y + (btn_h - bth) // 2),
        btn_text,
        fill="white",
        font=FONT,
    )

    login_button = (btn_x, btn_y, btn_w, btn_h)

    # Decoy clickable button (e.g., Help) in the lower-right area.
    decoy_w, decoy_h = 110, 35
    decoy_x = IMG_WIDTH - decoy_w - 40
    decoy_y = btn_y
    draw.rectangle(
        [
            (decoy_x, decoy_y),
            (decoy_x + decoy_w, decoy_y + decoy_h),
        ],
        outline="black",
        fill=(180, 180, 180),
    )
    decoy_text = "Help"
    dtw, dth = _text_size(draw, decoy_text, FONT)
    draw.text(
        (decoy_x + (decoy_w - dtw) // 2, decoy_y + (decoy_h - dth) // 2),
        decoy_text,
        fill="black",
        font=FONT,
    )

    elements = LoginUIElements(
        username_box=layout.username_box,
        password_box=layout.password_box,
        login_button=login_button,
    )

    return img, elements


def _overlay_som_marks(
    img: Image.Image,
    elements: List[Tuple[int, Tuple[int, int, int, int]]],
) -> Image.Image:
    """Overlay Set-of-Marks numbered labels on interactive elements.

    Uses the style from the SoM paper: black squares with white numbers.
    Labels are positioned at the top-left corner of each element.

    Args:
        img: The base screenshot image.
        elements: List of (index, (x, y, w, h)) tuples for each interactive element.
                  Index is the 1-based element number shown in the label.

    Returns:
        A copy of the image with [1], [2], [3], etc. labels overlaid.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Load a slightly larger font for SoM labels
    try:
        som_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        som_font = ImageFont.load_default()

    for idx, bounds in elements:
        x, y, w, h = bounds
        label = f"[{idx}]"

        # Measure text size
        text_bbox = draw.textbbox((0, 0), label, font=som_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Add padding for the box
        padding = 4
        box_width = text_width + padding * 2
        box_height = text_height + padding * 2

        # Position ABOVE and to the LEFT of the element (not inside)
        # This ensures labels don't obscure content
        box_x = x - 4
        box_y = y - box_height - 2

        # Ensure box stays within image bounds
        if box_y < 0:
            # If no room above, position to the left of the element
            box_y = y + 4
        if box_x < 0:
            box_x = 4

        # Draw black rectangle background (SoM paper style)
        draw.rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            fill="black",
        )

        # Draw white text centered in the box
        text_x = box_x + padding
        text_y = box_y + padding
        draw.text((text_x, text_y), label, fill="white", font=som_font)

    return img


# Element index mapping for the login screen (1-based for human readability)
SOM_USERNAME_FIELD = 1
SOM_PASSWORD_FIELD = 2
SOM_LOGIN_BUTTON = 3


def _draw_logged_in_screen(username: str) -> Image.Image:
    """Simple logged-in confirmation screen."""

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(210, 230, 210))
    draw = ImageDraw.Draw(img)
    text = f"Welcome, {username}!"
    tw, th = _text_size(draw, text, FONT_TITLE)
    tx = (IMG_WIDTH - tw) // 2
    ty = (IMG_HEIGHT - th) // 2
    draw.text((tx, ty), text, fill="darkgreen", font=FONT_TITLE)
    return img


def _save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _center(bounds: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bounds
    cx = x + w // 2
    cy = y + h // 2
    return _normalize(cx, cy)


def _bbox_normalized(bounds: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """Convert pixel bounds (x, y, w, h) to normalized bbox (x_min, y_min, x_max, y_max)."""
    x, y, w, h = bounds
    x_min = x / IMG_WIDTH
    y_min = y / IMG_HEIGHT
    x_max = (x + w) / IMG_WIDTH
    y_max = (y + h) / IMG_HEIGHT
    return (x_min, y_min, x_max, y_max)


def _script_login_episode(
    root: Path,
    episode_id: str,
    username: str,
    password: str,
    jitter: bool = True,
) -> Episode:
    """Create a scripted login episode with a fixed sequence of steps.

    Steps (6 total):
    - Step 0: blank login screen → click username field.
    - Step 1: username field focused → type username.
    - Step 2: username typed → click password field.
    - Step 3: password field focused → type password.
    - Step 4: password typed → click login button.
    - Step 5: logged-in screen → DONE.

    Each step includes bounding boxes for clickable elements to support
    bbox-based click hit evaluation.
    """

    steps: List[Step] = []

    # Sample a single layout for the entire episode (controls jitter vs no-jitter).
    layout = _compute_login_layout(jitter=jitter)

    # Compute normalized bounding boxes for all elements
    username_bbox = _bbox_normalized(layout.username_box)
    password_bbox = _bbox_normalized(layout.password_box)
    login_bbox = _bbox_normalized(layout.login_button)

    # Step 0: blank login screen → click username field
    cx, cy = _center(layout.username_box)
    img0, _ = _draw_login_screen(layout=layout, jitter=False)
    img0_path = root / f"{episode_id}_step_0.png"
    _save_image(img0, img0_path)
    obs0 = Observation(image_path=str(img0_path))
    steps.append(
        Step(
            t=0.0,
            observation=obs0,
            action=Action(type="click", x=cx, y=cy, bbox=username_bbox),
            thought="Focus the username field.",
        )
    )

    # Step 1: username field focused → type username
    img1, _ = _draw_login_screen(username="", layout=layout, jitter=False)
    img1_path = root / f"{episode_id}_step_1.png"
    _save_image(img1, img1_path)
    obs1 = Observation(image_path=str(img1_path))
    steps.append(
        Step(
            t=1.0,
            observation=obs1,
            action=Action(type="type", text=username),
            thought="Type the username.",
        )
    )

    # Step 2: username typed → click password field
    cx_pw, cy_pw = _center(layout.password_box)
    img2, _ = _draw_login_screen(username=username, layout=layout, jitter=False)
    img2_path = root / f"{episode_id}_step_2.png"
    _save_image(img2, img2_path)
    obs2 = Observation(image_path=str(img2_path))
    steps.append(
        Step(
            t=2.0,
            observation=obs2,
            action=Action(type="click", x=cx_pw, y=cy_pw, bbox=password_bbox),
            thought="Focus the password field.",
        )
    )

    # Step 3: password field focused → type password
    img3, _ = _draw_login_screen(username=username, layout=layout, jitter=False)
    img3_path = root / f"{episode_id}_step_3.png"
    _save_image(img3, img3_path)
    obs3 = Observation(image_path=str(img3_path))
    steps.append(
        Step(
            t=3.0,
            observation=obs3,
            action=Action(type="type", text=password),
            thought="Type the password.",
        )
    )

    # Step 4: password typed → click login button
    cx_btn, cy_btn = _center(layout.login_button)
    img4, _ = _draw_login_screen(username=username, password=password, layout=layout, jitter=False)
    img4_path = root / f"{episode_id}_step_4.png"
    _save_image(img4, img4_path)
    obs4 = Observation(image_path=str(img4_path))
    steps.append(
        Step(
            t=4.0,
            observation=obs4,
            action=Action(type="click", x=cx_btn, y=cy_btn, bbox=login_bbox),
            thought="Submit the login form.",
        )
    )

    # Step 5: logged-in screen → DONE
    img5 = _draw_logged_in_screen(username=username)
    img5_path = root / f"{episode_id}_step_5.png"
    _save_image(img5, img5_path)
    obs5 = Observation(image_path=str(img5_path))
    steps.append(
        Step(
            t=5.0,
            observation=obs5,
            action=Action(type="done"),
            thought="Login successful; workflow complete.",
        )
    )

    episode = Episode(
        id=episode_id,
        goal=f"Log in with username '{username}' and password '{password}'",
        steps=steps,
        summary="Successful login via username and password.",
        success=True,
        workflow_id="login_basic",
    )

    return episode


def _script_login_episode_som(
    root: Path,
    episode_id: str,
    username: str,
    password: str,
    jitter: bool = True,
) -> Episode:
    """Create a scripted login episode with Set-of-Marks (SoM) overlay.

    This variant generates screenshots with numbered labels [1], [2], [3] on
    interactive elements, and uses element_index instead of raw coordinates
    for click actions.

    Steps (6 total):
    - Step 0: SoM login screen → click element [1] (username field)
    - Step 1: username field focused → type username
    - Step 2: username typed → click element [2] (password field)
    - Step 3: password field focused → type password
    - Step 4: password typed → click element [3] (login button)
    - Step 5: logged-in screen → DONE
    """

    steps: List[Step] = []

    # Sample a single layout for the entire episode
    layout = _compute_login_layout(jitter=jitter)

    # Compute normalized bounding boxes for all elements
    username_bbox = _bbox_normalized(layout.username_box)
    password_bbox = _bbox_normalized(layout.password_box)
    login_bbox = _bbox_normalized(layout.login_button)

    # Define element mapping for SoM overlay
    som_elements = [
        (SOM_USERNAME_FIELD, layout.username_box),
        (SOM_PASSWORD_FIELD, layout.password_box),
        (SOM_LOGIN_BUTTON, layout.login_button),
    ]

    # Step 0: SoM login screen → click username field [1]
    cx, cy = _center(layout.username_box)
    img0, _ = _draw_login_screen(layout=layout, jitter=False)
    img0_som = _overlay_som_marks(img0, som_elements)
    img0_path = root / f"{episode_id}_step_0.png"
    _save_image(img0_som, img0_path)
    obs0 = Observation(image_path=str(img0_path))
    steps.append(
        Step(
            t=0.0,
            observation=obs0,
            action=Action(
                type="click",
                x=cx,
                y=cy,
                bbox=username_bbox,
                element_index=SOM_USERNAME_FIELD,
            ),
            thought="Focus the username field by clicking element [1].",
        )
    )

    # Step 1: username field focused → type username into element [1]
    img1, _ = _draw_login_screen(username="", layout=layout, jitter=False)
    img1_som = _overlay_som_marks(img1, som_elements)
    img1_path = root / f"{episode_id}_step_1.png"
    _save_image(img1_som, img1_path)
    obs1 = Observation(image_path=str(img1_path))
    steps.append(
        Step(
            t=1.0,
            observation=obs1,
            action=Action(type="type", text=username, element_index=SOM_USERNAME_FIELD),
            thought="Type the username into element [1].",
        )
    )

    # Step 2: username typed → click password field [2]
    cx_pw, cy_pw = _center(layout.password_box)
    img2, _ = _draw_login_screen(username=username, layout=layout, jitter=False)
    img2_som = _overlay_som_marks(img2, som_elements)
    img2_path = root / f"{episode_id}_step_2.png"
    _save_image(img2_som, img2_path)
    obs2 = Observation(image_path=str(img2_path))
    steps.append(
        Step(
            t=2.0,
            observation=obs2,
            action=Action(
                type="click",
                x=cx_pw,
                y=cy_pw,
                bbox=password_bbox,
                element_index=SOM_PASSWORD_FIELD,
            ),
            thought="Focus the password field by clicking element [2].",
        )
    )

    # Step 3: password field focused → type password into element [2]
    img3, _ = _draw_login_screen(username=username, layout=layout, jitter=False)
    img3_som = _overlay_som_marks(img3, som_elements)
    img3_path = root / f"{episode_id}_step_3.png"
    _save_image(img3_som, img3_path)
    obs3 = Observation(image_path=str(img3_path))
    steps.append(
        Step(
            t=3.0,
            observation=obs3,
            action=Action(type="type", text=password, element_index=SOM_PASSWORD_FIELD),
            thought="Type the password into element [2].",
        )
    )

    # Step 4: password typed → click login button [3]
    cx_btn, cy_btn = _center(layout.login_button)
    img4, _ = _draw_login_screen(
        username=username, password=password, layout=layout, jitter=False
    )
    img4_som = _overlay_som_marks(img4, som_elements)
    img4_path = root / f"{episode_id}_step_4.png"
    _save_image(img4_som, img4_path)
    obs4 = Observation(image_path=str(img4_path))
    steps.append(
        Step(
            t=4.0,
            observation=obs4,
            action=Action(
                type="click",
                x=cx_btn,
                y=cy_btn,
                bbox=login_bbox,
                element_index=SOM_LOGIN_BUTTON,
            ),
            thought="Submit the login form by clicking element [3].",
        )
    )

    # Step 5: logged-in screen → DONE (no SoM needed)
    img5 = _draw_logged_in_screen(username=username)
    img5_path = root / f"{episode_id}_step_5.png"
    _save_image(img5, img5_path)
    obs5 = Observation(image_path=str(img5_path))
    steps.append(
        Step(
            t=5.0,
            observation=obs5,
            action=Action(type="done"),
            thought="Login successful; workflow complete.",
        )
    )

    episode = Episode(
        id=episode_id,
        goal=f"Log in with username '{username}' and password '{password}'",
        steps=steps,
        summary="Successful login via username and password (SoM mode).",
        success=True,
        workflow_id="login_basic_som",
    )

    return episode


def generate_synthetic_sessions(
    num_sessions: int = 10,
    seed: int | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    jitter: bool = True,
    use_som: bool = False,
) -> List[Session]:
    """Generate a list of synthetic Sessions with semantic login episodes.

    Each Session currently contains a single login Episode. Images for all
    steps are written to `output_dir` (default: `synthetic_data/` under the
    current working directory).

    Args:
        num_sessions: Number of sessions to generate.
        seed: Random seed for reproducibility.
        output_dir: Directory to write images to.
        jitter: Whether to apply slight position jitter to UI elements.
        use_som: If True, generate Set-of-Marks (SoM) annotated screenshots
                 with numbered element labels [1], [2], [3], and use element
                 indices for click actions instead of raw coordinates.
    """

    if seed is not None:
        random.seed(seed)

    if output_dir is None:
        # Centralize synthetic assets under a single top-level directory.
        # Callers can still override this, but by default we write to
        # `synthetic/data` instead of scattering `synthetic_*` folders.
        suffix = "_som" if use_som else ""
        output_root = Path("synthetic") / f"data{suffix}"
    else:
        output_root = Path(output_dir)

    sessions: List[Session] = []

    for i in range(num_sessions):
        session_id = f"session_{i:04d}"
        episode_id = f"{session_id}_login"
        # Simple deterministic but varied credentials
        username = f"user{i}"
        password = f"pass{i}123"

        session_dir = output_root / session_id

        if use_som:
            episode = _script_login_episode_som(
                session_dir,
                episode_id,
                username,
                password,
                jitter=jitter,
            )
        else:
            episode = _script_login_episode(
                session_dir,
                episode_id,
                username,
                password,
                jitter=jitter,
            )

        session = Session(id=session_id, episodes=[episode], meta={"scenario": "login", "use_som": use_som})
        sessions.append(session)

    return sessions


