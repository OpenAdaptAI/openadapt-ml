from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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


def _draw_login_screen(username: str = "", password: str = "") -> tuple[Image.Image, LoginUIElements]:
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

    # Small jitter helper to keep elements within bounds.
    def _jitter(x: int, y: int, max_offset: int = 10) -> tuple[int, int]:
        dx = random.randint(-max_offset, max_offset)
        dy = random.randint(-max_offset, max_offset)
        jx = max(0, min(IMG_WIDTH, x + dx))
        jy = max(0, min(IMG_HEIGHT, y + dy))
        return jx, jy

    # Username label and box
    label_x = 200
    uname_label_y = 160
    box_w, box_h = 360, 40
    uname_box_y = uname_label_y + 24

    draw.text((label_x, uname_label_y), "Username:", fill="black", font=FONT)

    uname_x, uname_y = _jitter(label_x, uname_box_y)
    uname_x = max(20, min(IMG_WIDTH - box_w - 20, uname_x))
    uname_y = max(uname_label_y + 10, min(IMG_HEIGHT - box_h - 100, uname_y))

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

    username_box = (uname_x, uname_y, box_w, box_h)

    # Password label and box
    pw_label_y = uname_y + box_h + 30
    pw_box_y = pw_label_y + 24
    draw.text((label_x, pw_label_y), "Password:", fill="black", font=FONT)

    pw_x, pw_y = _jitter(label_x, pw_box_y)
    pw_x = max(20, min(IMG_WIDTH - box_w - 20, pw_x))
    pw_y = max(pw_label_y + 10, min(IMG_HEIGHT - box_h - 80, pw_y))

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

    password_box = (pw_x, pw_y, box_w, box_h)

    # Login button
    btn_w, btn_h = 140, 45
    base_btn_x = (IMG_WIDTH - btn_w) // 2
    base_btn_y = pw_y + box_h + 50
    btn_x, btn_y = _jitter(base_btn_x, base_btn_y)
    btn_x = max(20, min(IMG_WIDTH - btn_w - 20, btn_x))
    btn_y = max(pw_y + box_h + 20, min(IMG_HEIGHT - btn_h - 40, btn_y))

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
        username_box=username_box,
        password_box=password_box,
        login_button=login_button,
    )

    return img, elements


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


def _script_login_episode(root: Path, episode_id: str, username: str, password: str) -> Episode:
    """Create a scripted login episode with a fixed sequence of steps.

    Steps:
    - Step 0: blank login screen.
    - Step 1: click username field.
    - Step 2: type username.
    - Step 3: click password field.
    - Step 4: type password.
    - Step 5: click login.
    - Step 6: DONE on logged-in screen.
    """

    steps: List[Step] = []

    # Step 0: initial blank login screen
    img0, elems0 = _draw_login_screen()
    img0_path = root / f"{episode_id}_step_0.png"
    _save_image(img0, img0_path)
    obs0 = Observation(image_path=str(img0_path))
    # No action yet; use a no-op wait
    steps.append(
        Step(
            t=0.0,
            observation=obs0,
            action=Action(type="wait"),
            thought="Initial login screen displayed.",
        )
    )

    # Step 1: click username field
    cx, cy = _center(elems0.username_box)
    img1, elems1 = _draw_login_screen()
    img1_path = root / f"{episode_id}_step_1.png"
    _save_image(img1, img1_path)
    obs1 = Observation(image_path=str(img1_path))
    steps.append(
        Step(
            t=1.0,
            observation=obs1,
            action=Action(type="click", x=cx, y=cy),
            thought="Focus the username field.",
        )
    )

    # Step 2: username typed
    img2, elems2 = _draw_login_screen(username=username)
    img2_path = root / f"{episode_id}_step_2.png"
    _save_image(img2, img2_path)
    obs2 = Observation(image_path=str(img2_path))
    steps.append(
        Step(
            t=2.0,
            observation=obs2,
            action=Action(type="type", text=username),
            thought="Type the username.",
        )
    )

    # Step 3: click password field
    cx_pw, cy_pw = _center(elems2.password_box)
    img3, elems3 = _draw_login_screen(username=username)
    img3_path = root / f"{episode_id}_step_3.png"
    _save_image(img3, img3_path)
    obs3 = Observation(image_path=str(img3_path))
    steps.append(
        Step(
            t=3.0,
            observation=obs3,
            action=Action(type="click", x=cx_pw, y=cy_pw),
            thought="Focus the password field.",
        )
    )

    # Step 4: password typed (masked visually)
    img4, elems4 = _draw_login_screen(username=username, password=password)
    img4_path = root / f"{episode_id}_step_4.png"
    _save_image(img4, img4_path)
    obs4 = Observation(image_path=str(img4_path))
    steps.append(
        Step(
            t=4.0,
            observation=obs4,
            action=Action(type="type", text=password),
            thought="Type the password.",
        )
    )

    # Step 5: click login button
    cx_btn, cy_btn = _center(elems4.login_button)
    img5, elems5 = _draw_login_screen(username=username, password=password)
    img5_path = root / f"{episode_id}_step_5.png"
    _save_image(img5, img5_path)
    obs5 = Observation(image_path=str(img5_path))
    steps.append(
        Step(
            t=5.0,
            observation=obs5,
            action=Action(type="click", x=cx_btn, y=cy_btn),
            thought="Submit the login form.",
        )
    )

    # Step 6: logged-in screen + DONE
    img6 = _draw_logged_in_screen(username=username)
    img6_path = root / f"{episode_id}_step_6.png"
    _save_image(img6, img6_path)
    obs6 = Observation(image_path=str(img6_path))
    steps.append(
        Step(
            t=6.0,
            observation=obs6,
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


def generate_synthetic_sessions(
    num_sessions: int = 10,
    seed: int | None = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> List[Session]:
    """Generate a list of synthetic Sessions with semantic login episodes.

    Each Session currently contains a single login Episode. Images for all
    steps are written to `output_dir` (default: `synthetic_data/` under the
    current working directory).
    """

    if seed is not None:
        random.seed(seed)

    if output_dir is None:
        # Centralize synthetic assets under a single top-level directory.
        # Callers can still override this, but by default we write to
        # `synthetic/data` instead of scattering `synthetic_*` folders.
        output_root = Path("synthetic") / "data"
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
        episode = _script_login_episode(session_dir, episode_id, username, password)

        session = Session(id=session_id, episodes=[episode], meta={"scenario": "login"})
        sessions.append(session)

    return sessions


