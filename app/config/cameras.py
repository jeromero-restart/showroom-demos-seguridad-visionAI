from dataclasses import dataclass


@dataclass
class Camera:
    id: str
    name: str
    video_path: str
    description: str
    resolution: str  # e.g., "720p", "1080p"
    fps: int


import os as _os
_BASE = _os.environ.get(
    "DEMO_CAMERAS_DIR",
    "/app/demo_cameras"  # default Docker path (./data/demo_cameras mounted here)
)

DEMO_CAMERAS = [
    Camera(
        id="cam_001",
        name="Playa de Estacionamiento",
        video_path=_os.path.abspath(_os.path.join(_BASE, "parking.mp4")),
        description="Cámara de estacionamiento exterior — detección de vehículos y personas (720p, 30fps)",
        resolution="720p",
        fps=30,
    ),
    Camera(
        id="cam_002",
        name="Intersección Vial",
        video_path=_os.path.abspath(_os.path.join(_BASE, "street.mp4")),
        description="Cámara de cruce urbano — detección de personas y animales en tránsito (720p, 25fps)",
        resolution="720p",
        fps=25,
    ),
    Camera(
        id="cam_003",
        name="Acceso al Edificio",
        video_path=_os.path.abspath(_os.path.join(_BASE, "building.mp4")),
        description="Cámara de entrada principal — control de acceso y detección mixta (720p, 30fps)",
        resolution="720p",
        fps=30,
    ),
]

REAL_CAMERAS = []

# Combined list for API endpoints
ALL_CAMERAS = DEMO_CAMERAS + REAL_CAMERAS
