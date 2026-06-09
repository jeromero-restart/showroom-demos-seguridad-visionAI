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

def _cam(id, name, file, description, resolution="HD", fps=25):
    return Camera(
        id=id,
        name=name,
        video_path=_os.path.abspath(_os.path.join(_BASE, file)),
        description=description,
        resolution=resolution,
        fps=fps,
    )


DEMO_CAMERAS = [
    _cam("cam_001", "Playa de Estacionamiento",
         "parking.mp4",
         "Cámara de estacionamiento exterior — detección de vehículos y personas.",
         resolution="720p", fps=30),
    _cam("cam_002", "Intersección Vial",
         "street.mp4",
         "Cámara de cruce urbano — detección de personas y vehículos en tránsito.",
         resolution="720p", fps=25),
    _cam("cam_003", "Acceso al Edificio",
         "building.mp4",
         "Cámara de entrada principal — control de acceso y detección mixta.",
         resolution="720p", fps=30),
    _cam("cam_004", "Peatonal — Personas",
         "personas.mp4",
         "Zona peatonal — detección y conteo de personas.",
         resolution="720p", fps=25),
]

REAL_CAMERAS = []

# Combined list for API endpoints
ALL_CAMERAS = DEMO_CAMERAS + REAL_CAMERAS
