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
    # --- Vehículos ---
    _cam("cam_001", "Tránsito Vehicular",
         "autos.mp4",
         "Flujo de vehículos — detección y conteo de autos."),
    _cam("cam_002", "Calle — Autos y Personas",
         "autos_personas_calle.mp4",
         "Vía urbana — detección mixta de vehículos y personas."),
    _cam("cam_003", "Cámara B/N — Autos y Personas",
         "autos_personas_byn.mp4",
         "Cámara monocromática — detección de vehículos y personas."),

    # --- Personas / conteo ---
    _cam("cam_004", "Peatonal — Personas",
         "personas.mp4",
         "Zona peatonal — detección y conteo de personas."),
    _cam("cam_005", "Peatonal 2 — Personas",
         "personas_2.mp4",
         "Zona peatonal (segunda toma) — conteo de personas."),

    # --- Permanencia (DWELL) ---
    _cam("cam_006", "Permanencia — Persona en Zona",
         "permanencia_persona.mp4",
         "Persona permaneciendo en zona — ideal para regla de permanencia (dwell)."),

    # --- Seguridad / vandalismo ---
    _cam("cam_007", "Vandalismo — Ingreso a Zona",
         "vandalismo_ingreso_zona.mp4",
         "Ingreso no autorizado a zona restringida — regla de dirección de ingreso."),
    _cam("cam_008", "Vandalismo — Perímetro",
         "vandalismo_perimetro.mp4",
         "Intrusión en perímetro — detección de presencia en zona crítica."),
    _cam("cam_009", "Vandalismo — Permanencia",
         "vandalismo_permanencia.mp4",
         "Merodeo / permanencia prolongada en zona crítica — regla de permanencia."),
]

REAL_CAMERAS = []

# Combined list for API endpoints
ALL_CAMERAS = DEMO_CAMERAS + REAL_CAMERAS
