"""
pipeline.artifacts — on-disk artifact path layout helpers.

Artifact tree for a single video
---------------------------------
artifacts/<video_stem>/
  01_track/
    tracks.txt          MOT-format tracking output (0-based frames)
    tracklets.pkl       {id: Tracklet} dict with reused ReID features
  02_refine/
    refined.txt              refined MOT output
    refined_tracklets.pkl    {id: Tracklet} dict (ids match refined.txt)
  03_team/
    refined.txt              refined MOT output with team_id in col 9
    track_attributes.json    {id: {class, team, gk}} per-track aggregation
  profiles/
    <stage_name>.json   per-stage profiling data
    summary.json        aggregated summary across all stages
    summary.md          human-readable Markdown table

Public API
----------
ArtifactPaths  — dataclass-like object returned by ``get_artifact_paths``.
get_artifact_paths(video_path, base_dir=None) -> ArtifactPaths
ensure_dirs(paths: ArtifactPaths) -> None
"""

from __future__ import annotations

import os
from pathlib import Path


class ArtifactPaths:
    """
    Resolved on-disk paths for all pipeline artifacts of one video.

    Attributes
    ----------
    root : Path
        ``<base_dir>/<video_stem>/`` — top-level directory for this video.
    track_dir : Path
        ``root/01_track/``
    tracks_txt : Path
        ``root/01_track/tracks.txt``
    tracklets_pkl : Path
        ``root/01_track/tracklets.pkl``
    refine_dir : Path
        ``root/02_refine/``
    refined_txt : Path
        ``root/02_refine/refined.txt``
    refined_tracklets_pkl : Path
        ``root/02_refine/refined_tracklets.pkl``
    team_dir : Path
        ``root/03_team/``
    team_refined_txt : Path
        ``root/03_team/refined.txt`` (refined MOT output with team_id in col 9)
    track_attributes_json : Path
        ``root/03_team/track_attributes.json`` ({id: {class, team, gk}})
    profiles_dir : Path
        ``root/profiles/``
    """

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.track_dir: Path = root / "01_track"
        self.tracks_txt: Path = self.track_dir / "tracks.txt"
        self.tracklets_pkl: Path = self.track_dir / "tracklets.pkl"
        self.refine_dir: Path = root / "02_refine"
        self.refined_txt: Path = self.refine_dir / "refined.txt"
        self.refined_tracklets_pkl: Path = self.refine_dir / "refined_tracklets.pkl"
        self.team_dir: Path = root / "03_team"
        self.team_refined_txt: Path = self.team_dir / "refined.txt"
        self.track_attributes_json: Path = self.team_dir / "track_attributes.json"
        self.profiles_dir: Path = root / "profiles"

    def profile_json(self, stage_name: str) -> Path:
        """Return the path for a stage's profile JSON file.

        Parameters
        ----------
        stage_name:
            Identifier used for the JSON filename, e.g. ``"01_track"``.

        Returns
        -------
        Path
            ``profiles_dir / "<stage_name>.json"``
        """
        return self.profiles_dir / f"{stage_name}.json"

    def __repr__(self) -> str:  # pragma: no cover
        return f"ArtifactPaths(root={self.root!r})"


def get_artifact_paths(
    video_path: "str | os.PathLike[str]",
    base_dir: "str | os.PathLike[str] | None" = None,
) -> ArtifactPaths:
    """Return an :class:`ArtifactPaths` object for *video_path*.

    Parameters
    ----------
    video_path:
        Path (or filename) of the input video.  Only the *stem* (filename
        without extension) is used.
    base_dir:
        Root directory under which ``<stem>/`` is created.  Defaults to
        ``artifacts/`` relative to the current working directory.

    Returns
    -------
    ArtifactPaths
        Resolved paths — directories are **not** created; call
        :func:`ensure_dirs` to create them.
    """
    video_path = Path(video_path)
    if base_dir is None:
        base_dir = Path("artifacts")
    else:
        base_dir = Path(base_dir)
    root = base_dir / video_path.stem
    return ArtifactPaths(root)


def ensure_dirs(paths: ArtifactPaths) -> None:
    """Create all artifact directories for *paths* (no-op if they exist).

    Parameters
    ----------
    paths:
        An :class:`ArtifactPaths` instance whose directories should exist.
    """
    for directory in (
        paths.track_dir,
        paths.refine_dir,
        paths.team_dir,
        paths.profiles_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
