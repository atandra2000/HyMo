"""Project paths: data, checkpoints, logs, eval results.

A :class:`ProjectPaths` is a small frozen dataclass that holds the
canonical locations for all project artifacts. Derived from
:attr:`RunConfig.output_dir` and friends.

The class is deliberately minimal — it does *not* create directories.
The trainer / data pipeline / eval scripts create the directories they
need via :meth:`ensure`. This keeps the class free of side effects and
easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hymo.core.config import RunConfig
from hymo.core.exceptions import PathsError

__all__ = ["ProjectPaths", "PathsError"]

# ``PathsError`` lives in :mod:`hymo.core.exceptions` alongside the other
# HyMo exception classes. It is re-exported here so callers that already
# import it from :mod:`hymo.utils.paths`` (the original home) keep working.


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths.

    All paths are :class:`pathlib.Path`. Directories are not created
    until :meth:`ensure` is called.

    Attributes
    ----------
    root : Path
        The project root (typically the CWD at run time).
    output_dir : Path
        The output directory (e.g. ``checkpoints/pretrain``).
    log_dir : Path
        The log directory (e.g. ``logs``).
    eval_dir : Path
        The eval output directory.
    data_dir : Path
        The data directory (shards, val.bin).
    """

    root: Path
    output_dir: Path
    log_dir: Path
    eval_dir: Path
    data_dir: Path

    @classmethod
    def from_config(cls, config: RunConfig, root: str | Path | None = None) -> ProjectPaths:
        """Build a :class:`ProjectPaths` from a :class:`RunConfig`.

        Parameters
        ----------
        config : RunConfig
        root : str, Path, or None
            Project root. Defaults to the current working directory.
            The configured subpaths (``output_dir`` etc.) are joined
            with ``root`` so the resulting paths are absolute.
        """
        root = Path(root) if root is not None else Path.cwd()
        return cls(
            root=root,
            output_dir=root / config.output_dir,
            log_dir=root / config.log_dir,
            eval_dir=root / config.eval_dir,
            data_dir=root / "data",
        )

    # ---- Common subpaths -------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        """The directory holding the latest + best + per-step checkpoints."""
        return self.output_dir

    @property
    def latest_checkpoint(self) -> Path:
        """The 'latest' DCP checkpoint subdirectory."""
        return self.output_dir / "latest"

    @property
    def best_checkpoint(self) -> Path:
        """The 'best' DCP checkpoint subdirectory (lowest val loss)."""
        return self.output_dir / "best"

    @property
    def metrics_path(self) -> Path:
        """The JSONL metrics log path."""
        return self.log_dir / "metrics.jsonl"

    @property
    def trainer_log_path(self) -> Path:
        """The trainer's stdout/stderr log path."""
        return self.log_dir / "trainer.log"

    @property
    def eval_results_path(self) -> Path:
        """The held-out eval results JSON."""
        return self.eval_dir / "eval_results.json"

    @property
    def data_shards_dir(self) -> Path:
        """The data shards directory."""
        return self.data_dir / "shards"

    @property
    def val_bin_path(self) -> Path:
        """The held-out validation set binary path."""
        return self.data_dir / "tokens" / "val.bin"

    # ---- Side effects ----------------------------------------------------

    def ensure(self) -> None:
        """Create every directory the project needs.

        Idempotent. Raises :class:`PathsError` on permission errors.
        """
        for d in (self.output_dir, self.log_dir, self.eval_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise PathsError(f"Failed to create {d}: {e}") from e

    def __repr__(self) -> str:
        return (
            f"ProjectPaths(root={self.root}, output_dir={self.output_dir}, "
            f"log_dir={self.log_dir}, eval_dir={self.eval_dir})"
        )
