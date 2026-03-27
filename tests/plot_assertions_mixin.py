import os
import tempfile
from pathlib import Path
from typing import Any, Callable


class PlotAssertionsMixin:
    def _assert_plot_artifacts_written(self, requested_html_path: Path) -> None:
        if requested_html_path.exists() and requested_html_path.is_file():
            self.assertGreater(requested_html_path.stat().st_size, 0)
            return

        root = requested_html_path.with_suffix("")
        multi_file_candidates = [
            html_file
            for html_file in requested_html_path.parent.glob(f"{root.name}_*.html")
            if html_file.stat().st_size > 0
        ]
        if multi_file_candidates:
            return

        self.fail(
            "No plot artifacts were written. Checked single-file output "
            f"{requested_html_path} and multi-file prefix {root.name}_*.html in "
            f"{requested_html_path.parent}."
        )

    def _assert_plot_written(
        self,
        plot_fn: Callable[..., Any],
        stem: str,
        plot_kwargs: dict[str, Any],
    ) -> None:
        plot_dir = os.environ.get("LCIL_TEST_PLOT_DIR")
        if plot_dir:
            output_dir = Path(plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{stem}.html"
            plot_fn(**plot_kwargs, html_path=str(html_path))
            self._assert_plot_artifacts_written(html_path)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / f"{stem}.html"
            plot_fn(**plot_kwargs, html_path=str(html_path))
            self._assert_plot_artifacts_written(html_path)
