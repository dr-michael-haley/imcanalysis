"""Capture real NapariSBT widgets using disposable, synthetic demonstration data.

Run from the repository root with the GUI environment's Python. No user project,
Napari preferences, or recent-workspace settings are read or modified.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUTPUT = REPO / "docs/source/_static/napari_sbt"


def main():
    import faulthandler
    faulthandler.dump_traceback_later(90, exit=True)
    print("Preparing disposable documentation capture", flush=True)
    with tempfile.TemporaryDirectory(prefix="naparisbt-guide-") as directory:
        project = Path(directory)
        os.environ["NAPARI_CONFIG"] = str(project / "napari-settings.yaml")
        # Widget captures do not need JIT compilation or writes to the installed
        # environment's Numba cache (which can be slow/read-only on workstations).
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = str(project / "numba-cache")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np
        import pandas as pd
        import anndata as ad
        import tifffile
        from qtpy.QtCore import QSettings, QTimer
        from qtpy.QtGui import QFont, QFontDatabase
        from qtpy.QtWidgets import QApplication, QMessageBox
        import napari
        from SpatialBiologyToolkit.napari_sbt import launch
        from SpatialBiologyToolkit.napari_sbt import onboarding

        print("GUI imports ready", flush=True)
        qt_app = QApplication.instance() or QApplication([])
        # The offscreen Windows Qt plugin does not enumerate system fonts.
        if sys.platform == "win32":
            font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for filename in ("segoeui.ttf", "segoeuib.ttf", "seguisym.ttf", "seguiemj.ttf"):
                font_path = font_dir / filename
                if font_path.is_file():
                    QFontDatabase.addApplicationFont(str(font_path))
            qt_app.setFont(QFont("Segoe UI", 10))

        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(project / "settings"))
        QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, str(project / "settings"))
        # QSettings(organisation, application) can still use the Windows registry
        # despite setDefaultFormat. Isolate the onboarding factory explicitly.
        onboarding.QSettings = lambda *args, **kwargs: QSettings(
            str(project / "workspace-settings.ini"), QSettings.IniFormat
        )
        (project / "masks").mkdir()
        markers = ["CD3", "CD20", "CD68"]
        records, values = [], []
        yy, xx = np.indices((192, 192))
        for roi in ("Demo_A", "Demo_B"):
            masks = np.zeros((192, 192), dtype=np.uint16)
            images = np.zeros((3, 192, 192), dtype=np.float32)
            for number in range(1, 37):
                row, col = divmod(number - 1, 6)
                cell = (yy - (18 + 30 * row)) ** 2 + (xx - (18 + 30 * col)) ** 2 < 10 ** 2
                masks[cell] = number
                group = (number - 1) % 3
                signal = np.array([.12, .12, .12])
                signal[group] = .9
                for index in range(3):
                    images[index, cell] = signal[index]
                records.append({"ROI": roi, "ObjectNumber": number,
                                "Population": ("T cells", "B cells", "Myeloid")[group],
                                "Sample": "Demonstration"})
                values.append(signal)
            tifffile.imwrite(project / "masks" / f"{roi}.tiff", masks)
            folder = project / "images" / roi
            folder.mkdir(parents=True)
            for marker, pixels in zip(markers, images, strict=True):
                tifffile.imwrite(folder / f"{marker}.tiff", pixels)
        obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(len(records))])
        obs["Population"] = pd.Categorical(obs["Population"], categories=["T cells", "B cells", "Myeloid"])
        data = ad.AnnData(np.asarray(values, dtype=np.float32), obs=obs, var=pd.DataFrame(index=markers))
        data.uns["Population_colors"] = ["#22c55e", "#60a5fa", "#fb923c"]
        data.write_h5ad(project / "demo.h5ad")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        print("Launching demonstration workspace", flush=True)
        viewer, controller, dock = launch(
            viewer=napari.Viewer(show=False), project_root=project,
            anndata_path=project / "demo.h5ad", masks_folder=project / "masks",
            images_folders=[project / "images"], welcome=True,
        )
        # Render the real dock, not the OpenGL canvas (headless machines have no GL).
        dock.setFloating(True)
        dock.resize(1120, 1000)
        dock.show()

        def capture(filename, width=1120, height=1000):
            dock.resize(width, height)
            QApplication.processEvents()
            if not dock.grab().save(str(OUTPUT / filename)):
                raise RuntimeError(f"Could not save {filename}")
            print(f"Captured {filename}", flush=True)

        controller.tabs.setCurrentIndex(0)
        capture("home.png", 900, 690)
        controller.workspace_flow.show_step(0)
        controller.tabs.setCurrentIndex(controller._workflow_tab_indices["setup"])
        controller.project_edit.setText(str(project))
        controller.workspace_flow.files_toggle.setChecked(True)
        capture("workspace.png", 1040, 920)
        controller.workflow_combo.setCurrentIndex(controller.workflow_combo.findData("population_qc"))
        controller.name_edit.setText("Demonstration review")
        controller.experiment_edit.setText(str(project / "napari_sbt" / "demonstration"))
        # Only the disposable example's creation prompt is accepted.
        original_question = QMessageBox.question
        QMessageBox.question = lambda *args, **kwargs: QMessageBox.Yes
        try:
            controller.create_experiment()
        finally:
            QMessageBox.question = original_question
        controller.load_roi("Demo_A")
        controller.suggest_population_qc_markers()
        controller.load_population_qc_view()
        controller.tabs.setCurrentIndex(controller._workflow_tab_indices["explore"])
        capture("explore.png")
        controller.tabs.setCurrentIndex(controller._workflow_tab_indices["population_qc"])
        capture("population-qc.png", 1120, 1300)

        def capture_help():
            dialog = QApplication.activeModalWidget()
            if dialog is None:
                raise RuntimeError("Help popup did not open")
            dialog.grab().save(str(OUTPUT / "help.png"))
            dialog.accept()

        QTimer.singleShot(200, capture_help)
        controller.show_help("population_qc", "RGB verification recipe", section="RGB verification recipe")
        print("Captured help.png", flush=True)
        dock.close()
        viewer.close()
        QApplication.processEvents()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
