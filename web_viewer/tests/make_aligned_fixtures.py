"""Synthetic, non-clinical aligned images for local browser verification."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

root = Path(__file__).resolve().parents[1] / "test-results" / "review-inputs" / "aligned"
for folder in ("Histology", "Probability", "Regions", "RGBA", "WrongSize"):
    (root / folder).mkdir(parents=True, exist_ok=True)
y, x = np.mgrid[:256, :256]
rgb = np.stack([220 - x // 3, 100 + y // 3, 150 + x // 4], axis=-1).astype(np.uint8)
for i, roi in enumerate(("ROI_A", "ROI_A_2")):
    Image.fromarray(rgb).save(root / "Histology" / f"{roi}.png")
    labels = np.where(x < 120, 1, 2).astype(np.uint16)
    labels[(x < 20) | (y < 20)] = 0
    tifffile.imwrite(root / "Regions" / f"{roi}.tiff", labels)
    rgba = np.concatenate([rgb.astype(np.uint16) * 257, np.full((256, 256, 1), 32768, dtype=np.uint16)], axis=-1)
    tifffile.imwrite(root / "RGBA" / f"{roi}.tiff", rgba, photometric="rgb", extrasamples="unassalpha")
# Deliberately absent from ROI_A_2 to verify missing-ROI behaviour.
tifffile.imwrite(root / "Probability" / "ROI_A.tiff", (x / 255).astype(np.float32))
tifffile.imwrite(root / "WrongSize" / "ROI_A.tiff", np.zeros((32, 32), dtype=np.float32))
(root / "Regions" / "label_mapping.csv").write_text('value,label,color\n0,Unlabelled,transparent\n1,"Tissue, left",#00ffff\n2,Right region,#ff00ff\n', encoding="utf-8")
print(root)

# Extra IMC channels are deliberately absent from AnnData; DNA3 also collides
# with a channel name in the original Images collection.
for roi, prefix in (("ROI_A", "44_90"), ("ROI_A_2", "55_20")):
    folder = root / "ExtraIMC" / roi
    folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(folder / f"{prefix}_Ir193_DNA3.tiff", (y / 255 * 200).astype(np.float32))
    tifffile.imwrite(folder / "ExtraMarker.tiff", (x / 255 * 100).astype(np.float32))
dictionary_folder = root.parent / "dictionaries"
dictionary_folder.mkdir(exist_ok=True)
(dictionary_folder / "synthetic_annotations_dictionary.csv").write_text('original_num,name,label\n0,*Background*,0\n1,Left tissue,4\n2,Right tissue,3\n', encoding="utf-8")
