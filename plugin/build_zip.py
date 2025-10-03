
import os, zipfile, pathlib

def build_zip():
    here = pathlib.Path(__file__).parent
    plugin_dir = here / "roadxref_plugin"
    out = here.parent / "dist"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "roadxref_qgis_plugin.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in plugin_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(here))
    print(f"ZIP gerado: {zip_path}")

if __name__ == "__main__":
    build_zip()
