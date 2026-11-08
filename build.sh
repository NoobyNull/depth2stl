#!/bin/bash
# Build a standalone compiled binary with Nuitka, then install it for this user.
#   ./build.sh          build + install
#   ./build.sh build    build only  -> build/depth2stl
#   ./build.sh install  install the existing build/depth2stl
# Builds from a clean venv holding only requirements*.txt (numpy, pillow, nuitka).
set -euo pipefail
cd "$(dirname "$0")"
VERSION=1.1.0

[[ "${1:-}" == install ]] || {
[[ -x .venv/bin/nuitka ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt -r requirements-build.txt; }
.venv/bin/nuitka --onefile --enable-plugin=tk-inter --jobs="$(nproc)" \
  --include-package=OpenGL --include-package=pyopengltk --noinclude-pytest-mode=nofollow --noinclude-unittest-mode=nofollow --noinclude-setuptools-mode=nofollow \
  --output-dir=build --output-filename=depth2stl --remove-output \
  --product-version="$VERSION" --onefile-tempdir-spec='{CACHE_DIR}/depth2stl/{VERSION}' \
  --assume-yes-for-downloads depth2stl.py
}
[[ "${1:-}" == build ]] && exit 0

BIN=~/.local/bin; APPS=~/.local/share/applications; ICONS=~/.local/share/icons/hicolor/256x256/apps
mkdir -p "$BIN" "$APPS" "$ICONS"
install -m755 build/depth2stl "$BIN/depth2stl"
install -m644 icon.png "$ICONS/depth2stl.png"
# GPU depth estimator: runs in a torch venv (ROCm/CUDA), not bundled in the binary
TORCH_PY=${TORCH_PY:-$HOME/Projects/timesfm-mnq/.venv/bin/python}
LIB=~/.local/lib/depth2stl; mkdir -p "$LIB"; install -m644 depth_estimate.py depth2stl.py "$LIB/"
printf '#!/bin/bash\nexec "%s" "%s/depth_estimate.py" "$@"\n' "$TORCH_PY" "$LIB" > "$BIN/depth2stl-estimate"; chmod 755 "$BIN/depth2stl-estimate"
sed "s|@BIN@|$BIN|" depth2stl.desktop > "$APPS/depth2stl.desktop"
D=$(xdg-user-dir DESKTOP 2>/dev/null || echo ~/Desktop); [[ -d "$D" ]] && install -m755 "$APPS/depth2stl.desktop" "$D/depth2stl.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q ~/.local/share/icons/hicolor 2>/dev/null || true
echo "installed: $BIN/depth2stl  ($APPS/depth2stl.desktop)"
