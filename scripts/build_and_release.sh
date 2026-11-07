#!/usr/bin/env bash
# Helper script to build wheels and trigger a GitHub Release
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
    echo "Usage: ./scripts/build_and_release.sh <vX.Y.Z>"
    echo "Example: ./scripts/build_and_release.sh v2.0.0"
    exit 1
fi

echo "=== Building Nava Packages ==="
cd "$ROOT"

# Clean old dists
rm -rf dist nava_sdk/dist

echo "1. Building nava_sdk..."
(cd nava_sdk && python3 -m build)

echo "2. Building root package..."
python3 -m build

echo "3. Verifying package distributions with twine..."
python3 -m twine check dist/* nava_sdk/dist/*

echo "=== Build Successful! ==="
echo ""
echo "To publish a release on GitHub:"
echo "  git add ."
echo "  git commit -m 'Release $VERSION'"
echo "  git tag $VERSION"
echo "  git push origin main --tags"
echo ""
echo "GitHub Actions will automatically:"
echo "  - Run all unit tests"
echo "  - Build wheels and source distributions"
echo "  - Generate the GitHub Release with attached assets"
echo "  - Publish nava-fa-speech to PyPI"

