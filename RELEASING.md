# Releasing QBITEL Bridge

This document describes the release process for QBITEL Bridge.

## Versioning

QBITEL Bridge follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes or incompatible protocol modifications
- **MINOR**: New features, new protocol adapters, backward-compatible additions
- **PATCH**: Bug fixes, security patches, documentation updates

## Release Checklist

### Pre-Release

1. **Ensure all CI checks pass** on the `main` branch
   ```bash
   make test
   make lint
   make scan
   ```

2. **Update version numbers** in:
   - `pyproject.toml` → `version`
   - `helm/qbitel-bridge/Chart.yaml` → `version` and `appVersion`
   - `helm/qbitel-bridge/values.yaml` → all image `tag` fields
   - `ai_engine/__init__.py` → `__version__`

3. **Update CHANGELOG.md** with release notes:
   - New features
   - Bug fixes
   - Breaking changes
   - Security fixes
   - Dependency updates

4. **Run full test suite** including integration tests:
   ```bash
   make test
   make test-python
   cd rust/dataplane && cargo test
   ```

5. **Verify Docker builds**:
   ```bash
   make docker-images
   ```

### Creating the Release

1. **Create a release branch** (for major/minor releases):
   ```bash
   git checkout -b release/v1.x.0
   ```

2. **Tag the release**:
   ```bash
   git tag -a v1.x.0 -m "Release v1.x.0: <brief description>"
   git push origin v1.x.0
   ```

3. **Create GitHub Release**:
   ```bash
   gh release create v1.x.0 \
     --title "QBITEL Bridge v1.x.0" \
     --notes-file CHANGELOG.md \
     --latest
   ```

### Post-Release

1. **Verify published artifacts**:
   - Container images pushed to GHCR
   - Helm chart updated
   - PyPI package published (if applicable)

2. **Update documentation** if needed

3. **Announce the release** through appropriate channels

## Hotfix Process

For critical security fixes:

1. Branch from the release tag: `git checkout -b hotfix/v1.x.1 v1.x.0`
2. Apply the fix with tests
3. Tag and release: `v1.x.1`
4. Cherry-pick the fix back to `main`

## Container Image Tags

- `latest` — latest stable release (avoid in production)
- `v1.x.0` — specific version (recommended for production)
- `main` — latest build from main branch (unstable)
- `sha-<commit>` — specific commit build
