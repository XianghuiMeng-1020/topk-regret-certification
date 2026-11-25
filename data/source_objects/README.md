# Source objects

This directory is intentionally empty in Git. Populate it with:

```bash
make data
```

The command reads [`../../manifests/SOURCE_OBJECTS.json`](../../manifests/SOURCE_OBJECTS.json), downloads each missing object from its dated public URL, and keeps it only if its byte length and SHA-256 match the pin. Expected size: about 5.9 GB (TLC 0.22 GB, GH Archive 5.69 GB).

Status after a check is written to `../../manifests/SOURCE_OBJECTS_STATUS.json` (local, not tracked).
