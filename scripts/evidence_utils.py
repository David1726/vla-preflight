"""Remove local user/workspace paths from shareable validation evidence."""

from pathlib import Path
import re


def sanitized(value, root):
    if isinstance(value, dict):
        return {key: sanitized(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized(item, root) for item in value]
    if not isinstance(value, str):
        return value
    root = Path(root).resolve()
    for path in (root.as_uri(), str(root), root.as_posix()):
        value = re.sub(re.escape(path), "$PROJECT", value, flags=re.IGNORECASE)
    return re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", "$USER_HOME", value, flags=re.IGNORECASE)
