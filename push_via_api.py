#!/usr/bin/env python3
"""Push a local commit to GitHub through the REST API.

`git push` cannot reach github.com from this machine (the local proxy
answers 502 to the CONNECT tunnel), but `gh api` works.  This rebuilds the
local HEAD commit server-side: blobs -> tree -> commit -> ref update.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys

REPO = "Wppgenius111/Fudan_iCourse_Subscriber"
BRANCH = "main"


def gh(path: str, payload: dict | None = None, method: str | None = None) -> dict:
    cmd = ["gh", "api", path]
    if method:
        cmd += ["-X", method]
    if payload is not None:
        cmd += ["--input", "-"]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode() if payload is not None else None,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode() + "\n")
        raise SystemExit(f"gh api failed: {path}")
    out = proc.stdout.decode().strip()
    return json.loads(out) if out else {}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, check=True,
        cwd="/tmp/fics-probe",
    ).stdout.decode()


def main() -> None:
    parent = gh(f"/repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    base_tree = gh(f"/repos/{REPO}/git/commits/{parent}")["tree"]["sha"]
    print(f"parent    = {parent}")
    print(f"base_tree = {base_tree}")

    changed = git("diff", "--name-only", "HEAD~1", "HEAD").split()
    print(f"changed   = {changed}")

    entries = []
    for path in changed:
        with open(f"/tmp/fics-probe/{path}", "rb") as f:
            content = base64.b64encode(f.read()).decode()
        blob = gh(f"/repos/{REPO}/git/blobs",
                  {"content": content, "encoding": "base64"})
        print(f"  blob {path} -> {blob['sha']}")
        entries.append({"path": path, "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})

    tree = gh(f"/repos/{REPO}/git/trees",
              {"base_tree": base_tree, "tree": entries})
    print(f"tree      = {tree['sha']}")

    message = git("log", "-1", "--pretty=%B")
    commit = gh(f"/repos/{REPO}/git/commits",
                {"message": message, "tree": tree["sha"], "parents": [parent]})
    print(f"commit    = {commit['sha']}")

    gh(f"/repos/{REPO}/git/refs/heads/{BRANCH}",
       {"sha": commit["sha"], "force": False}, method="PATCH")
    print(f"ref {BRANCH} -> {commit['sha']}  ✓")


if __name__ == "__main__":
    main()
