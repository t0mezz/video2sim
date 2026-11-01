#!/usr/bin/env bash
# Fetch the Panda model of MuJoCo Menagerie for P5.4 into data/, which git ignores (Apache-2.0).
set -euo pipefail
commit=b0655057a9df2abe39ea7fc989914ba6f248301c
target=data/third_party/mujoco_menagerie
if [ ! -d "$target/.git" ]; then
  git clone -q --filter=blob:none --no-checkout --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$target"
fi
git -C "$target" sparse-checkout set franka_emika_panda
git -C "$target" checkout -q "$commit"
