#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

dockerfile_managed_packages=(
    causal-conv1d
    cuda-core
    cuda-python
    flash-attn
    mamba-ssm
)

verify_image() {
    local image_dir=$1
    local lockfile="$image_dir/Pipfile.lock"
    local temporary_dir
    local python_version
    local package

    test -f "$lockfile" || {
        echo "$lockfile: missing generated file" >&2
        return 1
    }

    temporary_dir=$(mktemp -d)
    cp "$image_dir/Pipfile" "$lockfile" "$temporary_dir/"

    (
        cd "$temporary_dir"

        PIPENV_NOSPIN=1 PIPENV_IGNORE_VIRTUALENVS=1 pipenv verify &&

        python_version=$(jq -r '._meta.requires.python_version' Pipfile.lock) &&

        jq -r '
            .default | to_entries[] | select(.value.version) |
            ((.key | ascii_downcase | gsub("_"; "-")) + .value.version)
        ' Pipfile.lock | sort -u >locked-constraints &&

        PIPENV_NOSPIN=1 PIPENV_IGNORE_VIRTUALENVS=1 \
            pipenv requirements --from-pipfile --no-lock \
            >pipfile-requirements &&

        uv pip compile \
            --index-strategy unsafe-best-match \
            --python-platform linux \
            "--python-version=$python_version" \
            --constraint locked-constraints \
            --output-file resolved-requirements \
            pipfile-requirements >/dev/null &&

        grep -E '^[[:alnum:]_.-]+==' resolved-requirements |
            sort -u >resolved-packages &&

        for package in "${dockerfile_managed_packages[@]}"; do
            sed -i "/^$package==/d" resolved-packages
        done &&

        comm -23 resolved-packages locked-constraints >missing-packages
    ) || {
        echo "$lockfile: dependencies required by $image_dir/Pipfile cannot be resolved" >&2
        rm -rf "$temporary_dir"
        return 1
    }

    test ! -s "$temporary_dir/missing-packages" || {
        echo "$lockfile: missing dependencies required by $image_dir/Pipfile:" >&2
        sed 's/^/  - /' "$temporary_dir/missing-packages" >&2
        rm -rf "$temporary_dir"
        return 1
    }

    echo "$image_dir: Pipfile dependencies are satisfied by Pipfile.lock"
    rm -rf "$temporary_dir"
}

status=0
image_count=0
while IFS= read -r -d '' pipfile; do
    image_dir=${pipfile%/Pipfile}
    [[ "$image_dir" == *openmpi* ]] && continue
    image_count=$((image_count + 1))
    verify_image "$image_dir" || status=1
done < <(find images/runtime/training -type f -name Pipfile -print0 | sort -z)

if [[ "$image_count" -eq 0 ]]; then
    echo "No non-MPI runtime Pipfile images found; skipping"
fi

exit "$status"
