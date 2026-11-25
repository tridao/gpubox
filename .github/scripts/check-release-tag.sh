#!/usr/bin/env bash

if [[ "$1" != "refs/tags/v$2" ]]; then
    printf 'Publish from tag v%s, not %s.\n' "$2" "$1" >&2
    exit 1
fi
