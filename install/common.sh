#!/bin/bash

# Helper: append line to file if not already present
append_if_missing() {
    local line="$1"
    local file="$2"

    grep -Fxq "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# Helper: remove any line matching a pattern from a file
remove_matching() {
    local pattern="$1"
    local file="$2"

    sed -i "/$pattern/d" "$file" 2>/dev/null
}