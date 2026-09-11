#!/bin/bash

TARGET_DIR="home/sbrooke/op-vp/build"
TARGET_NAME="opvp"
FULL_PATH="$TARGET_DIR/$TARGET_NAME"

echo "Starting '$TARGET_NAME' watcher (for $TARGET_DIR)..."

while true; do
    # Check if the symlink already exists
    if [ -L "$FULL_PATH" ]; then
        echo "Found '$TARGET_NAME' (symlink). Executing..."
        "$FULL_PATH"
        echo "'$TARGET_NAME' stopped/killed. Resetting watcher..."
        continue
    fi

    echo "Waiting for '$TARGET_NAME' to be created..."

    # If it doesn't exist yet, block until a CREATE event happens
    while read -r filename; do
        if [ "$filename" = "$TARGET_NAME" ]; then
            # Wait for executable to be fully created
            sleep 0.1
            
            if [ -L "$FULL_PATH" ]; then
                echo "New '$TARGET_NAME' (simlink) created"
                break
            fi
        fi
        echo "'$TARGET_NAME' not a simlink"
    done < <(inotifywait -e create,moved_to --format '%f' "$TARGET_DIR" 2>/dev/null)

    # Execute the symlink
    if [ -L "$FULL_PATH" ]; then
        echo "Found '$TARGET_NAME' (symlink). Executing..."
        "$FULL_PATH"
        echo "'$TARGET_NAME' stopped/killed. Resetting watcher..."
    fi
done
