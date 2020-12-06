printf "NOTE that this is only a demo script; src/main.py implements a CLI."

dependencies() {
    echo TODO
}

run() {
    python src/main.py \
        --input-path models/icosahedron.ply \
        --n-contractions 7 \
        --simple-pair-selection \
        2>/dev/null \
    && rm -rf src/__pycache__
}

run  || dependencies
