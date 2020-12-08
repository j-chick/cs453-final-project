printf "NOTE that this is only a demo script; src/main.py implements a CLI.\n"

dependencies() {
    echo "TODO dependencies()"
}

run() {
    rm -f temp/output.ply
    python src/main.py \
        --input-path models/panther.ply \
        --n-contractions 200 \
        --simple-pair-selection \
        2>/dev/null \
    && rm -rf src/__pycache__
    vedo temp/output.ply
}

run || dependencies
