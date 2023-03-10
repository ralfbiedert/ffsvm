#
# Publish all crates.
#

PROJECT_ROOT="$( cd "$(dirname "$0")/.." ; pwd -P )" # this file

function update_readme() {
    cd "$PROJECT_ROOT"/"$1"
    cargo readme --no-license --no-title > README.md
}

update_readme "."