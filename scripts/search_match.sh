#!/usr/bin/env bash
# Grep a set of files for a string and record the names of the matching files in a doc.
#
# Replaces the throwaway one-liner
#     for f in `search 38607192`; do echo $f; cat $f | grep 084a5335; done
# with something that keeps a record of which files actually matched.
#
# Usage:
#   scripts/search_match.sh <file-pattern> <string> [-o doc] [-d dir] [-q]
#   search 38607192 | scripts/search_match.sh - 084a5335        # feed your own file list
#
# Examples:
#   # every slurm outfile of array job 38607192 that mentions the override hash
#   scripts/search_match.sh 38607192 084a5335
#
#   # same, but append to a doc of your choosing and stay quiet on stdout
#   scripts/search_match.sh 38607192 084a5335 -o notes/hash_hits.md -q
#
# <file-pattern> is matched against file names as *<file-pattern>* under <dir>
# (default: scripts/slurm_outfiles). Pass "-" to read the file list from stdin
# instead, so an existing `search` shell function can still do the finding.
#
# The doc (default: scripts/slurm_outfiles/matches_<pattern>_<string>.md) gets one
# section per run, listing each matching file and its number of matching lines.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_DIR="${SEARCH_DIR:-$PROJECT_DIR/scripts/slurm_outfiles}"

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -lt 2 ]] && usage 1

pattern="$1"
needle="$2"
shift 2

doc=''
quiet=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out)  doc="$2"; shift 2 ;;
        -d|--dir)  SEARCH_DIR="$2"; shift 2 ;;
        -q|--quiet) quiet=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

# Where to record the hits.
if [[ -z "$doc" ]]; then
    slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }
    doc="$SEARCH_DIR/matches_$(slug "$pattern")_$(slug "$needle").md"
fi
mkdir -p "$(dirname "$doc")"

# Collect the candidate files: stdin when the pattern is "-", otherwise a name search.
files=()
if [[ "$pattern" == '-' ]]; then
    while IFS= read -r f; do
        [[ -n "$f" ]] && files+=("$f")
    done
else
    if [[ ! -d "$SEARCH_DIR" ]]; then
        echo "search dir not found: $SEARCH_DIR (set SEARCH_DIR or pass -d)" >&2
        exit 1
    fi
    doc_abs="$(cd "$(dirname "$doc")" && pwd)/$(basename "$doc")"
    while IFS= read -r f; do
        # never search the doc we are about to append to
        [[ "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" == "$doc_abs" ]] && continue
        files+=("$f")
    done < <(find "$SEARCH_DIR" -type f -name "*$pattern*" | sort)
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "no files matched pattern '$pattern'" >&2
    exit 1
fi

# Grep each file; a file with at least one matching line gets written to the doc.
hits=()
counts=()
for f in "${files[@]}"; do
    if [[ ! -r "$f" ]]; then
        echo "skipping unreadable file: $f" >&2
        continue
    fi
    n=$(grep -c -- "$needle" "$f")
    if [[ "$n" -gt 0 ]]; then
        hits+=("$f")
        counts+=("$n")
        [[ $quiet -eq 0 ]] && printf '%s\t%s match(es)\n' "$f" "$n"
    fi
done

if [[ "$pattern" == '-' ]]; then
    scope='files from stdin'
else
    scope="\`*$pattern*\` under \`$SEARCH_DIR\`"
fi

{
    printf '\n## %s in %s — %s\n\n' "$needle" "$scope" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'searched %d file(s)\n\n' "${#files[@]}"
    if [[ ${#hits[@]} -eq 0 ]]; then
        printf 'no matches\n'
    else
        for i in "${!hits[@]}"; do
            printf -- '- `%s` (%s match(es))\n' "${hits[$i]}" "${counts[$i]}"
        done
    fi
} >> "$doc"

if [[ $quiet -eq 0 ]]; then
    printf '%d/%d file(s) matched; recorded in %s\n' "${#hits[@]}" "${#files[@]}" "$doc"
fi

[[ ${#hits[@]} -gt 0 ]]
