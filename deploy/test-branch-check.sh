#!/usr/bin/env bash
# Exercises check_financedata_branch() against real git repos, by lifting the
# function straight out of deploy.sh so the test can't drift from the script.
set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")" && pwd)/deploy.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

log() { echo "    $*"; }
eval "$(awk '/^check_financedata_branch\(\)/,/^}$/' "$SCRIPT")"

# A bare "origin" with main, plus a clone we can put in various states.
setup() {          # $1 = scenario
    rm -rf "$WORK/origin" "$WORK/fd"
    git init --bare -q -b main "$WORK/origin"
    git clone -q "$WORK/origin" "$WORK/fd" 2>/dev/null
    git -C "$WORK/fd" config user.email t@t; git -C "$WORK/fd" config user.name t
    echo one > "$WORK/fd/a.txt"; git -C "$WORK/fd" add -A
    git -C "$WORK/fd" commit -qm "base"; git -C "$WORK/fd" push -q origin main
    FINANCEDATA_DIR="$WORK/fd"; FINANCEDATA_BRANCH="main"
}

pass=0; fail=0
check() {          # $1 = label, $2 = expected branch, $3... = expected in output
    local label=$1 want=$2; shift 2
    local out; out=$(check_financedata_branch 2>&1)
    local got; got=$(git -C "$WORK/fd" rev-parse --abbrev-ref HEAD)
    local ok=1
    [ "$got" = "$want" ] || { ok=0; echo "  ✗ $label: on '$got', expected '$want'"; }
    for needle in "$@"; do
        grep -q -- "$needle" <<<"$out" || { ok=0; echo "  ✗ $label: no '$needle' in output"; }
    done
    if [ $ok = 1 ]; then echo "  ✓ $label"; pass=$((pass+1)); else
        echo "$out" | sed 's/^/      /'; fail=$((fail+1)); fi
}

echo "1. already on the expected branch — silent"
setup
out=$(check_financedata_branch 2>&1)
if [ -z "$out" ]; then echo "  ✓ no output"; pass=$((pass+1));
else echo "  ✗ expected silence, got: $out"; fail=$((fail+1)); fi

echo "2. on a branch already merged into main, clean — repoints itself"
setup
git -C "$WORK/fd" checkout -qb feature
echo two >> "$WORK/fd/a.txt"; git -C "$WORK/fd" commit -qam "feature work"
git -C "$WORK/fd" push -q origin feature:main          # merged into main upstream
check "repointed" main "repointed FinanceData to main" "nothing was lost"

echo "3. on a branch with commits NOT on main — refuses, keeps the commits"
setup
git -C "$WORK/fd" checkout -qb feature
echo local >> "$WORK/fd/a.txt"; git -C "$WORK/fd" commit -qam "unmerged work"
check "left alone" feature "NOT switching" "avoid discarding"

echo "4. dirty working tree — refuses even when it would otherwise be safe"
setup
git -C "$WORK/fd" checkout -qb feature
echo scratch > "$WORK/fd/uncommitted.txt"
check "dirty" feature "working tree is dirty" "NOT switching"

echo "5. no upstream for the expected branch — gives up quietly, changes nothing"
setup
git -C "$WORK/fd" checkout -qb feature
git -C "$WORK/fd" remote set-url origin "$WORK/nonexistent"
check "unreachable" feature "could not fetch"

echo
echo "$pass passed, $fail failed"
[ $fail = 0 ]
