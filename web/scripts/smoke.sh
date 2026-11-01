#!/usr/bin/env bash
# Headless Chrome smoke test: every disease x step must load without JS errors.
set -u
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PORT=4317
mkdir -p out
npm run build >/dev/null 2>&1 || { echo "build failed"; exit 1; }
npx vite preview --port $PORT --strictPort >out/preview.log 2>&1 &
PREVIEW=$!
trap 'kill $PREVIEW 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -sf "http://localhost:$PORT/" >/dev/null && break; sleep 0.5; done

shot() { # url outfile
  rm -rf out/profile
  perl -e 'alarm 45; exec @ARGV' "$CHROME" --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader \
    --enable-logging=stderr --virtual-time-budget=6000 --window-size=1280,800 \
    --user-data-dir=out/profile --screenshot="$2" "$1" 2>out/stderr.log >/dev/null
  if grep -E 'Uncaught|TypeError|ReferenceError' out/stderr.log; then return 1; fi
  [ -f "$2" ] || { echo "no screenshot $2"; return 1; }
}

# Step count = length of STEP_FACTORIES in src/main.js (override with STEPS=n).
STEPS="${STEPS:-$(node -e 'const s=require("fs").readFileSync("src/main.js","utf8");const m=s.match(/STEP_FACTORIES\s*=\s*\[([^\]]*)\]/);if(!m)process.exit(1);console.log(m[1].split(",").filter((x)=>x.trim()).length)')}"
[ -n "$STEPS" ] && [ "$STEPS" -ge 1 ] 2>/dev/null || { echo "could not derive step count from src/main.js"; exit 1; }
echo "steps: $STEPS"

FAIL=0
for slug in $(node -e 'for(const d of require("./public/data/index.json"))console.log(d.slug)'); do
  for step in $(seq 1 "$STEPS"); do
    shot "http://localhost:$PORT/?disease=$slug&step=$step" "out/$slug-$step.png" || { echo "FAIL $slug step $step"; FAIL=1; }
  done
done
shot "http://localhost:$PORT/?disease=nope" out/nope.png || { echo "FAIL nope"; FAIL=1; }
echo "screenshots: $(ls out/*.png | wc -l)"
exit $FAIL
