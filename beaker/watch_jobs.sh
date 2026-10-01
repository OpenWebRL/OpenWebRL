#!/usr/bin/env bash
# Periodic health check for OpenWebRL Beaker jobs (driven every 15 min).
#
# Prints one line per job in outputs/rl_full/_launched/JOBS (one "<id> <label>"
# per line, '#' comments allowed), plus a line for anything that needs attention.
# Emits nothing routine unless a state changed or a job is unhealthy, so it can be
# driven from a Monitor without flooding the conversation.
#
# Detects three failure shapes, not just a non-zero exit:
#   * terminal states (failed / canceled / preempted)
#   * a crash signature in the logs while still "running"
#   * a silent hang: log has not grown since the previous check while running
#     (the B300 runs hung for 1-2h after cuda-graph capture with no output and no
#     non-zero exit, so exit codes alone miss it). STALL_CHECKS sets how many
#     frozen checks count as a hang.
set -u
STATE_DIR="${STATE_DIR:-/tmp/owrl_watch}"
# Number of consecutive checks with a frozen log before calling it a hang.
STALL_CHECKS="${STALL_CHECKS:-8}"
JOBS_FILE="${JOBS_FILE:-/weka/oe-training-default/new_peters/OpenWebRL/outputs/rl_full/_launched/JOBS}"
mkdir -p "$STATE_DIR"

[ -f "$JOBS_FILE" ] || { echo "[watch] no jobs file at $JOBS_FILE"; exit 0; }

while read -r id label; do
  case "$id" in ''|\#*) continue;; esac
  # Normalise to a single token: beaker prints e.g. "1 pending", and a state with
  # a space breaks the "$st $n $stalls" state file (read misaligns the fields, so
  # change detection never fires and every check re-prints).
  # A retried experiment reports both states at once, e.g. "1 preempted by
  # system, 1 pending". Report the live one -- a preempted job that beaker has
  # already re-queued is not a crash to alert on.
  raw=$(beaker experiment get "$id" 2>/dev/null | tail -1)
  st=$(printf '%s' "$raw" | grep -oE "[0-9]+ (running|starting|pending)" | tail -1 | tr ' ' '_')
  [ -z "$st" ] && st=$(printf '%s' "$raw" \
       | grep -oE "[0-9]+ (succeeded|failed|preempted|canceled)" | tail -1 | tr ' ' '_')
  [ -z "$st" ] && st="unknown"
  LG=$(beaker experiment logs "$id" 2>/dev/null)
  n=$(printf '%s' "$LG" | wc -l)
  # Fatal signatures only. A bare "Traceback" is not one: generate_turn_sample logs
  # a per-task traceback at WARNING level for every retried browser task, so
  # matching it alerted on a perfectly healthy rollout.
  crash=$(printf '%s' "$LG" | grep -aoE "no kernel image|Mismatched number of arguments|PTXAS error|Capture cuda graph failed|OutOfMemoryError|torch.OutOfMemoryError|NCCL.*(error|timeout)|ActorDiedError|RayTaskError|ray.exceptions|The actor died|ImportError: cannot import" | sort -u | tr '\n' ' ')
  prog=$(printf '%s' "$LG" | grep -aoE "rollout_id=[0-9]+" | tail -1)
  rew=$(printf '%s' "$LG" | grep -aoE "rollout/raw_reward_mean': ?[0-9.]+" | tail -1)
  ck=$(printf '%s' "$LG" | grep -aoE "iter_0*[1-9][0-9]*" | tail -1)

  f="$STATE_DIR/$id"
  prev_st=""; prev_n=0; stalls=0
  [ -f "$f" ] && read -r prev_st prev_n stalls < "$f"
  : "${stalls:=0}"

  alert=""
  case "$st" in
    *failed*|*canceled*|*preempted*) alert="TERMINAL:$st";;
  esac
  if [ -z "$alert" ] && [ -n "$crash" ]; then alert="CRASH_SIG:$crash"; fi
  case "$st" in
    *running*)
      if [ "$n" -eq "$prev_n" ]; then
        stalls=$((stalls+1))
        [ "$stalls" -ge "$STALL_CHECKS" ] && [ -z "$alert" ] && alert="SILENT x$stalls checks (log frozen at $n lines)"
      else stalls=0; fi;;
    *) stalls=0;;
  esac

  echo "$st $n $stalls" > "$f"

  # Report a terminal state once, not every hour: once it is recorded in the state
  # file the alert would otherwise repeat forever for a job already dealt with.
  if [ "$st" = "$prev_st" ]; then
    case "$st" in *succeeded*|*failed*|*canceled*|*preempted*) alert="";; esac
  fi

  if [ -n "$alert" ] || [ "$st" != "$prev_st" ]; then
    echo "[watch] ${label:-$id} = $st ${prog:+$prog} ${ck:+$ck} ${rew:+$rew} ${alert:+<< $alert}"
  fi
done < "$JOBS_FILE"
