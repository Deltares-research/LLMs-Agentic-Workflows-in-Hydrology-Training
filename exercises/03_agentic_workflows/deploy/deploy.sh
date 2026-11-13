#!/usr/bin/env bash
# Deploy the Exercise 3 course app to Azure App Service: one image, N apps (one per group), one plan.
# Run from the repo root in Git Bash / WSL / Linux, logged in with `az login` (az CLI recent: `az upgrade`).
#
#   bash exercises/03_agentic_workflows/deploy/deploy.sh          # build + deploy
#   bash exercises/03_agentic_workflows/deploy/deploy.sh teardown # delete EVERYTHING (resource group)
#
# Afterwards also regenerate the Foundry key in the portal if the Foundry resource is outside this resource group.
set -euo pipefail

SUBSCRIPTION="${SUBSCRIPTION:-hydrology_llm}"
RG="${RG:-hydro-course-rg}"             # created by this script; teardown deletes it (and only it)
LOCATION="${LOCATION:-westeurope}"
PREFIX="${PREFIX:-hydro-ex2}"            # apps: <prefix>-g1 .. -gN  ->  https://<prefix>-g1.azurewebsites.net
GROUPS_N="${GROUPS_N:-1}"                # apps sharing the plan; Wflow runs queue per app, so raise if runs collide
PLAN_SKU="${PLAN_SKU:-P3V3}"             # 8 vCPU / 32 GB, shared by all apps
ACR="${ACR:-hydrocourse$(echo "$RG" | md5sum | cut -c1-6)}"  # registry name must be globally unique, alphanumeric
IMAGE="hydro-ex2:latest"

# Every az call targets $SUBSCRIPTION, without changing your CLI's default subscription.
az() { command az "$@" --subscription "$SUBSCRIPTION"; }

if [[ "${1:-}" == "teardown" ]]; then
  az group delete -n "$RG" --yes --no-wait
  echo "Deleting resource group $RG (apps, plan, registry, image). Regenerate the Foundry key too."
  exit 0
fi

[[ -f .env ]] || { echo "Run from the repo root; .env not found."; exit 1; }
grep -qE '^COURSE_PASSWORD=.+' .env || { echo "Add COURSE_PASSWORD=... to .env first (otherwise the app is open)."; exit 1; }

az provider register -n Microsoft.Web --wait
az provider register -n Microsoft.ContainerRegistry --wait
az group create -n "$RG" -l "$LOCATION" -o none
az acr show -g "$RG" -n "$ACR" -o none 2>/dev/null \
  || az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -o none

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "Building image in ACR (Julia + Wflow precompile + baseline run: ~20-40 min)..."
  # No log streaming: on Windows the az CLI crashes printing Julia's Unicode progress output. Queue, then poll.
  az acr build -r "$ACR" -t "$IMAGE" -f exercises/03_agentic_workflows/deploy/Dockerfile --timeout 7200 --no-logs . -o none
fi
RUN_ID=$(az acr task list-runs -r "$ACR" --top 1 --query "[0].runId" -o tsv | tr -d '\r')
while :; do
  STATUS=$(az acr task list-runs -r "$ACR" --top 1 --query "[0].status" -o tsv | tr -d '\r')
  case "$STATUS" in
    Succeeded) echo "Build $RUN_ID succeeded."; break ;;
    Failed|Canceled|Error|Timeout)
      echo "Build $RUN_ID: $STATUS. Log: az acr task logs -r $ACR --run-id $RUN_ID --subscription $SUBSCRIPTION"; exit 1 ;;
    *) echo "Build $RUN_ID: $STATUS ..."; sleep 60 ;;
  esac
done

ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)

az appservice plan create -g "$RG" -n "$PREFIX-plan" --is-linux --sku "$PLAN_SKU" -o none

for i in $(seq 1 "$GROUPS_N"); do
  APP="$PREFIX-g$i"
  az webapp create -g "$RG" -p "$PREFIX-plan" -n "$APP" --container-image-name "$ACR.azurecr.io/$IMAGE" -o none
  az webapp config appsettings set -g "$RG" -n "$APP" -o none --settings \
    DOCKER_REGISTRY_SERVER_URL="https://$ACR.azurecr.io" \
    DOCKER_REGISTRY_SERVER_USERNAME="$ACR_USER" \
    DOCKER_REGISTRY_SERVER_PASSWORD="$ACR_PASS" \
    WEBSITES_PORT=8000 \
    WEBSITES_CONTAINER_START_TIME_LIMIT=1800
  az webapp config set -g "$RG" -n "$APP" --web-sockets-enabled true --always-on true \
    --generic-configurations '{"healthCheckPath": "/_stcore/health"}' -o none
  az webapp update -g "$RG" -n "$APP" --https-only true -o none
  echo "Group $i: https://$APP.azurewebsites.net"
done

echo "Done. First page load per app starts the container (a few minutes for the image pull)."
echo "Teardown after the course:  bash $0 teardown"
