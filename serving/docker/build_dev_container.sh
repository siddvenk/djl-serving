#!/usr/bin/env bash

compose_target=$1
if [[ -z "$compose_target" ]]; then
  echo "Must provide docker compose target to build. Usage: ./build_dev_container.sh <target> <djl_version - optional> <djl_serving_version - optional>"
  exit 1
fi

djl_version=${3:-$(awk -F '=' '/djl / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)}
djl_serving_version=${4:-$(awk -F '=' '/serving / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)}

# clean up old artifacts
rm -rf distributions

cd ../../
./gradlew --refresh-dependencies :serving:dockerDeb -Psnapshot
cd serving/docker

docker compose build \
  --build-arg djl_version="$djl_version" \
  --build-arg djl_serving_version="$djl_serving_version" \
  "$compose_target"
