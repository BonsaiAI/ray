#!/usr/bin/env bash

# This needs to be run in the root directory

# Cause the script to exit if a single command fails.
set -e
set -x

bazel build "//:gcs_client_test" "//:asio_test" "//:libray_redis_module.so" "//:redis-server"

# Start Redis.
if [[ "${RAY_USE_NEW_GCS}" = "on" ]]; then
    ./src/credis/redis/src/redis-server \
        --loglevel warning \
        --loadmodule ./src/credis/build/src/libmember.so \
        --loadmodule ./src/ray/gcs/redis_module/libray_redis_module.so \
        --port 6379 &
else
    bazel run //:redis-cli -- -p 6379 shutdown || true
    sleep 1s
    bazel run //:redis-server -- --loglevel warning --loadmodule ./bazel-bin/libray_redis_module.so --port 6379 &
fi
sleep 1s

./bazel-bin/gcs_client_test
./bazel-bin/asio_test

sleep 1s
# ./bazel-genfiles/redis-cli -p 6379 shutdown
bazel run //:redis-cli -- -p 6379 shutdown
sleep 1s
