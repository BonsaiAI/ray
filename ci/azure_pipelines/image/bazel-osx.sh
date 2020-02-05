#!/bin/bash
################################################################################
##  File:  bazel.sh
##  Desc:  Installs Bazel
################################################################################

# Source the helpers for use with the script
source $HELPER_SCRIPTS/document.sh

echo "Add Bazel distribution URI as a package source"
brew tap bazelbuild/tap
echo "Install and update Bazel"
brew install bazelbuild/tap/bazel

DocumentInstalledItem "$(bazel --version)"