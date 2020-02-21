#!/usr/bin/env bash

# Tips:
# - TRAVIS set to true
# - TRAVIS_COMMIT is filled with Build.SourceVersion
# - TRAVIS_BRANCH is filled with one of the following variables:
#   * Build.SourceBranch
#   * System.PullRequest.TargetBranch
# - TRAVIS_PULL_REQUEST is filled with one of the following variables:
#   * Build.SourceVersion
#   * System.PullRequest.PullRequestNumber
# - TRAVIS_EVENT_TYPE is determined at tuntime based on the variable Build.Reason
# - TRAVIS_COMMIT_RANGE is filled with Build.SourceVersion
# - TRAVIS_OS_NAME is assumed already defined
# - TRAVIS_BUILD_DIR got replaced by Build.SourcesDirectory

# Cause the script to exit if a single command fails.
set -e

# TODO: [CI] remove after CI get stable
set -x

# Initialize travis script expected variables.
export PYTHON=$PYTHON_VERSION
echo "Determined PYTHON variable: $PYTHON"

export TRAVIS_COMMIT=$BUILD_SOURCEVERSION
echo "Determined TRAVIS_COMMIT variable: $TRAVIS_COMMIT"

export TRAVIS_BRANCH=$SYSTEM_PULLREQUEST_TARGETBRANCH && [[ -z $TRAVIS_BRANCH ]] && TRAVIS_BRANCH=$BUILD_SOURCEBRANCH
echo "Determined TRAVIS_BRANCH variable: $TRAVIS_BRANCH"

export TRAVIS_PULL_REQUEST=$SYSTEM_PULLREQUEST_PULLREQUESTNUMBER && [[ -z $TRAVIS_PULL_REQUEST ]] && TRAVIS_PULL_REQUEST=$BUILD_SOURCEVERSION
echo "Determined TRAVIS_PULL_REQUEST variable: $TRAVIS_PULL_REQUEST"

export TRAVIS_EVENT_TYPE="push" && [[ ${BUILD_REASON:-X} == "PullRequest" ]] && TRAVIS_EVENT_TYPE="pull_request"
echo "Determined TRAVIS_EVENT_TYPE variable: $TRAVIS_EVENT_TYPE"

export TRAVIS_COMMIT_RANGE=$BUILD_SOURCEVERSION
echo "Determined TRAVIS_COMMIT_RANGE variable: $TRAVIS_COMMIT_RANGE"

echo "Determined TRAVIS_OS_NAME variable: $TRAVIS_OS_NAME"

export TRAVIS_BUILD_DIR=$BUILD_SOURCESDIRECTORY
echo "Determined TRAVIS_BUILD_DIR variable: $TRAVIS_BUILD_DIR"

# export RAY_USE_NEW_GCS="on"
# echo "Determined RAY_USE_NEW_GCS variable: $RAY_USE_NEW_GCS"

# TODO: [CI] remove this step after adding a condition in 
# ci/travis/install-dependencies.sh that check first if 
# node is already installed before install it
echo $(node --version)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
echo "nvm sh downloaded and applied."

# Mac OS bug https://github.com/nvm-sh/nvm/issues/1245#issuecomment-555608208
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    npm config delete prefix
    export NVM_DIR="$HOME/.nvm"
    source $HOME/.nvm/nvm.sh
    nvm use --delete-prefix v6.17.1 --silent
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
fi
