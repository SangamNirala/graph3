#!/bin/bash
export NODE_OPTIONS="--max-old-space-size=4096 --max-semi-space-size=1024"
export GENERATE_SOURCEMAP=false
export DISABLE_ESLINT_PLUGIN=true
yarn start