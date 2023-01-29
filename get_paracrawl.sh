#!/bin/bash
for i in {0..1}; do
      wget https://neural.mt/data/paracrawl8-mono/en-$(printf "%03i" $i).gz
  done
