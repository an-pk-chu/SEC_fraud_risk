#!/usr/bin/env bash
# Downloads the Bao, Ke, Li, Yu & Zhang (2020) SEC AAER-labeled fraud dataset
# from its official public repository (github.com/JarFraud/FraudDetection).
# The raw file (~48MB) isn't committed to keep this repo lightweight.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv"
echo "Downloading data_FraudDetection_JAR2020.csv from JarFraud/FraudDetection..."
curl -L -o data_FraudDetection_JAR2020.csv "$URL"
echo "Done. $(wc -l < data_FraudDetection_JAR2020.csv) rows saved to data/data_FraudDetection_JAR2020.csv"
