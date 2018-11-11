#!/bin/bash

# compile
sbt package

HOST="c133-01"
USER="binetruy"
TARGET_DIR="~/spark-tp3"
SPARK_HOME="kaggle/spark"
JAR_NAME="spark-tp3_2.11-1.0.jar"

# upload
ssh $USER@$HOST "mkdir -p $TARGET_DIR/"
scp target/scala-2.11/$JAR_NAME $USER@$HOST:$TARGET_DIR

# exec
# ssh $USER@$HOST "./$SPARK_HOME/bin/spark-submit $JAR_NAME"
ssh $USER@$HOST "./$SPARK_HOME/bin/spark-submit $TARGET_DIR/$JAR_NAME"

# repatriate result
# scp $USER@$HOST:$TARGET_DIR/results/*.csv result.csv
