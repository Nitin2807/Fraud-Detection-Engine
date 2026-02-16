# Session Handoff - Fraud Detection Engine

Date: 2026-02-15
Project path: `D:\Fraud Detection Engine`

## What we diagnosed
- Kafka producer (`src/ingestion/stream_generator.py`) was working.
- Consumer startup failed for multiple environment reasons, not only code.
- Main runtime blockers were:
  - Spark + Windows environment issues (`HADOOP_HOME`/`winutils.exe`).
  - Java mismatch (`JAVA_HOME` was JDK 25 at one point; Spark 3.5 expects Java 17).
  - Wrong interpreter being used in runs (system Python instead of conda env).
  - Later: PyTorch import failure (`WinError 1114`, `c10.dll`) in `FDE_env`.

## Code changes made

### 1) `src/processing/spark_consumer.py`
- Replaced broken scalar `@pandas_udf` scoring path with `mapInPandas` batch scoring.
- Added Kafka reader stability options:
  - `startingOffsets=latest`
  - `failOnDataLoss=false`
- Made model/scaler/columns artifact paths robust via project-root resolution (`pathlib.Path`).
- Added runtime checks:
  - Parse Java version and fail fast if Java > 17.
  - Clear malformed `HADOOP_HOME` values containing `%...%` placeholders.
- Spark package loading now conditional:
  - Always loads Kafka package.
  - Loads Mongo connector only when `OUTPUT_MODE == "mongodb"`.

### 2) `requirements.txt` (Fraud project)
- Replaced `kafka` with `kafka-python`.
- Added `pyspark==3.5.0`.
- Added `pyarrow>=12,<16` (required by `mapInPandas`).

## Environment findings
- `FDE_env` now has Python 3.11 and proper env interpreter path:
  - `C:\Users\nitin\anaconda3\envs\FDE_env\python.exe`
- But `torch` import failed in `FDE_env` with DLL init error (`c10.dll`, WinError 1114), likely wheel/runtime conflict.

## Next-run checklist (copy/paste)

Run in **Anaconda Prompt (cmd)**, not PowerShell (PowerShell activation was unreliable):

```cmd
conda activate FDE_env
python -c "import sys; print(sys.executable)"
```

Expected: path under `...\anaconda3\envs\FDE_env\python.exe`.

### Fix PyTorch first
```cmd
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
python -c "import torch; print(torch.__version__)"
```
If still failing:
```cmd
conda install -y vs2015_runtime
python -c "import torch; print(torch.__version__)"
```

### Spark Java/Hadoop vars (same cmd session)
```cmd
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jre-17.0.8.101-hotspot
set PATH=%JAVA_HOME%\bin;%PATH%

set HADOOP_HOME=C:\hadoop
set hadoop.home.dir=C:\hadoop
set PATH=%HADOOP_HOME%\bin;%PATH%
```

Note: `C:\hadoop\bin\winutils.exe` must exist.

### Install project deps
```cmd
pip install -r "D:\Fraud Detection Engine\requirements.txt"
```

### Run order
Terminal 1:
```cmd
python "D:\Fraud Detection Engine\src\processing\spark_consumer.py"
```
Terminal 2:
```cmd
python "D:\Fraud Detection Engine\src\ingestion\stream_generator.py"
```

## If still broken next session
Share these outputs first:
```cmd
python -c "import sys; print(sys.executable)"
java -version
echo %JAVA_HOME%
echo %HADOOP_HOME%
python -c "import torch; print(torch.__version__)"
```

And then paste full `spark_consumer.py` traceback.

## Suggested future cleanup
- Pin `torch==2.5.1` in `requirements.txt` to avoid unstable wheel upgrades.
- Optionally move consumer execution into Docker later to avoid Windows Spark/Hadoop friction.

## Update (2026-02-16)

### Current status
- `FDE_env` is now healthy and usable.
- Java in active terminal is correct:
  - `JAVA_HOME=C:\Program Files\Eclipse Adoptium\jre-17.0.8.101-hotspot`
  - `where java` shows Java 17 first.
- Hadoop helper is present:
  - `where winutils.exe` -> `C:\hadoop\bin\winutils.exe`

### Root cause recap (final)
- `HADOOP_HOME` had previously been persisted incorrectly as `%HADOOP_HOME%\\bin`.
- VS Code terminal and standalone cmd were not always inheriting the same environment snapshot.
- Some earlier runs used wrong Python interpreter and wrong Java (JDK 25), causing Spark startup failures.

### Verified-good checks before running consumer
Run these in the same terminal where you run Spark:
```cmd
conda activate FDE_env
echo %JAVA_HOME%
where java
where winutils.exe
python -c "import os; print(os.environ.get('JAVA_HOME')); print(os.environ.get('HADOOP_HOME')); import torch,pyspark,pyarrow; print(torch.__version__, pyspark.__version__, pyarrow.__version__)"
```

### Run sequence
Terminal 1:
```cmd
python "D:\Fraud Detection Engine\src\processing\spark_consumer.py"
```
Terminal 2:
```cmd
python "D:\Fraud Detection Engine\src\ingestion\stream_generator.py"
```

### If Spark fails again
Paste output of:
```cmd
echo %JAVA_HOME%
echo %HADOOP_HOME%
where java
where winutils.exe
python -c "import os; print(os.environ.get('JAVA_HOME')); print(os.environ.get('HADOOP_HOME'))"
```

## Update (2026-02-16 - Working State Achieved)

### Status
- End-to-end streaming now works.
- `spark_consumer.py` successfully reads from Kafka topic and scores messages.
- `stream_generator.py` producer is publishing and consumer is receiving.

### Final blockers resolved
1. Windows Spark runtime and env mismatch issues
- Fixed Java mismatch (using Java 17 for Spark 3.5).
- Fixed broken historical `HADOOP_HOME` values.
- Added/verified `winutils.exe` setup.
- Enforced local driver binding in Spark (`127.0.0.1`) to avoid Python worker callback timeout.

2. Interpreter consistency
- Ensured Spark driver/worker use the same Python interpreter (`FDE_env`) in `spark_consumer.py`.

3. Model loading mismatch
- Consumer `Autoencoder` decoder architecture was aligned with training model shape so `state_dict` loads.

4. Pandas/Arrow output schema mismatch
- Kept raw output frame types intact for Spark schema (`amount` remains string in output).
- Used separate cleaned frame for inference only.

5. Tensor conversion dtype failure
- Forced final model feature matrix to strict numeric `float32` before creating torch tensor.

### Current recommended run flow
Terminal 1:
```cmd
conda run -n FDE_env python "D:\Fraud Detection Engine\src\processing\spark_consumer.py"
```
Terminal 2:
```cmd
conda run -n FDE_env python "D:\Fraud Detection Engine\src\ingestion\stream_generator.py"
```

### Tomorrow focus areas
- Walk through full data flow: Kafka -> Spark parse -> feature alignment -> scaler -> autoencoder -> anomaly score.
- Define anomaly thresholding and alerting policy for scores.
- Add lightweight logging/checkpoint strategy for reliable restarts.

## Update (2026-02-16 - Evaluation + Mongo Pipeline Refactor)

### What was implemented
- Added offline evaluation script: `src/results/evaluate_autoencoder.py`
- Added recall-constrained threshold selection (target recall = 0.95).
- Saved artifacts to `src/results/`:
  - `threshold.json`
  - `metrics.json`
  - `threshold_sweep.csv`
  - `scored_holdout.csv`
  - `metrics_summary.md`
- Updated stream inference (`src/processing/spark_consumer.py`) to:
  - load threshold from `src/results/threshold.json`
  - add `is_suspected_fraud`
  - add `risk_band` (`low` / `medium` / `high`)
- Updated `src/results/mongo_writer.py` to read Mongo URI from `.env` key `uri`.
- Added `python-dotenv` to `requirements.txt`.
- Replaced empty `README.md` with architecture + run + evaluation documentation.

### Smoke test result
- `evaluate_autoencoder.py` ran successfully in `FDE_env` and generated all artifacts.
- Selected threshold (current run): `0.15471555782343915`

### Important caveat
- Scaler load warning appeared due sklearn version mismatch (artifact saved with 1.7.2, runtime 1.8.0).
- Action recommended: retrain/resave scaler + model artifacts in the current environment version for strict reproducibility.
