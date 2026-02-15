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
