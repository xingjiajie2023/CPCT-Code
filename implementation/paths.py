import os
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get("DCIT_DATA_ROOT",str(PACKAGE.parent/"real_data/DCIT_REAL_DATA_v1"))).expanduser().resolve()
OUTPUT=Path(os.environ.get("CPCT_OUTPUT_ROOT",str(PACKAGE/"outputs"))).expanduser().resolve()
PROFILE_DIR=OUTPUT/"source"
