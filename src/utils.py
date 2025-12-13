# Helper functions that we can treate as utilities ..these are common functions which we write here to promote reusability

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
