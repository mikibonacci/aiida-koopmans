from collections import UserDict
from typing import Any

from koopmans.base import BaseModel
from koopmans.files import File
from koopmans.status import Status
from pydantic import Field


class AiiDAStepData(BaseModel):
    status: Status
    workchain: int | None = None
    remote_folder: int | None = None
    input_files: dict[str, File] = Field(default_factory=dict)


class StepsDict(UserDict[str, AiiDAStepData]):
    """A dict class that enforces AiiDAStepData as values."""

    def __setitem__(self, key, value):
        if not isinstance(value, AiiDAStepData):
            raise TypeError("Value must be an instance of AiiDAStepData")
        super().__setitem__(key, value)


class AiiDAStepsData(BaseModel):
    """
    This class is used to store the step data in a dictionary.
    It contains the following information:
    - step_data = {calc.directory: {'workchain': workchain, 'remote_folder': remote_folder}}
    and any other info we need for AiiDA.
    """

    configuration: dict[str, Any] = Field(default_factory=dict)
    steps: StepsDict = Field(default_factory=StepsDict)
    pseudo_family: str | None = None
    structure: int | None = None

    # def some model validator or computed field:
    # here we add the logic to populate configuration by default
    # 1. we look for codes stored in AiiDA at localhost, e.g. pw-version@localhost,
    # 2. we look for codes in the PATH,
    # 3. if we don't find the code in AiiDA db but in the PATH, we store it in AiiDA db.
    # 4. if we don't find the code in AiiDA db and in the PATH and not configuration is provided, we raise an error.
    # 5. if no resource info in configuration, we try to look at PARA_PREFIX env var.
