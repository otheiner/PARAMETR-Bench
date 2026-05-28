import re
import pandas as pd
from typing import ClassVar
from dataclasses import dataclass, field


@dataclass
class Metarubric:
    """
    A template + dataframe that unpacks into individual rubric criteria.

    Attributes:
        key:                    - short snake_case key to identify the metarubric
        source:                 - name of the dataframe in ground_truth.json that contains the data for this metarubric
        dimension:               - skill dimension; allowed values: "scientific reasoning" | "data handling" | "image data extraction" | "instructions following"
                                  (can be used for weighting and analysis by skill type)
        name:                   - short human readable name
        description:            - f-string with {column_name} placeholders - this gets unpacked
                                  to individual rubric criteria by unpacking metarubric
        weight:                 - overall weight of the metarubric item (when the metarubric
                                  is unpacked to multiple rubric criteria, the weight is distributed
                                  equally among them)
        columns:                - list of column names extracted from description placeholders
        dataframe:              - dataframe with columns corresponding to the placeholders in
                                  description
    """

    ALLOWED_DIMENSIONS: ClassVar[frozenset[str]] = frozenset({
        'scientific reasoning',
        'data handling',
        'image data extraction',
        'instructions following',
    })

    key :             str
    source:           str
    dimension:         str
    name:             str
    description:      str
    weight:           float = 1.0

    # Not passed in __init__ — computed from description
    columns:        list[str]    = field(init=False)
    dataframe:      pd.DataFrame = field(init=False)


    def __post_init__(self):
        """Called automatically after __init__."""
        self.columns   = re.findall(r'\{(\w+)[^}]*\}', self.description)
        self.dataframe = pd.DataFrame(columns=self.columns)


    def unpack(self) -> list[str]:
        """Expand description with each row of dataframe."""
        if self.source == 'none':
            return [self.description]

        return [
            self.description.format(**row.to_dict())
            for _, row in self.dataframe.iterrows()
        ]
