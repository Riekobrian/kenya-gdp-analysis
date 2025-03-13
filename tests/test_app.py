import pandas as pd
import pytest
import sys
import os

# Add the parent directory to path so we can import functions from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import calculate_real_gdp


def test_calculate_real_gdp():
    # Create a simple test dataframe
    test_data = pd.DataFrame(
        {
            "Year": [1999, 2000, 2001],
            "Nominal GDP (Ksh Million)": [1000000, 1100000, 1200000],
            "Annual GDP Growth (%)": [2.0, 3.0, 4.0],
            "Real GDP (Ksh Million)": [None, 1000000, 1040000],
        }
    )

    # Process the data
    result = calculate_real_gdp(test_data)

    # Check that calculated values are as expected
    assert (
        not result["Calculated Real GDP"].isna().any()
    ), "There should be no NaN values in Calculated Real GDP"
    assert (
        result.loc[result["Year"] == 2000, "Calculated Real GDP"].values[0] == 1000000
    )

    # Calculate expected value for 1999 based on growth rate
    expected_1999 = 1000000 / (1 + 0.02)
    assert (
        abs(
            result.loc[result["Year"] == 1999, "Calculated Real GDP"].values[0]
            - expected_1999
        )
        < 1
    ), "1999 Calculated Real GDP should match expected value"
