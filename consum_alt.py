# %%
from traffic.core import Traffic
from mass import FuelEstimator
import numpy as np
import altair as alt
import pandas as pd
from flight import (
    FlightProfiles,
    FlightProfileGenerator,
    _to_df,
    gen_flight_profile,
    FlightPhaseEstimator,
    gentraj,
)
import openturns as ot
import openturns.viewer as viewer

ot.RandomGenerator.SetSeed(0)

# %%

inputDesign = ot.Sample.ImportFromCSVFile("results/input/A320_2300km.csv")
outputDesign = ot.Sample.ImportFromCSVFile("results/output/A320_2300km.csv")

df_in = inputDesign.asDataFrame()
df_out = outputDesign.asDataFrame()

# -------------------------------------------------------
#
# Chart for L / 100 RPKs
#
# -------------------------------------------------------
ac_type = "A320"
df_all = pd.concat([df_in, df_out], axis=1)

base_3 = (
    alt.Chart(
        df_all.sample(5000, random_state=0),
        title=alt.Title("Consumption in liters of kerosen", anchor="start"),
    )
    .mark_circle()
    .encode(
        (alt.X("cruising altitude").bin(maxbins=22).title("Altitude")),
        (alt.Y("y0", scale=alt.Scale(domain=(0, 10000)))),
    )
)

mean = base_3.transform_regression("cruising altitude", "y0").mark_line().encode()

# %%

chart = (
    (base_3 + mean)
    .configure_axisX(titleAnchor="middle")
    .configure_header(titleFontSize=0, labelAngle=0, labelFontSize=0)
    .configure_text(font="Calibri")
    .configure_legend(
        strokeColor="gray",
        fillColor="#EEEEEE",
        padding=10,
        cornerRadius=10,
        orient="top-right",
    )
).properties(width=800, height=400)

chart
# %%
