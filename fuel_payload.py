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

ac_type = "A320"

inputDesign = ot.Sample.ImportFromCSVFile("results/input/A320_580km.csv")
outputDesign = ot.Sample.ImportFromCSVFile("results/output/A320_580km.csv")

df_in = inputDesign.asDataFrame()
df_out = outputDesign.asDataFrame()

# -------------------------------------------------------
#
# Chart for L / 100 RPKs
#
# -------------------------------------------------------

df_all = pd.concat([df_in, df_out], axis=1)
seats = pd.read_csv("data/seats.csv")
seats = int(seats.query("ac_type == @ac_type").nb_seats)
df_all["nb_people"] = df_all["load factor"] * seats
df_all["ratio"] = (
    df_all["y0"] / df_all["nb_people"] / df_all["cruising range"] * 100 / 0.817
)
df_all["payload"] = df_all["load factor"] * seats * df_all["avg person weight"]
base_1 = (
    alt.Chart(
        df_all.sample(5000, random_state=0),
        title=alt.Title("Consumption in liters of kerosen per 100 RPK", anchor="start"),
    )
    .mark_circle()
    .encode(
        (
            alt.X("load factor", scale=alt.Scale(domain=(0, 1)))
            .bin(maxbins=20)
            .title("Load Factor")
        ),
        (alt.Y("y0", scale=alt.Scale(domain=(2600, 4000))).title("")),
        color=alt.datum("A320 (580 km)"),
    )
)

base_1
# %%

line = (
    alt.Chart(df_all.sample(5000, random_state=0))
    .mark_line()
    .encode(x="load factor", y="mean(y0)")
)

band = (
    alt.Chart(df_all.sample(5000, random_state=0))
    .mark_errorband(extent="ci")
    .encode(
        x="load factor",
        y=alt.Y("y0").title("Miles/Gallon"),
    )
)

band
# %%
