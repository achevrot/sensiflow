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

inputDesign = ot.Sample.ImportFromCSVFile("results/input/A320_2300km.csv")
outputDesign = ot.Sample.ImportFromCSVFile("results/output/A320_2300km.csv")

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

base_1 = (
    alt.Chart(
        df_all.sample(5000, random_state=0),
        title=alt.Title("Consumption in liters of kerosen per 100 RPK", anchor="start"),
    )
    .mark_circle()
    .encode(
        (
            alt.X("load factor", scale=alt.Scale(domain=(0.10, 1)))
            .bin(maxbins=20)
            .axis(format="%")
            .title("Load Factor")
        ),
        (alt.Y("ratio", scale=alt.Scale(domain=(0, 18))).title("")),
        color=alt.datum("A320 (2000 km)"),
    )
)


# %%

inputDesign = ot.Sample.ImportFromCSVFile("results/input/A321_2300km.csv")
outputDesign = ot.Sample.ImportFromCSVFile("results/output/A321_2300km.csv")

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


base_2 = (
    alt.Chart(
        df_all.sample(5000, random_state=0),
        title=alt.Title("Consumption in liters of kerosen per 100 RPK", anchor="start"),
    )
    .mark_circle(color="#FFAA00")
    .encode(
        (
            alt.X("load factor", scale=alt.Scale(domain=(0.10, 1)))
            .bin(maxbins=22)
            .axis(format="%")
            .title("Load Factor")
        ),
        (alt.Y("ratio", scale=alt.Scale(domain=(0, 30))).title("")),
        color=alt.datum("A320 (800 km)"),
    )
)

# %%

chart = (
    (base_2 + base_1)
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
