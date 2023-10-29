# %%

from traffic.core import Traffic
from mass import FuelEstimator
import numpy as np
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
mission_size = 2000


def test_openturns(X):
    # Transforming the input into np array
    # Xarray = np.array(X, copy=False)

    # Getting data from X
    # age = Xarray[:, 2]

    # Fuel Calculation with PDFs

    cumul = []
    for sample in X:
        descent_thrust = sample[0]
        cas_const_cl = sample[1]
        mach_const_cl = sample[2]
        cas_const_de = sample[3]
        mach_const_de = sample[4]
        range_cr = sample[5]
        mach_cr = sample[6]

        traj = gentraj(
            ac_type,
            cas_const_cl=cas_const_cl,
            mach_const_cl=mach_const_cl,
            cas_const_de=cas_const_de,
            mach_const_de=mach_const_de,
            range_cr=range_cr,
            alt_cr=9753.6,
            mach_cr=mach_cr,
            dt=60,
        )

        fe = FuelEstimator(
            ac_type=ac_type,
            passenger_mass=100,
            load_factor=0.819,
            descent_thrust=descent_thrust,
        )
        df = FlightPhaseEstimator()(_to_df(traj))
        fp = FlightProfiles.from_df(df)
        cumul.append([fe(fp).to_df().fc.iloc[-1]])

    return cumul


# test_openturns([[100, 0.8, 0.07]])

# %%


def get_dist(var):
    if var["statmodel"] == "beta":
        return ot.Beta(
            var["statmodel_params"][0],
            var["statmodel_params"][1],
            var["minimum"],
            var["maximum"],
        )
    elif var["statmodel"] == "norm":
        return ot.TruncatedDistribution(
            ot.Normal(*var["statmodel_params"]),
            var["minimum"],
            var["maximum"],
        )
    elif var["statmodel"] == "gamma":
        return ot.Gamma(
            var["statmodel_params"][0],
            1 / var["statmodel_params"][2],
            var["statmodel_params"][1],
        )


# %%

fun = ot.PythonFunction(7, 1, func=test_openturns, func_sample=test_openturns)
fpg = FlightProfileGenerator(ac_type=ac_type)
distribution = ot.ComposedDistribution(
    [
        ot.TruncatedDistribution(
            ot.Normal(0.3, 0.2), 0.05, ot.TruncatedDistribution.LOWER
        ),  # X2
        get_dist(fpg.wrap.climb_const_vcas()),  # X3
        get_dist(fpg.wrap.climb_const_mach()),  # X4
        get_dist(fpg.wrap.descent_const_vcas()),  # X5
        get_dist(fpg.wrap.descent_const_mach()),  # X6
        (ot.Normal(0, 0.1) + 1) * mission_size,  # X7
        get_dist(fpg.wrap.cruise_mach()),  # X9
    ]
)

distribution.setDescription(
    [
        "descent thrust",
        "cas climbing",
        "mach climbing",
        "cas descent",
        "mach descent",
        "range deviation",
        "cruising mach",
    ]
)

# %%

size = 2000
sie = ot.SobolIndicesExperiment(distribution, size)
inputDesign = sie.generate()
input_names = distribution.getDescription()
inputDesign.setDescription(input_names)
print("Sample size: ", inputDesign.getSize())

# %%

outputDesign = fun(inputDesign)

inputDesign.exportToCSVFile(f"results/input/{ac_type}_{mission_size}km_fixed_alt.csv")
outputDesign.exportToCSVFile(f"results/output/{ac_type}_{mission_size}km_fixed_alt.csv")

# %%

df_in = inputDesign.asDataFrame()
df_out = outputDesign.asDataFrame()
idx = df_out.y0[df_out.y0.isnull()].index
inputDesign = ot.Sample(df_in.drop(index=idx).values)
outputDesign = ot.Sample(df_out.drop(index=idx).values)

# %%
inputDesign.setDescription(distribution.getDescription())
sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)

# %%

output_dimension = fun.getOutputDimension()
for i in range(output_dimension):
    print("Output #", i)
    first_order = sensitivityAnalysis.getFirstOrderIndices(i)
    total_order = sensitivityAnalysis.getTotalOrderIndices(i)
    print("    First order indices: ", first_order)
    print("    Total order indices: ", total_order)

agg_first_order = sensitivityAnalysis.getAggregatedFirstOrderIndices()
agg_total_order = sensitivityAnalysis.getAggregatedTotalOrderIndices()
print("Agg. first order indices: ", agg_first_order)
print("Agg. total order indices: ", agg_total_order)
# %%

graph = sensitivityAnalysis.draw()
view = viewer.View(graph)

# %%
import altair as alt

points_1 = list(sensitivityAnalysis.getFirstOrderIndices())
points_t = list(sensitivityAnalysis.getTotalOrderIndices())
points = points_1 + points_t
lower_bound_1 = list(sensitivityAnalysis.getFirstOrderIndicesInterval().getLowerBound())
lower_bound_t = list(sensitivityAnalysis.getTotalOrderIndicesInterval().getLowerBound())
lower_bound = lower_bound_1 + lower_bound_t
upper_bound_1 = list(sensitivityAnalysis.getFirstOrderIndicesInterval().getUpperBound())
upper_bound_t = list(sensitivityAnalysis.getTotalOrderIndicesInterval().getUpperBound())
upper_bound = upper_bound_1 + upper_bound_t
names = list(distribution.getDescription()) * 2
cat = ["First Order"] * len(points_1) + ["Total Order"] * len(points_1)


source = pd.DataFrame(
    {
        "points": points,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "names": names,
        "cat": cat,
    }
)

# the base chart
base = alt.Chart(source)

means = (
    base.mark_circle(size=80)
    .encode(
        alt.X("points").scale(domain=(-0.15, 1.5)),
        alt.Y("cat", title=None),
        # alt.Row("names"),
        alt.Color("cat", title="Sobol indices (A320 - Fixed Payload + Altitude)")
        # yOffset="cat:N",
    )
    .properties(width=600, height=30)
)

errorbars = base.mark_errorbar(thickness=3).encode(
    alt.X("lower_bound", title="index value"),
    alt.X2("upper_bound"),
    alt.Y("cat"),
    # alt.Row("names"),
    alt.Color("cat")
    # y=alt.Y("names"),
    # yOffset="cat:N",
    # color=alt.value("#4682b4"),
)
names = base.mark_text(align="right").encode(alt.Text("names"), x=alt.value(590))

chart = (
    (means + errorbars + names)
    .facet(row="names")
    .configure_axisY(labelFontSize=0, tickSize=0)
    .configure_axisX(titleAnchor="start")
    .configure_header(titleFontSize=0, labelAngle=0, labelFontSize=0)
    .configure_legend(orient="top", titleFontSize=20, titleLimit=2000)
    .configure_facet(spacing=0)
    .configure_text(font="Calibri")
    .properties(bounds="flush")
)
chart

# %%

df_all = pd.concat([df_in, df_out], axis=1)

alt.Chart(df_all.sample(5000, random_state=0)).mark_circle(size=6).encode(
    x=alt.X("cas climbing", scale=alt.Scale(domain=(140, 161))),
    y=alt.Y("y0", scale=alt.Scale(domain=(3650, 3850))),
)

# %%
