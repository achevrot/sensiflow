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
        alt_cr = sample[6]
        mach_cr = sample[7]

        traj = gentraj(
            ac_type,
            cas_const_cl=cas_const_cl,
            mach_const_cl=mach_const_cl,
            cas_const_de=cas_const_de,
            mach_const_de=mach_const_de,
            range_cr=range_cr,
            alt_cr=alt_cr,
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

fun = ot.PythonFunction(8, 1, func=test_openturns, func_sample=test_openturns)
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
        get_dist(fpg.wrap.cruise_alt()) * 1000,  # X8
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
        "cruising altitude",
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

inputDesign = ot.Sample.ImportFromCSVFile("results/input/A320_2000km_wo_weight.csv")
outputDesign = ot.Sample.ImportFromCSVFile("results/output/A320_2000km_wo_weight.csv")

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

# corr_analysis = ot.CorrelationAnalysis(inputDesign, outputDesign)
# pcc_indices = corr_analysis.computePCC()
# print(pcc_indices)
# # %%

# graph = ot.SobolIndicesAlgorithm.DrawCorrelationCoefficients(
#     pcc_indices, input_names, "PCC coefficients - Wing weight"
# )
# view = viewer.View(graph)

# # %%

# X = ot.RandomVector(distribution)
# Y = ot.CompositeRandomVector(fun, X)
# taylor = ot.TaylorExpansionMoments(Y)
# print(taylor.getImportanceFactors())
# # %%
# graph = taylor.drawImportanceFactors()
# graph.setTitle("Taylor expansion imporfance factors - Wing weight")
# view = viewer.View(graph)
# # %%

# sizePCE = 1000
# inputDesignPCE = distribution.getSample(sizePCE)
# outputDesignPCE = fun(inputDesignPCE)
# # %%
# algo = ot.FunctionalChaosAlgorithm(inputDesignPCE, outputDesignPCE, distribution)

# algo.run()
# result = algo.getResult()
# print(result.getResiduals())
# print(result.getRelativeErrors())

# # %%

# sensitivityAnalysis = ot.FunctionalChaosSobolIndices(result)
# print(sensitivityAnalysis)
# firstOrder = [sensitivityAnalysis.getSobolIndex(i) for i in range(len(input_names))]
# totalOrder = [
#     sensitivityAnalysis.getSobolTotalIndex(i) for i in range(len(input_names))
# ]
# graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(input_names, firstOrder, totalOrder)
# graph.setTitle("Sobol indices by Polynomial Chaos Expansion - wing weight")
# view = viewer.View(graph)

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
        alt.Color("cat", title="Sobol indices (A320 - Fixed Payload)")
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
