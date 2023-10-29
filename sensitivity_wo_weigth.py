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

outputDesign = fun(inputDesign)

inputDesign.exportToCSVFile(f"results/input/{ac_type}_{mission_size}km_wo_weight.csv")
outputDesign.exportToCSVFile(f"results/output/{ac_type}_{mission_size}km_wo_weight.csv")

# %%
