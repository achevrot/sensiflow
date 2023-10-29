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

ac_type = input("ac_type: ")
mission_size = int(input("cruise range: "))


def test_openturns(X):
    # Transforming the input into np array
    # Xarray = np.array(X, copy=False)

    # Getting data from X
    # age = Xarray[:, 2]

    # Fuel Calculation with PDFs

    cumul = []
    for sample in X:
        load_factor = sample[0]
        weight_person = sample[1]
        descent_thrust = sample[2]
        cas_const_cl = sample[3]
        mach_const_cl = sample[4]
        cas_const_de = sample[5]
        mach_const_de = sample[6]
        range_cr = sample[7]
        alt_cr = sample[8]
        mach_cr = sample[9]

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
            passenger_mass=weight_person,
            load_factor=load_factor,
            descent_thrust=descent_thrust,
        )
        df = FlightPhaseEstimator()(_to_df(traj))
        fp = FlightProfiles.from_df(df)
        cumul.append([fe(fp, last_point=True).to_df().fc.item()])

    return cumul


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

fun = ot.PythonFunction(10, 1, func=test_openturns, func_sample=test_openturns)
fpg = FlightProfileGenerator(ac_type=ac_type)
distribution = ot.ComposedDistribution(
    [
        ot.TruncatedDistribution(
            ot.Normal(0.819, 0.2), 1, ot.TruncatedDistribution.UPPER
        ),  # X0
        ot.TruncatedDistribution(
            (ot.Normal(0, 0.2) + 1) * 100, 80, ot.TruncatedDistribution.LOWER
        ),  # X1
        ot.TruncatedDistribution(
            ot.Normal(0.3, 0.2), 0.05, ot.TruncatedDistribution.LOWER
        ),  # X2
        get_dist(fpg.wrap.climb_const_vcas()),  # X3
        get_dist(fpg.wrap.climb_const_mach()),  # X4
        get_dist(fpg.wrap.descent_const_vcas()),  # X5
        get_dist(fpg.wrap.descent_const_mach()),  # X6
        (ot.Normal(0, 0.01) + 1) * mission_size,  # X7
        get_dist(fpg.wrap.cruise_alt()) * 1000 * 3.28084,  # X8
        get_dist(fpg.wrap.cruise_mach()),  # X9
    ]
)

distribution.setDescription(
    [
        "load factor",
        "avg person weight",
        "descent thrust",
        "cas climbing",
        "mach climbing",
        "cas descent",
        "mach descent",
        "cruising range",
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

# %%

inputDesign.exportToCSVFile(f"results/input/{ac_type}_{mission_size}km.csv")
outputDesign.exportToCSVFile(f"results/output/{ac_type}_{mission_size}km.csv")
# %%
