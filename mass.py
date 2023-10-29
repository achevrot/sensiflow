# %%

from typing import Annotated, Any
from impunity import impunity
import pandas as pd
import numpy as np
from openap import prop
from fleet import FleetData
import openturns.viewer as viewer
from math import asin
from openap import FuelFlow, Thrust, prop, Drag
from flight import FlightProfileGenerator, FlightProfiles
from matplotlib import pylab as plt


class FuelEstimator:
    def __init__(self, ac_type, eng_type=None, use_synonym=True, **kwargs):
        self.use_synonym = use_synonym
        self.ac_type = ac_type
        self.eng_type = eng_type
        self.thrust = Thrust(ac=self.ac_type, eng=self.eng_type)

        self.drag = Drag(ac=ac_type)
        if "descent_thrust" in kwargs:
            self.descent_ratio = kwargs["descent_thrust"]
        else:
            self.descent_ratio = 0.07
        self.fuelflow = FuelFlow(ac=self.ac_type, eng=self.eng_type, **kwargs)
        self.mass = Mass(
            self, ac_type=self.ac_type.split("-")[0], eng_type=self.eng_type, **kwargs
        )

    def __call__(self, flight_profiles, last_point=False):
        length = len(flight_profiles)

        def generate():
            for fp in flight_profiles:
                mass = self.mass.compute_tow(fp)
                if mass is None:
                    print("bobo")
                yield self.compute_fuel(fp, mass, last_point)

        return FlightProfiles(generate(), length)

    def compute_fuel(
        self,
        flight_profile: Annotated[Any, "dimensionless"],
        mass: Annotated[Any, "kg"],
        last_point=False,
    ) -> None:
        """
        Compute fuel from flight profile:
        Iteration over Dataframe needed:
            Implemented via numpy array iteration (faster than iterrows or itertuples):
            https://towardsdatascience.com/heres-the-most-efficient-way-to-iterate-through-your-pandas-dataframe-4dad88ac92ee
            TODO: cython, numba implementation ?
        """

        def compute_thr(
            fp: Annotated[Any, "dimensionless"],
            v: Annotated[Any, "m/s"],
            h: Annotated[Any, "ft"],
            vs: Annotated[Any, "m/s"],
            m: Annotated[Any, "kg"],
        ) -> Annotated[Any, "newton"]:
            if fp == "TO":
                return self.thrust.takeoff(tas=v, alt=h)
            if fp == "CL":
                return self.thrust.climb(tas=v, alt=h, roc=vs)
            if fp == "DE":
                return self.thrust.descent_idle(tas=v, alt=h, ratio=self.descent_ratio)
            if fp == "NA":
                return np.nan
            else:
                angle: Annotated[Any, "radians"] = asin(vs / v)
                return self.drag.clean(tas=v, alt=h, mass=m, path_angle=angle)

        def compute_ff(
            fp: Annotated[Any, "dimensionless"],
            v: Annotated[Any, "m/s"],
            h: Annotated[Any, "ft"],
            vs: Annotated[Any, "m/s"],
            m: Annotated[Any, "kg"],
            thr: Annotated[Any, "newton"],
        ) -> Annotated[Any, "kg/s"]:
            if fp == "TO":
                return self.fuelflow.takeoff(tas=v, alt=h, throttle=1)

            if fp == "CL":
                return self.fuelflow.at_thrust(thr, alt=h)

            angle: Annotated[Any, "radians"] = asin(vs / v)
            return self.fuelflow.enroute(
                mass=m, tas=v, alt=h, path_angle=angle, limit=True
            )

        flight_profile = flight_profile.assign(
            thr=np.NaN, ff=np.NaN, fc=np.NaN, m=np.NaN
        )

        t: Annotated[Any, "s"] = flight_profile.t.values
        fp: Annotated[Any, "dimensionless"] = flight_profile.fp.values
        v: Annotated[Any, "m/s"] = flight_profile.v.values
        h: Annotated[Any, "m"] = flight_profile.h.values
        vs: Annotated[Any, "m/s"] = flight_profile.vs.values
        thr: Annotated[Any, "newton"] = flight_profile.thr.values
        ff: Annotated[Any, "kg/s"] = flight_profile.ff.values
        fc: Annotated[Any, "kg"] = flight_profile.fc.values
        m: Annotated[Any, "kg"] = flight_profile.m.values

        thr[0], ff[0], fc[0], m[0] = 0.0, 0.0, 0.0, mass
        for i in range(1, len(flight_profile)):
            thr[i] = compute_thr(
                fp[i],
                v[i],
                h[i],
                vs[i],
                m[i - 1],
            )
            if thr[i] == np.NaN:
                thr[i] = thr[i - 1]
            ff[i] = compute_ff(fp[i], v[i], h[i], vs[i], m[i - 1], thr[i])
            if ff[i] == np.NaN:
                ff[i] = ff[i - 1]
            fc[i] = ff[i] * (t[i] - t[i - 1])
            m[i] = m[i - 1] - fc[i]

        flight_profile = flight_profile.assign(fc=flight_profile.fc.cumsum(skipna=True))
        flight_profile["ff"] = flight_profile["ff"].astype("float")
        if last_point:
            last_point = flight_profile.query("m==m").iloc[-1]
            return pd.DataFrame.from_records(
                [(last_point.id, last_point.s, last_point.fc, last_point.m)],
                columns=["id", "fd", "fc", "m"],
            )
        return flight_profile


class Mass:
    def __init__(self, fe, ac_type, eng_type=None, **kwargs):
        self.aircraft = prop.aircraft(ac_type)
        self.fleet = FleetData(ac_type)
        self.fpg = FlightProfileGenerator(ac_type=ac_type, eng_type=eng_type)
        self.fe = fe
        if "passenger_mass" in kwargs and "load_factor" in kwargs:
            self.passenger_mass = kwargs["passenger_mass"]
            self.load_factor = kwargs["load_factor"]
        else:
            self.passenger_mass = [100]
            self.load_factor = [0.819]

    @property
    def oew(self):
        return self.aircraft["limits"]["OEW"]

    def compute_payload_mass(self) -> Annotated[float, "kg"]:
        return self.load_factor * self.fleet.get_avg_num_seats() * self.passenger_mass

    def compute_tow(self, flight_profile, **kwargs) -> Annotated[Any, "kg"]:
        """
        See appendix G FEAT paper for algorithm details fuel load estimation:
        https://ars.els-cdn.com/content/image/1-s2.0-S136192092030715X-mmc8.pdf
        """

        res_cruise: Annotated[Any, "kg"] = self.fpg.gen_cruise_for_fuel_reserve()
        alt_flight: Annotated[Any, "kg"] = self.fpg.gen_flight_for_alternate_fuel()

        payload = self.compute_payload_mass()

        tow = self.oew + payload + 0.15 * self.oew
        cumul = []
        while True:
            # Fuel for trip
            f_trip: Annotated[Any, "kg"] = self.fe.compute_fuel(
                flight_profile, mass=tow, last_point=True
            ).fc.item()

            # 5% of trip fuel
            f_cont: Annotated[Any, "kg"] = f_trip * 0.05
            landing_mass: Annotated[Any, "kg"] = tow - f_trip

            #
            f_res: Annotated[Any, "kg"] = self.fe.compute_fuel(
                res_cruise, mass=landing_mass, last_point=True
            ).fc.item()
            f_alt: Annotated[Any, "kg"] = self.fe.compute_fuel(
                alt_flight, mass=landing_mass, last_point=True
            ).fc.item()
            # + 1 for taxying (most case)
            m_fuel: Annotated[Any, "kg"] = f_trip + f_cont + f_res + f_alt + 1
            new_tow: Annotated[Any, "kg"] = self.oew + payload + m_fuel
            cumul.append((f_trip, f_cont, f_res, f_alt, m_fuel, tow, new_tow))
            if abs(tow - new_tow) < 10:
                break
            tow = new_tow
        return tow


# %%
