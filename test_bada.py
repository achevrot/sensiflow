# -*- coding: utf-8 -*-
# %%
import os
import sys
import inspect
from math import *
from dataclasses import dataclass
from importlib import util
import pandas as pd

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


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/pyBADA")

import pybada.atmosphere as atm
import pybada.conversions as conv
import pybada.TCL as TCL
from pybada.flightTrajectory import FlightTrajectory as FT

# import BADA3, BADA4 and BADAH modules
# handling missing/selective availability of BADA 3/4/H modules
from pybada.bada4 import Bada4Aircraft

# %%


@dataclass
class target:
    ROCDtarget: float = None
    slopetarget: float = None
    acctarget: float = None
    ESFtarget: float = None


def bada_calc(
    ac_type,
    payload,
    fuel,
    cruising_dist,  # NM
    Mcl,
    Mcr,
    Mdes,
    Vcl2,
    Vdes2,
    Hp_CR=33000,  # [ft] CRUISing level,
):
    # define path to the folder where BADA model can be found (modify if necessary)
    aircraft_path_BADA3 = parentdir + "/pybada/aircraft/BADA3/"
    aircraft_path_BADA4 = parentdir + "/pybada/aircraft/BADA4/"
    output_path = parentdir + "/sensiflow/bada_test"

    # initialization of BADA3/4
    # uncomment for testing different BADA family if available
    AC = Bada4Aircraft(aircraft_path_BADA4, ac_type)

    # create a Flight Trajectory object to store the output from TCL segment calculations
    ft = FT()

    # default parameters
    speedType = "CAS"  # {M, CAS, TAS}
    wS = 0  # [kt] wind speed
    ba = 0  # [deg] bank angle
    DeltaTau = 0  # [K] delta temperature from ISA

    # Initial conditions
    m_init = AC.OEW + payload + fuel  # [kg] initial mass
    Hp_RWY = 0  # [ft] RWY altitude

    # take-off conditions
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(Hp_RWY), DeltaTau=DeltaTau
    )  # atmosphere properties at RWY altitude
    [cas_cl1, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(Hp_RWY), mass=m_init, theta=theta, delta=delta, DeltaTau=DeltaTau
    )  # [m/s] take-off CAS

    # BADA speed schedule
    [Vcl1, _, _] = AC.flightEnvelope.getSpeedSchedule(
        phase="Climb"
    )  # BADA Climb speed schedule
    [Vcr1, Vcr2, _] = AC.flightEnvelope.getSpeedSchedule(
        phase="Cruise"
    )  # BADA Cruise speed schedule
    [Vdes1, _, _] = AC.flightEnvelope.getSpeedSchedule(
        phase="Descent"
    )  # BADA Descent speed schedule

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # CLIMB to threshold altitude 1500ft at take-off speed
    # ------------------------------------------------
    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(cas_cl1),
        Hp_init=Hp_RWY,
        Hp_final=1499,
        m_init=m_init,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for below 3000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(2999), DeltaTau=DeltaTau
    )
    [cas_cl2, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(2999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas_cl2),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to threshold altitude 3000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=2999,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for below 4000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(3999), DeltaTau=DeltaTau
    )
    [cas_cl3, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(3999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas_cl3),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to threshold altitude 4000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=3999,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for below 5000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(4999), DeltaTau=DeltaTau
    )
    [cas_cl4, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(4999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas_cl4),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to threshold altitude 5000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=4999,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for below 6000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(5999), DeltaTau=DeltaTau
    )
    [cas_cl5, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(5999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas_cl5),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to threshold altitude 6000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=5999,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for below 10000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(9999), DeltaTau=DeltaTau
    )
    [cas_cl6, speedUpdated] = AC.ARPM.climbSpeed(
        h=conv.ft2m(9999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas_cl6),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to threshold altitude 10000ft
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=9999,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # accelerate according to BADA ARPM for above 10000ft and below crossover altitude
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(Vcl2),
        Hp_init=Hp,
        control=None,
        phase="Climb",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # CLIMB to crossover altitude
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # calculate the crosover altitude for climb phase
    crossoverAltitude = conv.m2ft(atm.crossOver(Vcl2, Mcl))

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=crossoverAltitude,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # climb at M from crossover altitude
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="M",
        v=Mcl,
        Hp_init=Hp,
        Hp_final=Hp_CR,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # if not at CR speed -> adapt the speed first (acc/dec)
    # ------------------------------------------------
    # current values
    Hp, m_final, M_final = ft.getFinalValue(AC, ["Hp", "mass", "M"])

    if M_final < Mcr:
        control = target(acctarget=0.5)
        flightTrajectory = TCL.accDec(
            AC=AC,
            speedType="M",
            v_init=M_final,
            v_final=Mcr,
            Hp_init=Hp,
            control=control,
            phase="Cruise",
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
        ft.append(AC, flightTrajectory)

    # CRUISE for 200 NM
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    flightTrajectory = TCL.constantSpeedCruise(
        AC=AC,
        lengthType="distance",
        length=cruising_dist,
        speedType="M",
        v=Mcr,
        Hp_init=Hp,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # # CRUISE for 200 NM
    # # ------------------------------------------------
    # # current values

    # Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    # flightTrajectory = TCL.constantSpeedCruise(
    #     AC=AC,
    #     lengthType="distance",
    #     length=200,
    #     step_length=50,
    #     maxRFL=36000,
    #     speedType="M",
    #     v=Mcr,
    #     Hp_init=Hp,
    #     m_init=m_final,
    #     stepClimb=True,
    #     wS=wS,
    #     bankAngle=ba,
    #     DeltaTau=DeltaTau,
    # )
    # ft.append(AC, flightTrajectory)

    # acc/dec to DESCENT speed during the descend
    # ------------------------------------------------
    # current values
    Hp, m_final, M_final = ft.getFinalValue(AC, ["Hp", "mass", "M"])

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="M",
        v_init=M_final,
        v_final=Mdes,
        Hp_init=Hp,
        phase="Descent",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend to crossover altitude
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    # calculate the crosover altitude for descend phase
    crossoverAltitude = conv.m2ft(atm.crossOver(Vdes2, Mdes))

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="M",
        v=Mdes,
        Hp_init=Hp,
        Hp_final=crossoverAltitude,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend to FL100
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(Vdes2),
        Hp_init=Hp,
        Hp_final=10000,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # decelerate according to BADA ARPM for below FL100
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(9999), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(9999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas),
        Hp_init=Hp,
        phase="Descent",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend to 6000ft
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(cas),
        Hp_init=Hp,
        Hp_final=6000,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # decelerate according to BADA ARPM for below 6000
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(5999), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(5999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas),
        Hp_init=Hp,
        phase="Descent",
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend to 5000ft
    # ------------------------------------------------
    # current values
    Hp, m_final = ft.getFinalValue(AC, ["Hp", "mass"])

    flightTrajectory = TCL.constantSpeedRating(
        AC=AC,
        speedType="CAS",
        v=conv.ms2kt(cas),
        Hp_init=Hp,
        Hp_final=5000,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope to next altitude threshold
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    if AC.BADAFamily.BADA3:
        flightTrajectory = TCL.constantSpeedSlope(
            AC=AC,
            speedType="CAS",
            v=CAS_final,
            Hp_init=Hp,
            Hp_final=3700,
            slopetarget=-3.0,
            config="AP",
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
    elif AC.BADAFamily.BADA4:
        flightTrajectory = TCL.constantSpeedSlope(
            AC=AC,
            speedType="CAS",
            v=CAS_final,
            Hp_init=Hp,
            Hp_final=3000,
            slopetarget=-3.0,
            config=None,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )

    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope while decelerating
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(2999), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(2999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    control = target(slopetarget=-3.0)
    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas),
        Hp_init=Hp,
        control=control,
        phase="Descent",
        config="AP",
        speedBrakes=True,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope to next altitude threshold
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    if Hp > 2000:
        flightTrajectory = TCL.constantSpeedSlope(
            AC=AC,
            speedType="CAS",
            v=CAS_final,
            Hp_init=Hp,
            Hp_final=2000,
            slopetarget=-3.0,
            config=None,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
        ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope while decelerating
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(1999), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(1999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    control = target(slopetarget=-3.0)
    flightTrajectory = TCL.accDec(
        AC=AC,
        speedType="CAS",
        v_init=CAS_final,
        v_final=conv.ms2kt(cas),
        Hp_init=Hp,
        control=control,
        phase="Descent",
        config="LD",
        speedBrakes=True,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope to next altitude threshold
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    if Hp > 1500:
        flightTrajectory = TCL.constantSpeedSlope(
            AC=AC,
            speedType="CAS",
            v=CAS_final,
            Hp_init=Hp,
            Hp_final=1500,
            slopetarget=-3.0,
            config="LD",
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
        ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope while decelerating
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(1499), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(1499), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    control = target(slopetarget=-3.0)
    if AC.BADAFamily.BADA3:
        flightTrajectory = TCL.accDec(
            AC=AC,
            speedType="CAS",
            v_init=CAS_final,
            v_final=conv.ms2kt(cas),
            Hp_init=Hp,
            control=control,
            phase="Descent",
            config="LD",
            speedBrakes=True,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
    elif AC.BADAFamily.BADA4:
        flightTrajectory = TCL.accDec(
            AC=AC,
            speedType="CAS",
            v_init=CAS_final,
            v_final=conv.ms2kt(cas),
            Hp_init=Hp,
            control=control,
            phase="Descent",
            config="LD",
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope to next altitude threshold
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    if Hp > 1000:
        flightTrajectory = TCL.constantSpeedSlope(
            AC=AC,
            speedType="CAS",
            v=CAS_final,
            Hp_init=Hp,
            Hp_final=1000,
            slopetarget=-3.0,
            config=None,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
        ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope while decelerating
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    # get BADA target speed from BADA ARPM procedure for the altitude bracket below
    [theta, delta, sigma] = atm.atmosphereProperties(
        h=conv.ft2m(999), DeltaTau=DeltaTau
    )
    [cas, speedUpdated] = AC.ARPM.descentSpeed(
        h=conv.ft2m(999), mass=m_final, theta=theta, delta=delta, DeltaTau=DeltaTau
    )

    control = target(slopetarget=-3.0)
    if AC.BADAFamily.BADA3:
        flightTrajectory = TCL.accDec(
            AC=AC,
            speedType="CAS",
            v_init=CAS_final,
            v_final=conv.ms2kt(cas),
            Hp_init=Hp,
            control=control,
            phase="Descent",
            config=None,
            speedBrakes=True,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
    elif AC.BADAFamily.BADA4:
        flightTrajectory = TCL.accDec(
            AC=AC,
            speedType="CAS",
            v_init=CAS_final,
            v_final=conv.ms2kt(cas),
            Hp_init=Hp,
            control=control,
            phase="Descent",
            config=None,
            m_init=m_final,
            wS=wS,
            bankAngle=ba,
            DeltaTau=DeltaTau,
        )
    ft.append(AC, flightTrajectory)

    # descend on ILS with 3deg glideslope to next altitude threshold
    # ------------------------------------------------
    # current values
    Hp, m_final, CAS_final = ft.getFinalValue(AC, ["Hp", "mass", "CAS"])

    flightTrajectory = TCL.constantSpeedSlope(
        AC=AC,
        speedType="CAS",
        v=CAS_final,
        Hp_init=Hp,
        Hp_final=Hp_RWY,
        slopetarget=-3.0,
        config=None,
        m_init=m_final,
        wS=wS,
        bankAngle=ba,
        DeltaTau=DeltaTau,
    )
    ft.append(AC, flightTrajectory)

    # save the output to a CSV file
    # ------------------------------------------------
    return pd.DataFrame.from_dict(next(iter(ft.FT.items()))[1], orient="index").T


# %%


ac_type = "A320"
ac_type_bada = "A320-271N"
mission_size = 800


def test_openturns(X):
    # Transforming the input into np array
    # Xarray = np.array(X, copy=False)

    # Getting data from X
    # age = Xarray[:, 2]

    # Fuel Calculation with PDFs

    def fuel_estimate():
        load_factor = X[0][0]
        weight_person = X[0][1]
        cas_const_cl = X[0][2]
        mach_const_cl = X[0][3]
        cas_const_de = X[0][4]
        mach_const_de = X[0][5]
        range_cr = X[0][6]
        alt_cr = X[0][7]
        mach_cr = X[0][8]

        traj = gentraj(
            ac_type,
            cas_const_cl=cas_const_cl,
            mach_const_cl=mach_const_cl,
            cas_const_de=cas_const_de,
            mach_const_de=mach_const_de,
            range_cr=range_cr,
            alt_cr=alt_cr,
            mach_cr=mach_cr,
            dt=20,
        )

        fe = FuelEstimator(
            ac_type=ac_type,
            passenger_mass=weight_person,
            load_factor=load_factor,
        )
        df = FlightPhaseEstimator()(_to_df(traj))
        fp = FlightProfiles.from_df(df)
        return fe(fp).to_df().fc.iloc[-1]

    fuel = fuel_estimate()
    seats = pd.read_csv("data/seats.csv")
    seats = int(seats.query("ac_type == @ac_type").nb_seats)
    cumul = []
    for sample in X:
        load_factor = sample[0]
        weight_person = sample[1]
        cas_const_cl = sample[2]
        mach_const_cl = sample[3]
        cas_const_de = sample[4]
        mach_const_de = sample[5]
        range_cr = sample[6]
        alt_cr = sample[7]
        mach_cr = sample[8]

        traj = bada_calc(
            ac_type_bada,
            (int(load_factor * seats) * weight_person),
            fuel,
            range_cr * 0.539957,  # NM
            mach_const_cl,
            mach_cr,
            mach_const_de,
            cas_const_cl,
            cas_const_de,
            alt_cr,  # [ft] CRUISing level,
        )

        traj["ff"] = (traj["time"] - traj["time"].shift(1)) * traj["FUEL"]
        cumul.append([traj.ff.sum()])

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

fun = ot.PythonFunction(9, 1, func=test_openturns, func_sample=test_openturns)
fpg = FlightProfileGenerator(ac_type=ac_type)
distribution = ot.ComposedDistribution(
    [
        ot.TruncatedDistribution(
            ot.Normal(0.819, 0.2), 1, ot.TruncatedDistribution.UPPER
        ),  # X0
        ot.TruncatedDistribution(
            (ot.Normal(0, 0.2) + 1) * 100, 80, ot.TruncatedDistribution.LOWER
        ),  # X1
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
        "cas climbing",
        "mach climbing",
        "cas descent",
        "mach descent",
        "cruise range",
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

inputDesign.exportToCSVFile(f"bada_test/input/{ac_type}_{mission_size}km.csv")
outputDesign.exportToCSVFile(f"bada_test/output/{ac_type}_{mission_size}km.csv")

# %%

inputDesign = ot.Sample.ImportFromCSVFile("bada_test/input/A320_800km.csv")
outputDesign = ot.Sample.ImportFromCSVFile("bada_test/output/A320_800km.csv")


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
        alt.X("points").scale(domain=(-0.15, 1)),
        alt.Y("cat", title=None),
        # alt.Row("names"),
        alt.Color("cat", title="Sobol indices (A320 - 800km - BADA 4)")
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
alt.Chart(df_all.sample(5000, random_state=0)).mark_circle(size=60).encode(
    x=alt.X("range deviation").scale(domain=(3000, 10000)),
    y=alt.Y("y0").scale(domain=(50000, 70000)),
).interactive()

# %%
