# =============================================================================
# FILE: coh_systems_demo.py
# Five COH-modelled intelligent systems across different domains.
# Run with:  python coh_systems_demo.py
# =============================================================================
"""Demonstrates five intelligent systems modelled with the gismol COH toolkit.

Every system has at least three nested layers of COHObject, exercises all nine
components of the 9-tuple (C, A, M, N, E, I, T, G, D), and is validated /
simulated / reasoned over at the end.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- core ------------------------------------------------------------------
from gismol.core.coh_object import Attribute, COHObject, Method, State
from gismol.core.constraints import (
    GoalConstraint,
    IdentityConstraint,
    TriggerConstraint,
)
from gismol.core.daemon_impl import (
    BarrierDaemon,
    HomeostasisDaemon,
    PrecisionWeightedDaemon,
)
from gismol.core.daemons import (
    BoundaryDaemon,
    DaemonNetwork,
    InvariantDaemon,
    RateDaemon,
    SafetyDaemon,
)
from gismol.core.dynamics import ContinuousFlow, DiscreteTransition
from gismol.core.embedding import FeatureEmbedder, HashEmbedder

# ---- neural ----------------------------------------------------------------
from gismol.neural.components import (
    MLPComponent,
    PolicyComponent,
    PredictorComponent,
    ValueComponent,
)

# ---- validation ------------------------------------------------------------
from gismol.nlp.validation import COHValidator
from gismol.reasoners.advanced import CausalReasoner, MetaReasoner
from gismol.reasoners.base import Query
from gismol.reasoners.domain import DomainReasoner, get_domain
from gismol.reasoners.general import ConstraintReasoner, GeneralReasoner, GoalSolver

# ---- stability / reasoning -------------------------------------------------
from gismol.stability.lyapunov import ConstraintStabilityAnalyzer


# =============================================================================
# Shared helpers
# =============================================================================
def _merge(state, **kwargs):
    """Return a new State with the given updates applied (non-mutating)."""
    out = state.copy() if hasattr(state, "copy") else dict(state)
    for k, v in kwargs.items():
        out[k] = v
    return out


def _bounded(name, default, lo, hi, description=""):
    return Attribute(name, default, description=description, bounds=(lo, hi))


def attach_standard_daemons(coh: COHObject) -> COHObject:
    """Attach invariant + boundary daemons derived from the object's attributes."""
    if coh.I:
        coh.add_daemon(InvariantDaemon(constraints=list(coh.I)))
    bounds = {a.name: (a.bounds or (-1e9, 1e9)) for a in coh.A.values()}
    if bounds:
        coh.add_daemon(BoundaryDaemon(bounds))
    return coh


def banner(title: str):
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


# =============================================================================
# SYSTEM 1 — Autonomous Warehouse Robot Fleet
# =============================================================================
def build_warehouse_fleet() -> COHObject:
    """Hierarchy:

        Fleet  (level 0)
         |-- PickingZone        (level 1)
         |    |-- RobotR2       (level 2)
         |    |    |-- Navigation  (level 3)
         |    |    |-- Gripper     (level 3)
         |    |    |-- Battery     (level 3)
         |    |-- RobotR7
         |-- PackingZone
         |    |-- RobotP1
         |-- ShippingZone
              |-- RobotS3
    """

    # ---- level 3 : robot subsystems -------------------------------------
    def navigation(name: str) -> COHObject:
        nav = COHObject(name, description="local planner + obstacle avoidance")
        nav.add_attribute(_bounded("x", 0.0, -1000, 1000, "position x (m)"))
        nav.add_attribute(_bounded("y", 0.0, -1000, 1000, "position y (m)"))
        nav.add_attribute(_bounded("heading", 0.0, -math.pi, math.pi, "yaw (rad)"))
        nav.add_attribute(_bounded("obstacle_distance", 5.0, 0.0, 20.0, "nearest obstacle (m)"))
        nav.add_method(Method(
            "advance",
            lambda s, a=None: (_merge(s, x=s["x"] + 0.5 * math.cos(s["heading"]),
                                      y=s["y"] + 0.5 * math.sin(s["heading"])), 0.0),
            description="move one step forward"))
        nav.add_identity_constraint(IdentityConstraint(
            "safety_margin", lambda s: s["obstacle_distance"] > 0.3,
            description="never collide"))
        nav.add_goal_constraint(GoalConstraint(
            "progress", lambda s: s["x"] + s["y"], direction="max"))
        nav.add_daemon(SafetyDaemon(
            lambda s: s["obstacle_distance"] - 0.3,
            threshold=0.0, intervention="emergency_stop"))
        nav.add_neural(PredictorComponent("trajectory", 4, 2, seed=hash(name) & 0xFFFF),
                       name="trajectory_predictor")
        nav.set_embedder(FeatureEmbedder(["x", "y", "heading"]))
        return nav

    def gripper(name: str) -> COHObject:
        g = COHObject(name, description="end-effector")
        g.add_attribute(_bounded("openness", 1.0, 0.0, 1.0))
        g.add_attribute(_bounded("payload_kg", 0.0, 0.0, 25.0))
        g.add_attribute(_bounded("grip_force", 0.0, 0.0, 200.0))
        g.add_method(Method("grasp", lambda s, a=None: (_merge(s, openness=0.0, grip_force=40.0), 1.0)))
        g.add_method(Method("release", lambda s, a=None: (_merge(s, openness=1.0, grip_force=0.0), 0.5)))
        g.add_identity_constraint(IdentityConstraint(
            "payload_limit", lambda s: s["payload_kg"] <= 20.0))
        g.add_trigger_constraint(TriggerConstraint(
            "overload", "payload_warning", lambda s: s["payload_kg"] > 18.0,
            "alert_maintenance", priority=2))
        g.add_goal_constraint(GoalConstraint(
            "throughput", lambda s: -s["openness"], direction="max"))
        attach_standard_daemons(g)
        g.add_neural(MLPComponent("force_ctrl", [3, 8, 1], seed=11), name="force_controller")
        return g

    def battery(name: str) -> COHObject:
        b = COHObject(name, description="power subsystem")
        b.add_attribute(_bounded("charge", 1.0, 0.0, 1.0))
        b.add_attribute(_bounded("temperature", 25.0, -10.0, 70.0))
        b.add_attribute(_bounded("drain_rate", 0.0, 0.0, 1.0))
        b.add_method(Method("consume", lambda s, a=None: (_merge(s, charge=s["charge"] - 0.01), 0.0)))
        b.add_method(Method("recharge", lambda s, a=None: (_merge(s, charge=min(1.0, s["charge"] + 0.05)), 0.0)))
        b.add_identity_constraint(IdentityConstraint(
            "thermal_safe", lambda s: s["temperature"] < 60.0))
        b.add_trigger_constraint(TriggerConstraint(
            "low_battery", "charge_low", lambda s: s["charge"] < 0.2,
            "return_to_dock", priority=3))
        b.add_goal_constraint(GoalConstraint(
            "endurance", lambda s: s["charge"], direction="max"))
        b.add_daemon(HomeostasisDaemon("temperature", 25.0, gain=0.5))
        b.add_daemon(RateDaemon("temperature", max_rate=2.0))
        attach_standard_daemons(b)
        b.add_neural(MLPComponent("rul_predictor", [3, 12, 1], seed=22), name="remaining_useful_life")
        return b

    # ---- level 2 : robots ------------------------------------------------
    def robot(name: str, seed: int) -> COHObject:
        r = COHObject(name, description="autonomous mobile manipulator")
        r.add_attribute(_bounded("task_id", 0.0, 0.0, 1e6))
        r.add_attribute(_bounded("efficiency", 1.0, 0.0, 2.0))
        r.add_child(navigation(f"{name}.nav"))
        r.add_child(gripper(f"{name}.grip"))
        r.add_child(battery(f"{name}.bat"))
        r.add_method(Method(
            "perform_pick",
            lambda s, a=None: (_merge(s, efficiency=s["efficiency"] * 0.99), 2.0),
            description="execute one pick cycle"))
        r.add_identity_constraint(IdentityConstraint(
            "energy_positive", lambda s: r.C[f"{name}.bat"].state["charge"] > 0.0))
        r.add_goal_constraint(GoalConstraint(
            "productivity", lambda s: s["efficiency"], direction="max"))
        r.add_trigger_constraint(TriggerConstraint(
            "fault", "hardware_fault", lambda s: s["efficiency"] < 0.5,
            "handoff_to_peer", priority=5))
        attach_standard_daemons(r)
        r.add_neural(PolicyComponent("policy", input_dim=4, n_actions=4, seed=seed),
                     name="pick_policy")
        r.add_neural(ValueComponent("V", input_dim=4, seed=seed + 1), name="value")
        r.set_embedder(FeatureEmbedder(["efficiency"]))
        return r

    # ---- level 1 : zones -------------------------------------------------
    def zone(name: str, robot_names: List[str]) -> COHObject:
        z = COHObject(name, description=f"{name} operations")
        z.add_attribute(_bounded("throughput", 0.0, 0.0, 1000.0, "items/hour"))
        z.add_attribute(_bounded("backlog", 0.0, 0.0, 1000.0))
        z.add_attribute(_bounded("utilization", 0.0, 0.0, 1.0))
        for i, rn in enumerate(robot_names):
            z.add_child(robot(rn, seed=100 + i))
        z.add_method(Method(
            "dispatch",
            lambda s, a=None: (_merge(s, throughput=s["throughput"] + 10.0), 0.0),
            description="assign next task"))
        z.add_identity_constraint(IdentityConstraint(
            "utilization_valid", lambda s: 0.0 <= s["utilization"] <= 1.0))
        z.add_goal_constraint(GoalConstraint(
            "clear_backlog", lambda s: -s["backlog"], direction="max", weight=2.0))
        z.add_goal_constraint(GoalConstraint(
            "throughput", lambda s: s["throughput"], direction="max"))
        z.add_trigger_constraint(TriggerConstraint(
            "overload", "backlog_high", lambda s: s["backlog"] > 80.0,
            "add_robot", priority=2))
        attach_standard_daemons(z)
        z.add_neural(MLPComponent("forecast", [3, 16, 1], seed=hash(name) & 0xFF),
                     name="demand_forecast")
        z.set_embedder(FeatureEmbedder(["throughput", "backlog", "utilization"]))
        return z

    # ---- level 0 : fleet -------------------------------------------------
    fleet = COHObject("warehouse_fleet", description="autonomous warehouse")
    fleet.add_attribute(_bounded("total_throughput", 0.0, 0.0, 1e4))
    fleet.add_attribute(_bounded("fleet_health", 1.0, 0.0, 1.0))
    fleet.add_attribute(_bounded("energy_cost", 0.0, 0.0, 1e4))
    fleet.add_child(zone("picking_zone", ["R2", "R7"]))
    fleet.add_child(zone("packing_zone", ["P1", "P4"]))
    fleet.add_child(zone("shipping_zone", ["S3"]))

    fleet.add_method(Method(
        "rebalance", lambda s, a=None: (_merge(s, fleet_health=min(1.0, s["fleet_health"] + 0.02)), 1.0),
        description="redistribute load across zones"))

    fleet.add_identity_constraint(IdentityConstraint(
        "health_bounded", lambda s: 0.0 <= s["fleet_health"] <= 1.0))
    fleet.add_identity_constraint(IdentityConstraint(
        "energy_budget", lambda s: s["energy_cost"] < 1000.0))

    fleet.add_trigger_constraint(TriggerConstraint(
        "global_slowdown", "throughput_drop", lambda s: s["total_throughput"] < 10.0,
        "rebalance", priority=3))
    fleet.add_trigger_constraint(TriggerConstraint(
        "energy_spike", "cost_high", lambda s: s["energy_cost"] > 800.0,
        "shed_load", priority=4))

    fleet.add_goal_constraint(GoalConstraint(
        "max_throughput", lambda s: s["total_throughput"], direction="max", weight=3.0))
    fleet.add_goal_constraint(GoalConstraint(
        "min_energy", lambda s: s["energy_cost"], direction="min", weight=1.0))
    fleet.add_goal_constraint(GoalConstraint(
        "max_health", lambda s: s["fleet_health"], direction="max"))

    attach_standard_daemons(fleet)
    fleet.add_neural(MLPComponent("fleet_planner", [3, 32, 4], seed=7), name="fleet_planner")
    fleet.set_embedder(FeatureEmbedder(
        ["total_throughput", "fleet_health", "energy_cost"],
        ranges={"total_throughput": (0, 1e4), "energy_cost": (0, 1e4)}))

    # continuous flow at fleet level: throughput decays, backlog grows
    fleet.flow = ContinuousFlow(
        lambda s, a, th: {
            "total_throughput": -0.05 * s["total_throughput"] + 0.2,
            "energy_cost": 0.5,
        },
        ["total_throughput", "energy_cost"])
    fleet.jump = DiscreteTransition(lambda s, a, th: s)
    return fleet


# =============================================================================
# SYSTEM 2 — Personalised Healthcare Assistant
# =============================================================================
def build_health_assistant() -> COHObject:
    """Hierarchy:

        PatientAssistant             (level 0)
         |-- CardiovascularMonitor   (level 1)
         |    |-- HeartRateSensor    (level 2)
         |    |    |-- SignalFilter  (level 3)
         |    |-- BloodPressureSensor
         |-- MedicationManager
         |    |-- PillDispenser
         |    |-- AdherenceTracker
         |-- LifestyleCoach
              |-- SleepAdvisor
              |-- ActivityAdvisor
    """

    # ---- level 3 : signal processing -------------------------------------
    def signal_filter(name: str, seed: int) -> COHObject:
        f = COHObject(name, description="noise-reduction + feature extraction")
        f.add_attribute(_bounded("snr", 20.0, 0.0, 60.0, "signal-to-noise ratio (dB)"))
        f.add_attribute(_bounded("baseline", 0.0, -50.0, 250.0))
        f.add_attribute(_bounded("noise", 1.0, 0.0, 10.0))
        f.add_method(Method(
            "filter", lambda s, a=None: (_merge(s, noise=s["noise"] * 0.9), 0.0)))
        f.add_identity_constraint(IdentityConstraint("snr_ok", lambda s: s["snr"] > 5.0))
        f.add_goal_constraint(GoalConstraint("clean", lambda s: s["snr"], direction="max"))
        attach_standard_daemons(f)
        f.add_neural(MLPComponent("denoiser", [3, 8, 1], seed=seed), name="denoiser")
        return f

    # ---- level 2 : sensors -----------------------------------------------
    def heart_rate(name: str) -> COHObject:
        h = COHObject(name, description="photoplethysmography heart-rate sensor")
        h.add_attribute(_bounded("bpm", 70.0, 30.0, 220.0))
        h.add_attribute(_bounded("hrv", 40.0, 0.0, 200.0, "heart-rate variability (ms)"))
        h.add_child(signal_filter(f"{name}.filter", seed=201))
        h.add_method(Method("measure", lambda s, a=None: (s, 0.0)))
        h.add_identity_constraint(IdentityConstraint("bpm_range", lambda s: 30.0 <= s["bpm"] <= 220.0))
        h.add_trigger_constraint(TriggerConstraint(
            "tachycardia", "hr_high", lambda s: s["bpm"] > 150.0,
            "alert_emergency", priority=5))
        h.add_trigger_constraint(TriggerConstraint(
            "bradycardia", "hr_low", lambda s: s["bpm"] < 45.0,
            "alert_clinician", priority=5))
        h.add_goal_constraint(GoalConstraint("stable_hr", lambda s: -abs(s["bpm"] - 70), direction="max"))
        attach_standard_daemons(h)
        h.add_neural(PredictorComponent("arrhythmia", 3, 1, seed=202), name="arrhythmia_model")
        return h

    def blood_pressure(name: str) -> COHObject:
        b = COHObject(name, description="cuffless BP estimator")
        b.add_attribute(_bounded("systolic", 120.0, 60.0, 220.0))
        b.add_attribute(_bounded("diastolic", 80.0, 30.0, 140.0))
        b.add_child(signal_filter(f"{name}.filter", seed=203))
        b.add_method(Method("measure", lambda s, a=None: (s, 0.0)))
        b.add_identity_constraint(IdentityConstraint("systolic_gt_diastolic",
                                                     lambda s: s["systolic"] > s["diastolic"]))
        b.add_trigger_constraint(TriggerConstraint(
            "hypertensive_crisis", "bp_high",
            lambda s: s["systolic"] > 180.0 or s["diastolic"] > 120.0,
            "alert_emergency", priority=6))
        b.add_goal_constraint(GoalConstraint(
            "target_bp", lambda s: -abs(s["systolic"] - 120) - abs(s["diastolic"] - 80),
            direction="max"))
        attach_standard_daemons(b)
        b.add_neural(MLPComponent("bp_regressor", [3, 16, 2], seed=204), name="bp_model")
        return b

    def pill_dispenser(name: str) -> COHObject:
        d = COHObject(name, description="automated medication dispenser")
        d.add_attribute(_bounded("pills_remaining", 30.0, 0.0, 200.0))
        d.add_attribute(_bounded("last_dose_hours", 0.0, 0.0, 72.0))
        d.add_attribute(_bounded("adherence", 1.0, 0.0, 1.0))
        d.add_method(Method("dispense", lambda s, a=None: (_merge(
            s, pills_remaining=max(0.0, s["pills_remaining"] - 1), last_dose_hours=0.0), 1.0)))
        d.add_identity_constraint(IdentityConstraint("no_overdose", lambda s: s["last_dose_hours"] >= 4.0))
        d.add_trigger_constraint(TriggerConstraint(
            "refill_needed", "pills_low", lambda s: s["pills_remaining"] < 5.0,
            "order_refill", priority=2))
        d.add_trigger_constraint(TriggerConstraint(
            "missed_dose", "dose_late", lambda s: s["last_dose_hours"] > 12.0,
            "remind_patient", priority=3))
        d.add_goal_constraint(GoalConstraint("max_adherence", lambda s: s["adherence"], direction="max"))
        attach_standard_daemons(d)
        d.add_neural(MLPComponent("schedule_model", [3, 8, 1], seed=205), name="schedule")
        return d

    def adherence_tracker(name: str) -> COHObject:
        t = COHObject(name, description="adherence score")
        t.add_attribute(_bounded("adherence_30d", 1.0, 0.0, 1.0))
        t.add_attribute(_bounded("missed_doses", 0.0, 0.0, 100.0))
        t.add_method(Method("log_dose", lambda s, a=None: (_merge(s, adherence_30d=min(1.0, s["adherence_30d"] + 0.01)), 0.0)))
        t.add_goal_constraint(GoalConstraint("improve_adherence", lambda s: s["adherence_30d"], direction="max"))
        attach_standard_daemons(t)
        return t

    def sleep_advisor(name: str) -> COHObject:
        s = COHObject(name, description="sleep quality advisor")
        s.add_attribute(_bounded("sleep_hours", 7.0, 0.0, 14.0))
        s.add_attribute(_bounded("rem_fraction", 0.22, 0.0, 0.5))
        s.add_attribute(_bounded("sleep_score", 75.0, 0.0, 100.0))
        s.add_method(Method("recommend", lambda s, a=None: (_merge(s, sleep_score=min(100, s["sleep_score"] + 1)), 0.0)))
        s.add_identity_constraint(IdentityConstraint("hours_plausible", lambda s: 0.0 <= s["sleep_hours"] <= 14.0))
        s.add_goal_constraint(GoalConstraint("sleep_target", lambda s: s["sleep_score"], direction="max"))
        attach_standard_daemons(s)
        s.add_neural(MLPComponent("sleep_predictor", [3, 12, 1], seed=206), name="sleep_model")
        return s

    def activity_advisor(name: str) -> COHObject:
        a = COHObject(name, description="physical-activity advisor")
        a.add_attribute(_bounded("steps_today", 0.0, 0.0, 50000.0))
        a.add_attribute(_bounded("active_minutes", 0.0, 0.0, 300.0))
        a.add_method(Method("suggest_walk", lambda s, a=None: (_merge(s, steps_today=s["steps_today"] + 500), 0.0)))
        a.add_goal_constraint(GoalConstraint("reach_10k", lambda s: min(s["steps_today"], 10000), direction="max"))
        attach_standard_daemons(a)
        return a

    # ---- level 1 : sub-systems ------------------------------------------
    cardio = COHObject("cardiovascular_monitor", description="continuous cardiac telemetry")
    cardio.add_attribute(_bounded("risk_score", 0.1, 0.0, 1.0))
    cardio.add_attribute(_bounded("alert_count", 0.0, 0.0, 100.0))
    cardio.add_child(heart_rate("hr_sensor"))
    cardio.add_child(blood_pressure("bp_sensor"))
    cardio.add_method(Method("assess", lambda s, a=None: (_merge(s, risk_score=min(1.0, s["risk_score"] + 0.01)), 0.0)))
    cardio.add_identity_constraint(IdentityConstraint("risk_bounded", lambda s: 0.0 <= s["risk_score"] <= 1.0))
    cardio.add_goal_constraint(GoalConstraint("low_risk", lambda s: -s["risk_score"], direction="max"))
    attach_standard_daemons(cardio)
    cardio.add_neural(MLPComponent("risk_model", [4, 16, 1], seed=210), name="risk_model")

    med = COHObject("medication_manager", description="prescription & adherence")
    med.add_attribute(_bounded("active_prescriptions", 1.0, 0.0, 20.0))
    med.add_attribute(_bounded("interaction_risk", 0.0, 0.0, 1.0))
    med.add_child(pill_dispenser("dispenser"))
    med.add_child(adherence_tracker("adherence"))
    med.add_method(Method("reconcile", lambda s, a=None: (_merge(s, interaction_risk=max(0.0, s["interaction_risk"] - 0.01)), 0.0)))
    med.add_identity_constraint(IdentityConstraint("interaction_safe", lambda s: s["interaction_risk"] < 0.5))
    med.add_goal_constraint(GoalConstraint("adherence", lambda s: -s["interaction_risk"], direction="max"))
    attach_standard_daemons(med)
    med.add_neural(MLPComponent("interaction_model", [3, 12, 1], seed=211), name="interaction_model")

    life = COHObject("lifestyle_coach", description="behaviour change support")
    life.add_attribute(_bounded("wellness_score", 0.7, 0.0, 1.0))
    life.add_child(sleep_advisor("sleep"))
    life.add_child(activity_advisor("activity"))
    life.add_method(Method("coach", lambda s, a=None: (_merge(s, wellness_score=min(1.0, s["wellness_score"] + 0.005)), 0.0)))
    life.add_goal_constraint(GoalConstraint("wellness", lambda s: s["wellness_score"], direction="max"))
    attach_standard_daemons(life)

    # ---- level 0 : assistant --------------------------------------------
    pa = COHObject("patient_assistant", description="personalised healthcare companion")
    pa.add_attribute(_bounded("patient_id", 0.0, 0.0, 1e9))
    pa.add_attribute(_bounded("overall_health", 0.85, 0.0, 1.0))
    pa.add_attribute(_bounded("days_since_visit", 0.0, 0.0, 3650.0))
    pa.add_attribute(_bounded("consent_given", 1.0, 0.0, 1.0))
    pa.add_child(cardio)
    pa.add_child(med)
    pa.add_child(life)

    pa.add_method(Method(
        "daily_checkup",
        lambda s, a=None: (_merge(s, days_since_visit=s["days_since_visit"] + 1,
                                  overall_health=max(0.0, s["overall_health"] - 0.001)), 0.5)))

    pa.add_identity_constraint(IdentityConstraint("consent_required", lambda s: s["consent_given"] >= 1.0,
                                                  description="only operate with patient consent"))
    pa.add_identity_constraint(IdentityConstraint("health_bounded", lambda s: 0.0 <= s["overall_health"] <= 1.0))
    pa.add_trigger_constraint(TriggerConstraint(
        "annual_visit", "year_elapsed", lambda s: s["days_since_visit"] > 365.0,
        "schedule_checkup", priority=1))
    pa.add_trigger_constraint(TriggerConstraint(
        "consent_withdrawn", "consent_change", lambda s: s["consent_given"] < 1.0,
        "halt_monitoring", priority=10))
    pa.add_goal_constraint(GoalConstraint("max_health", lambda s: s["overall_health"], direction="max", weight=3.0))
    pa.add_goal_constraint(GoalConstraint("min_visits", lambda s: s["days_since_visit"], direction="min"))

    attach_standard_daemons(pa)
    pa.add_neural(MLPComponent("wellness_policy", [5, 32, 4], seed=220), name="wellness_policy")
    pa.set_embedder(FeatureEmbedder(["overall_health", "days_since_visit"],
                                    ranges={"overall_health": (0, 1), "days_since_visit": (0, 3650)}))
    pa.flow = ContinuousFlow(
        lambda s, a, th: {"overall_health": -0.0005, "days_since_visit": 1.0},
        ["overall_health", "days_since_visit"])
    return pa


# =============================================================================
# SYSTEM 3 — Smart Grid Energy Management
# =============================================================================
def build_smart_grid() -> COHObject:
    """Hierarchy:

        GridAuthority             (level 0)
         |-- NorthernRegion       (level 1)
         |    |-- MicrogridA      (level 2)
         |         |-- SolarFarm  (level 3)
         |         |-- WindFarm   (level 3)
         |         |-- Storage    (level 3)
         |-- SouthernRegion
              |-- MicrogridB
                   |-- SolarFarm
                   |-- Storage
    """

    def solar(name: str, capacity: float, seed: int) -> COHObject:
        s = COHObject(name, description="photovoltaic generation")
        s.add_attribute(_bounded("irradiance", 0.7, 0.0, 1.0, "normalised solar irradiance"))
        s.add_attribute(_bounded("output_mw", capacity * 0.6, 0.0, capacity))
        s.add_attribute(_bounded("panel_temp", 25.0, -20.0, 90.0))
        s.add_attribute(Attribute("capacity_mw", capacity))
        s.add_method(Method(
            "generate",
            lambda st, a=None: (_merge(st, output_mw=min(st["capacity_mw"], st["capacity_mw"] * st["irradiance"])), 0.0),
            description="produce power from irradiance"))
        s.add_identity_constraint(IdentityConstraint("output_valid", lambda st: 0.0 <= st["output_mw"] <= st["capacity_mw"]))
        s.add_identity_constraint(IdentityConstraint("thermal", lambda st: st["panel_temp"] < 85.0))
        s.add_trigger_constraint(TriggerConstraint(
            "overheat", "panel_hot", lambda st: st["panel_temp"] > 75.0,
            "reduce_output", priority=3))
        s.add_goal_constraint(GoalConstraint("max_output", lambda st: st["output_mw"], direction="max"))
        attach_standard_daemons(s)
        s.add_neural(PredictorComponent("irradiance_forecast", 3, 1, seed=seed), name="irradiance_model")
        return s

    def wind(name: str, capacity: float, seed: int) -> COHObject:
        w = COHObject(name, description="wind turbine array")
        w.add_attribute(_bounded("wind_speed", 8.0, 0.0, 40.0, "m/s"))
        w.add_attribute(_bounded("output_mw", capacity * 0.4, 0.0, capacity))
        w.add_attribute(Attribute("capacity_mw", capacity))
        w.add_method(Method(
            "generate",
            lambda st, a=None: (_merge(st, output_mw=min(st["capacity_mw"], st["capacity_mw"] * (st["wind_speed"] / 25.0))), 0.0)))
        w.add_identity_constraint(IdentityConstraint("cutout", lambda st: st["wind_speed"] <= 25.0))
        w.add_trigger_constraint(TriggerConstraint(
            "storm_shutdown", "high_wind", lambda st: st["wind_speed"] > 22.0,
            "feather_blades", priority=5))
        w.add_goal_constraint(GoalConstraint("max_output", lambda st: st["output_mw"], direction="max"))
        attach_standard_daemons(w)
        w.add_neural(PredictorComponent("wind_forecast", 3, 1, seed=seed), name="wind_model")
        return w

    def storage(name: str, capacity: float) -> COHObject:
        b = COHObject(name, description="grid-scale battery")
        b.add_attribute(_bounded("soc", 0.5, 0.0, 1.0, "state of charge"))
        b.add_attribute(Attribute("capacity_mwh", capacity))
        b.add_attribute(_bounded("power_mw", 0.0, -capacity, capacity))
        b.add_method(Method("charge", lambda st, a=None: (_merge(st, soc=min(1.0, st["soc"] + 0.02), power_mw=+capacity * 0.1), 0.0)))
        b.add_method(Method("discharge", lambda st, a=None: (_merge(st, soc=max(0.0, st["soc"] - 0.02), power_mw=-capacity * 0.1), 0.0)))
        b.add_identity_constraint(IdentityConstraint("soc_bounds", lambda st: 0.0 <= st["soc"] <= 1.0))
        b.add_trigger_constraint(TriggerConstraint(
            "deep_discharge", "soc_low", lambda st: st["soc"] < 0.15,
            "force_charge", priority=4))
        b.add_goal_constraint(GoalConstraint("hold_soc", lambda st: -abs(st["soc"] - 0.5), direction="max"))
        attach_standard_daemons(b)
        b.add_neural(MLPComponent("state_estimator", [3, 8, 1], seed=301), name="soc_estimator")
        return b

    def microgrid(name: str, solar_cap: float, wind_cap: float, storage_cap: float, seed: int) -> COHObject:
        m = COHObject(name, description="local balance area")
        m.add_attribute(_bounded("local_demand_mw", solar_cap * 0.3, 0.0, solar_cap + wind_cap))
        m.add_attribute(_bounded("balance_mw", 0.0, -(solar_cap + wind_cap), solar_cap + wind_cap))
        m.add_attribute(_bounded("frequency_hz", 50.0, 48.0, 52.0))
        m.add_child(solar(f"{name}.solar", solar_cap, seed))
        if wind_cap > 0:
            m.add_child(wind(f"{name}.wind", wind_cap, seed + 1))
        m.add_child(storage(f"{name}.storage", storage_cap))
        m.add_method(Method(
            "balance",
            lambda st, a=None: (_merge(st, balance_mw=st["balance_mw"] * 0.9,
                                       frequency_hz=50.0 + 0.01 * st["balance_mw"]), 0.0),
            description="activate storage to balance local supply/demand"))
        m.add_identity_constraint(IdentityConstraint(
            "frequency_window", lambda st: 49.5 <= st["frequency_hz"] <= 50.5,
            description="stay within statutory frequency band"))
        m.add_trigger_constraint(TriggerConstraint(
            "under_frequency", "freq_low", lambda st: st["frequency_hz"] < 49.8,
            "discharge_storage", priority=5))
        m.add_trigger_constraint(TriggerConstraint(
            "over_frequency", "freq_high", lambda st: st["frequency_hz"] > 50.2,
            "charge_storage", priority=5))
        m.add_goal_constraint(GoalConstraint("stability", lambda st: -abs(st["frequency_hz"] - 50.0), direction="max", weight=3.0))
        m.add_goal_constraint(GoalConstraint("balance", lambda st: -abs(st["balance_mw"]), direction="max"))
        attach_standard_daemons(m)
        m.add_neural(MLPComponent("frequency_predictor", [3, 16, 1], seed=seed + 5), name="freq_model")
        m.set_embedder(FeatureEmbedder(["local_demand_mw", "frequency_hz"]))
        m.flow = ContinuousFlow(
            lambda st, a, th: {"frequency_hz": -0.1 * (st["frequency_hz"] - 50.0)},
            ["frequency_hz"])
        return m

    def region(name: str, microgrids: List[str], base_load: float) -> COHObject:
        r = COHObject(name, description="regional balancing authority")
        r.add_attribute(_bounded("regional_demand_mw", base_load, 0.0, 5000.0))
        r.add_attribute(_bounded("reserve_margin", 0.2, 0.0, 1.0))
        r.add_attribute(_bounded("renewable_fraction", 0.5, 0.0, 1.0))
        for mg_name, sc, wc, stc, seed in microgrids:
            r.add_child(microgrid(mg_name, sc, wc, stc, seed))
        r.add_method(Method(
            "optimise_dispatch",
            lambda st, a=None: (_merge(st, reserve_margin=min(1.0, st["reserve_margin"] + 0.01)), 0.0)))
        r.add_identity_constraint(IdentityConstraint(
            "reserve_ok", lambda st: st["reserve_margin"] >= 0.1,
            description="keep operating reserve above N-1 criterion"))
        r.add_trigger_constraint(TriggerConstraint(
            "reserve_low", "reserve_alert", lambda st: st["reserve_margin"] < 0.15,
            "call_peaker_plant", priority=4))
        r.add_goal_constraint(GoalConstraint("high_renewables", lambda st: st["renewable_fraction"], direction="max"))
        r.add_goal_constraint(GoalConstraint("reserve_margin", lambda st: st["reserve_margin"], direction="max"))
        attach_standard_daemons(r)
        r.add_neural(MLPComponent("load_forecast", [4, 32, 1], seed=310), name="load_forecast")
        return r

    grid = COHObject("grid_authority", description="national transmission operator")
    grid.add_attribute(_bounded("national_demand_mw", 0.0, 0.0, 2e4))
    grid.add_attribute(_bounded("renewable_share", 0.4, 0.0, 1.0))
    grid.add_attribute(_bounded("co2_intensity", 300.0, 0.0, 800.0, "gCO2/kWh"))
    grid.add_attribute(_bounded("system_inertia", 6.0, 2.0, 12.0, "seconds"))
    grid.add_child(region("northern_region", [
        ("microgrid_north_a", 200.0, 150.0, 100.0, 401),
        ("microgrid_north_b", 120.0, 0.0, 60.0, 402),
    ], base_load=800.0))
    grid.add_child(region("southern_region", [
        ("microgrid_south_a", 300.0, 0.0, 150.0, 403),
    ], base_load=600.0))
    grid.add_method(Method(
        "balancing_action",
        lambda s, a=None: (_merge(s, renewable_share=min(1.0, s["renewable_share"] + 0.005),
                                  co2_intensity=max(0.0, s["co2_intensity"] - 0.5)), 0.0)))
    grid.add_identity_constraint(IdentityConstraint(
        "co2_budget", lambda s: s["co2_intensity"] < 500.0,
        description="national carbon budget"))
    grid.add_identity_constraint(IdentityConstraint(
        "inertia_floor", lambda s: s["system_inertia"] >= 3.0,
        description="minimum system inertia"))
    grid.add_trigger_constraint(TriggerConstraint(
        "carbon_alert", "co2_high", lambda s: s["co2_intensity"] > 450.0,
        "curtail_coal", priority=6))
    grid.add_trigger_constraint(TriggerConstraint(
        "inertia_alert", "inertia_low", lambda s: s["system_inertia"] < 4.0,
        "activate_synchronous_condenser", priority=7))
    grid.add_goal_constraint(GoalConstraint("renewables", lambda s: s["renewable_share"], direction="max", weight=2.0))
    grid.add_goal_constraint(GoalConstraint("co2_min", lambda s: s["co2_intensity"], direction="min", weight=2.0))
    grid.add_goal_constraint(GoalConstraint("inertia", lambda s: s["system_inertia"], direction="max"))
    attach_standard_daemons(grid)
    grid.add_neural(MLPComponent("dispatch_planner", [4, 64, 8], seed=450), name="dispatch_planner")
    grid.set_embedder(FeatureEmbedder(["national_demand_mw", "renewable_share", "co2_intensity"]))
    return grid


# =============================================================================
# SYSTEM 4 — Adaptive E-Learning Tutor
# =============================================================================
def build_elearning_tutor() -> COHObject:
    """Hierarchy:

        TutorPlatform              (level 0)
         |-- MathematicsTrack      (level 1)
         |    |-- AlgebraModule     (level 2)
         |    |    |-- LinearEqnsActivity (level 3)
         |    |    |-- QuadraticsActivity
         |    |-- GeometryModule
         |-- ScienceTrack
         |    |-- PhysicsModule
         |-- LanguageTrack
              |-- GrammarModule
    """

    def activity(name: str, kind: str, seed: int) -> COHObject:
        a = COHObject(name, description=f"{kind} learning activity")
        a.add_attribute(_bounded("mastery", 0.0, 0.0, 1.0))
        a.add_attribute(_bounded("difficulty", 0.5, 0.0, 1.0))
        a.add_attribute(_bounded("time_spent_min", 0.0, 0.0, 240.0))
        a.add_attribute(_bounded("attempts", 0.0, 0.0, 100.0))
        a.add_method(Method(
            "attempt",
            lambda s, a=None: (_merge(s,
                                      attempts=s["attempts"] + 1,
                                      mastery=min(1.0, s["mastery"] + 0.05 * (1 - s["difficulty"]))), 1.0)))
        a.add_method(Method(
            "hint",
            lambda s, a=None: (_merge(s, time_spent_min=s["time_spent_min"] + 2), 0.5)))
        a.add_identity_constraint(IdentityConstraint("mastery_bounded", lambda s: 0.0 <= s["mastery"] <= 1.0))
        a.add_trigger_constraint(TriggerConstraint(
            "frustration", "many_attempts", lambda s: s["attempts"] > 8.0 and s["mastery"] < 0.3,
            "offer_hint", priority=2))
        a.add_trigger_constraint(TriggerConstraint(
            "boredom", "mastered", lambda s: s["mastery"] > 0.95,
            "escalate_difficulty", priority=1))
        a.add_goal_constraint(GoalConstraint("mastery_goal", lambda s: s["mastery"], direction="max", weight=2.0))
        a.add_goal_constraint(GoalConstraint("efficiency", lambda s: -s["time_spent_min"], direction="max"))
        attach_standard_daemons(a)
        a.add_neural(MLPComponent("difficulty_tuner", [3, 8, 1], seed=seed), name="difficulty_tuner")
        return a

    def module(name: str, activities: List[str], seed: int) -> COHObject:
        m = COHObject(name, description="group of related activities")
        m.add_attribute(_bounded("module_mastery", 0.0, 0.0, 1.0))
        m.add_attribute(_bounded("completion", 0.0, 0.0, 1.0))
        m.add_attribute(_bounded("engagement", 0.7, 0.0, 1.0))
        for i, (aname, kind) in enumerate(activities):
            m.add_child(activity(f"{name}.{aname}", kind, seed + i))
        m.add_method(Method(
            "progress",
            lambda s, a=None: (_merge(s, completion=min(1.0, s["completion"] + 0.05),
                                      module_mastery=min(1.0, s["module_mastery"] + 0.03)), 0.0)))
        m.add_identity_constraint(IdentityConstraint("completion_bounded", lambda s: 0.0 <= s["completion"] <= 1.0))
        m.add_trigger_constraint(TriggerConstraint(
            "module_complete", "completion_high", lambda s: s["completion"] > 0.9,
            "recommend_next_module", priority=1))
        m.add_goal_constraint(GoalConstraint("mastery", lambda s: s["module_mastery"], direction="max"))
        attach_standard_daemons(m)
        m.add_neural(MLPComponent("recommender", [3, 16, 4], seed=seed + 50), name="recommender")
        return m

    def track(name: str, modules: List[Tuple], seed: int) -> COHObject:
        t = COHObject(name, description="subject-matter track")
        t.add_attribute(_bounded("track_progress", 0.0, 0.0, 1.0))
        t.add_attribute(_bounded("prerequisite_met", 1.0, 0.0, 1.0))
        for i, (mname, acts) in enumerate(modules):
            t.add_child(module(mname, acts, seed + 10 * i))
        t.add_method(Method(
            "advance", lambda s, a=None: (_merge(s, track_progress=min(1.0, s["track_progress"] + 0.02)), 0.0)))
        t.add_identity_constraint(IdentityConstraint("prereq_ok", lambda s: s["prerequisite_met"] >= 1.0,
                                                     description="prerequisites must be satisfied"))
        t.add_trigger_constraint(TriggerConstraint(
            "prereq_missing", "prereq_check", lambda s: s["prerequisite_met"] < 0.5,
            "remedial_module", priority=3))
        t.add_goal_constraint(GoalConstraint("track_mastery", lambda s: s["track_progress"], direction="max"))
        attach_standard_daemons(t)
        t.add_neural(MLPComponent("track_planner", [3, 32, 4], seed=seed + 100), name="track_planner")
        return t

    tutor = COHObject("tutor_platform", description="adaptive intelligent tutoring system")
    tutor.add_attribute(_bounded("learner_id", 0.0, 0.0, 1e9))
    tutor.add_attribute(_bounded("session_time_min", 0.0, 0.0, 600.0))
    tutor.add_attribute(_bounded("overall_mastery", 0.0, 0.0, 1.0))
    tutor.add_attribute(_bounded("affect_valence", 0.5, -1.0, 1.0, "self-reported mood"))
    tutor.add_attribute(_bounded("cognitive_load", 0.5, 0.0, 1.0))
    tutor.add_child(track("mathematics_track", [
        ("algebra_module", [("linear_equations", "skill"), ("quadratics", "skill")]),
        ("geometry_module", [("triangles", "concept"), ("circles", "concept")]),
    ], seed=500))
    tutor.add_child(track("science_track", [
        ("physics_module", [("kinematics", "skill"), ("forces", "skill")]),
        ("chemistry_module", [("stoichiometry", "skill")]),
    ], seed=530))
    tutor.add_child(track("language_track", [
        ("grammar_module", [("syntax", "skill"), ("morphology", "concept")]),
    ], seed=560))
    tutor.add_method(Method(
        "session_step",
        lambda s, a=None: (_merge(s,
                                  session_time_min=s["session_time_min"] + 5,
                                  overall_mastery=min(1.0, s["overall_mastery"] + 0.01)), 0.5)))
    tutor.add_identity_constraint(IdentityConstraint(
        "load_bounded", lambda s: 0.0 <= s["cognitive_load"] <= 1.0,
        description="never overwhelm the learner"))
    tutor.add_identity_constraint(IdentityConstraint(
        "session_bounded", lambda s: s["session_time_min"] <= 480.0))
    tutor.add_trigger_constraint(TriggerConstraint(
        "fatigue", "session_long", lambda s: s["session_time_min"] > 90.0,
        "suggest_break", priority=2))
    tutor.add_trigger_constraint(TriggerConstraint(
        "negative_affect", "mood_low", lambda s: s["affect_valence"] < -0.3,
        "switch_activity", priority=3))
    tutor.add_trigger_constraint(TriggerConstraint(
        "overload", "load_high", lambda s: s["cognitive_load"] > 0.85,
        "reduce_difficulty", priority=4))
    tutor.add_goal_constraint(GoalConstraint("max_mastery", lambda s: s["overall_mastery"], direction="max", weight=3.0))
    tutor.add_goal_constraint(GoalConstraint("positive_affect", lambda s: s["affect_valence"], direction="max"))
    tutor.add_goal_constraint(GoalConstraint("min_load", lambda s: s["cognitive_load"], direction="min"))
    attach_standard_daemons(tutor)
    tutor.add_neural(MLPComponent("policy", [5, 64, 6], seed=600), name="pedagogy_policy")
    tutor.add_neural(MLPComponent("knowledge_tracer", [5, 32, 1], seed=601), name="knowledge_tracer")
    tutor.set_embedder(FeatureEmbedder(["overall_mastery", "affect_valence", "cognitive_load"],
                                       ranges={"overall_mastery": (0, 1), "affect_valence": (-1, 1),
                                               "cognitive_load": (0, 1)}))
    return tutor


# =============================================================================
# SYSTEM 5 — Autonomous Trading Desk
# =============================================================================
def build_trading_desk() -> COHObject:
    """Hierarchy:

        TradingDesk                 (level 0)
         |-- ExecutionDesk          (level 1)
         |    |-- MarketMaker        (level 2)
         |    |    |-- QuoteEngine   (level 3)
         |    |    |-- InventoryManager (level 3)
         |    |-- Arbitrageur        (level 2)
         |-- RiskOffice             (level 1)
         |    |-- ValueAtRisk        (level 2)
         |    |-- LiquidityMonitor   (level 2)
         |-- ResearchDesk           (level 1)
              |-- SignalGenerator    (level 2)
    """

    # ---- level 3 : execution sub-systems --------------------------------
    def quote_engine(name: str) -> COHObject:
        q = COHObject(name, description="bid/ask quoting logic")
        q.add_attribute(_bounded("spread_bps", 5.0, 0.5, 100.0))
        q.add_attribute(_bounded("quote_size", 100.0, 0.0, 10000.0))
        q.add_attribute(_bounded("fill_ratio", 0.3, 0.0, 1.0))
        q.add_method(Method(
            "update_quote",
            lambda s, a=None: (_merge(s, spread_bps=max(0.5, s["spread_bps"] * 0.99)), 0.0)))
        q.add_identity_constraint(IdentityConstraint("spread_sane", lambda s: 0.5 <= s["spread_bps"] <= 100.0))
        q.add_goal_constraint(GoalConstraint("max_fill", lambda s: s["fill_ratio"], direction="max", weight=2.0))
        q.add_goal_constraint(GoalConstraint("min_spread", lambda s: s["spread_bps"], direction="min"))
        attach_standard_daemons(q)
        q.add_neural(MLPComponent("fill_predictor", [3, 12, 1], seed=701), name="fill_predictor")
        return q

    def inventory_manager(name: str) -> COHObject:
        i = COHObject(name, description="position & skew control")
        i.add_attribute(_bounded("position", 0.0, -10000.0, 10000.0, "shares"))
        i.add_attribute(_bounded("avg_price", 100.0, 0.0, 10000.0))
        i.add_attribute(_bounded("skew_bps", 0.0, -50.0, 50.0))
        i.add_method(Method("rebalance", lambda s, a=None: (_merge(s, position=s["position"] * 0.95), 0.0)))
        i.add_identity_constraint(IdentityConstraint("position_limit", lambda s: abs(s["position"]) <= 8000.0))
        i.add_trigger_constraint(TriggerConstraint(
            "position_breach", "position_high", lambda s: abs(s["position"]) > 7000.0,
            "flatten_book", priority=5))
        i.add_goal_constraint(GoalConstraint("inventory_flat", lambda s: -abs(s["position"]), direction="max"))
        attach_standard_daemons(i)
        i.add_neural(MLPComponent("skew_model", [3, 8, 1], seed=702), name="skew_model")
        return i

    def var_engine(name: str) -> COHObject:
        v = COHObject(name, description="value-at-risk estimator")
        v.add_attribute(_bounded("var_95_pct", 1.0, 0.0, 20.0, "95% VaR as % of capital"))
        v.add_attribute(_bounded("expected_shortfall", 1.5, 0.0, 30.0))
        v.add_attribute(_bounded("confidence", 0.95, 0.5, 0.999))
        v.add_method(Method("recompute", lambda s, a=None: (s, 0.0)))
        v.add_identity_constraint(IdentityConstraint("var_cap", lambda s: s["var_95_pct"] < 5.0,
                                                     description="internal risk appetite"))
        v.add_trigger_constraint(TriggerConstraint(
            "var_breach", "var_alert", lambda s: s["var_95_pct"] > 4.0,
            "de_risk_portfolio", priority=6))
        v.add_goal_constraint(GoalConstraint("risk_low", lambda s: -s["var_95_pct"], direction="max"))
        attach_standard_daemons(v)
        v.add_neural(MLPComponent("var_nn", [3, 16, 1], seed=703), name="var_nn")
        return v

    def liquidity_monitor(name: str) -> COHObject:
        l = COHObject(name, description="market microstructure monitoring")
        l.add_attribute(_bounded("depth", 1000.0, 0.0, 1e6))
        l.add_attribute(_bounded("imbalance", 0.0, -1.0, 1.0))
        l.add_attribute(_bounded("toxicity", 0.1, 0.0, 1.0, "adverse selection risk"))
        l.add_method(Method("refresh", lambda s, a=None: (s, 0.0)))
        l.add_identity_constraint(IdentityConstraint("toxicity_bounded", lambda s: 0.0 <= s["toxicity"] <= 1.0))
        l.add_trigger_constraint(TriggerConstraint(
            "toxic_flow", "toxicity_high", lambda s: s["toxicity"] > 0.7,
            "widen_spread", priority=4))
        l.add_goal_constraint(GoalConstraint("low_toxicity", lambda s: -s["toxicity"], direction="max"))
        attach_standard_daemons(l)
        return l

    def signal_generator(name: str) -> COHObject:
        s = COHObject(name, description="alpha signal generation")
        s.add_attribute(_bounded("signal_strength", 0.0, -1.0, 1.0))
        s.add_attribute(_bounded("sharpe_estimate", 0.0, -3.0, 5.0))
        s.add_attribute(_bounded("decay_halflife_days", 5.0, 0.5, 60.0))
        s.add_method(Method("generate", lambda s, a=None: (s, 0.0)))
        s.add_identity_constraint(IdentityConstraint("sharpe_plausible", lambda s: -3.0 <= s["sharpe_estimate"] <= 5.0))
        s.add_goal_constraint(GoalConstraint("high_sharpe", lambda s: s["sharpe_estimate"], direction="max", weight=2.0))
        attach_standard_daemons(s)
        s.add_neural(MLPComponent("alpha_model", [4, 32, 1], seed=704), name="alpha_model")
        return s

    # ---- level 2 : strategies -------------------------------------------
    mm = COHObject("market_maker", description="passive market-making strategy")
    mm.add_attribute(_bounded("pnl", 0.0, -1e6, 1e6))
    mm.add_attribute(_bounded("inventory_turnover", 0.0, 0.0, 100.0))
    mm.add_child(quote_engine("market_maker.quotes"))
    mm.add_child(inventory_manager("market_maker.inventory"))
    mm.add_method(Method(
        "quote_cycle",
        lambda s, a=None: (_merge(s, pnl=s["pnl"] + 5.0, inventory_turnover=s["inventory_turnover"] + 0.1), 1.0)))
    mm.add_identity_constraint(IdentityConstraint("pnl_bounded", lambda s: -1e6 <= s["pnl"] <= 1e6))
    mm.add_goal_constraint(GoalConstraint("max_pnl", lambda s: s["pnl"], direction="max", weight=2.0))
    mm.add_goal_constraint(GoalConstraint("low_turnover", lambda s: -s["inventory_turnover"], direction="max"))
    attach_standard_daemons(mm)
    mm.add_neural(MLPComponent("mm_policy", [3, 16, 4], seed=710), name="mm_policy")

    arb = COHObject("arbitrageur", description="cross-venue arbitrage strategy")
    arb.add_attribute(_bounded("pnl", 0.0, -1e6, 1e6))
    arb.add_attribute(_bounded("latency_us", 100.0, 10.0, 5000.0))
    arb.add_attribute(_bounded("opportunities_per_min", 0.0, 0.0, 1000.0))
    arb.add_child(quote_engine("arbitrageur.quotes"))
    arb.add_method(Method(
        "seek_arb",
        lambda s, a=None: (_merge(s, opportunities_per_min=s["opportunities_per_min"] + 0.1), 0.0)))
    arb.add_identity_constraint(IdentityConstraint("latency_cap", lambda s: s["latency_us"] < 3000.0))
    arb.add_trigger_constraint(TriggerConstraint(
        "latency_spike", "latency_high", lambda s: s["latency_us"] > 2000.0,
        "disable_strategy", priority=7))
    arb.add_goal_constraint(GoalConstraint("max_pnl", lambda s: s["pnl"], direction="max"))
    arb.add_goal_constraint(GoalConstraint("min_latency", lambda s: -s["latency_us"], direction="max"))
    attach_standard_daemons(arb)
    arb.add_neural(MLPComponent("arb_model", [3, 16, 2], seed=711), name="arb_model")

    # ---- level 1 : desks ------------------------------------------------
    exec_desk = COHObject("execution_desk", description="order execution")
    exec_desk.add_attribute(_bounded("total_pnl", 0.0, -1e7, 1e7))
    exec_desk.add_attribute(_bounded("slippage_bps", 2.0, 0.0, 50.0))
    exec_desk.add_attribute(_bounded("order_rate", 0.0, 0.0, 1e5))
    exec_desk.add_child(mm)
    exec_desk.add_child(arb)
    exec_desk.add_method(Method(
        "end_of_day",
        lambda s, a=None: (_merge(s, slippage_bps=max(0.0, s["slippage_bps"] - 0.1)), 0.0)))
    exec_desk.add_identity_constraint(IdentityConstraint("slippage_cap", lambda s: s["slippage_bps"] < 20.0))
    exec_desk.add_trigger_constraint(TriggerConstraint(
        "slippage_alert", "slippage_high", lambda s: s["slippage_bps"] > 10.0,
        "switch_to_passive", priority=3))
    exec_desk.add_goal_constraint(GoalConstraint("max_pnl", lambda s: s["total_pnl"], direction="max", weight=3.0))
    exec_desk.add_goal_constraint(GoalConstraint("min_slippage", lambda s: -s["slippage_bps"], direction="max"))
    attach_standard_daemons(exec_desk)
    exec_desk.add_neural(MLPComponent("exec_policy", [3, 32, 4], seed=720), name="exec_policy")

    risk_office = COHObject("risk_office", description="independent risk oversight")
    risk_office.add_attribute(_bounded("aggregate_var", 1.0, 0.0, 20.0))
    risk_office.add_attribute(_bounded("leverage", 2.0, 0.0, 30.0))
    risk_office.add_attribute(_bounded("stress_loss_pct", 2.0, 0.0, 50.0))
    risk_office.add_child(var_engine("risk_office.var"))
    risk_office.add_child(liquidity_monitor("risk_office.liquidity"))
    risk_office.add_method(Method(
        "daily_report",
        lambda s, a=None: (_merge(s, aggregate_var=max(0.0, s["aggregate_var"] - 0.01)), 0.0)))
    risk_office.add_identity_constraint(IdentityConstraint("leverage_cap", lambda s: s["leverage"] < 15.0))
    risk_office.add_identity_constraint(IdentityConstraint("stress_cap", lambda s: s["stress_loss_pct"] < 20.0))
    risk_office.add_trigger_constraint(TriggerConstraint(
        "leverage_breach", "leverage_high", lambda s: s["leverage"] > 12.0,
        "margin_call", priority=8))
    risk_office.add_trigger_constraint(TriggerConstraint(
        "stress_breach", "stress_high", lambda s: s["stress_loss_pct"] > 15.0,
        "emergency_de_risk", priority=9))
    risk_office.add_goal_constraint(GoalConstraint("low_var", lambda s: -s["aggregate_var"], direction="max"))
    risk_office.add_goal_constraint(GoalConstraint("low_leverage", lambda s: -s["leverage"], direction="max"))
    attach_standard_daemons(risk_office)
    risk_office.add_neural(MLPComponent("stress_model", [4, 32, 4], seed=730), name="stress_model")

    research = COHObject("research_desk", description="systematic signal research")
    research.add_attribute(_bounded("backtest_sharpe", 0.0, -3.0, 5.0))
    research.add_attribute(_bounded("live_sharpe", 0.0, -3.0, 5.0))
    research.add_attribute(_bounded("decay_ratio", 1.0, 0.0, 2.0))
    research.add_child(signal_generator("research_desk.signal"))
    research.add_method(Method(
        "iterate", lambda s, a=None: (_merge(s, backtest_sharpe=s["backtest_sharpe"] + 0.01), 0.0)))
    research.add_identity_constraint(IdentityConstraint("no_overfit", lambda s: s["decay_ratio"] > 0.3,
                                                        description="live/backtest ratio"))
    research.add_goal_constraint(GoalConstraint("high_sharpe", lambda s: s["live_sharpe"], direction="max"))
    attach_standard_daemons(research)
    research.add_neural(MLPComponent("research_policy", [3, 16, 2], seed=740), name="research_policy")

    # ---- level 0 : desk -------------------------------------------------
    desk = COHObject("trading_desk", description="autonomous multi-strategy trading desk")
    desk.add_attribute(_bounded("capital", 1e7, 0.0, 1e9))
    desk.add_attribute(_bounded("daily_pnl", 0.0, -1e7, 1e7))
    desk.add_attribute(_bounded("sharpe_30d", 0.0, -5.0, 10.0))
    desk.add_attribute(_bounded("max_drawdown_pct", 0.0, 0.0, 100.0))
    desk.add_child(exec_desk)
    desk.add_child(risk_office)
    desk.add_child(research)
    desk.add_method(Method(
        "close_of_day",
        lambda s, a=None: (_merge(s, daily_pnl=s["daily_pnl"] * 0.5,
                                  sharpe_30d=min(10.0, s["sharpe_30d"] + 0.01)), 0.0)))
    desk.add_identity_constraint(IdentityConstraint("capital_positive", lambda s: s["capital"] > 0.0))
    desk.add_identity_constraint(IdentityConstraint("drawdown_cap", lambda s: s["max_drawdown_pct"] < 25.0,
                                                    description="firm-level drawdown limit"))
    desk.add_trigger_constraint(TriggerConstraint(
        "drawdown_alert", "dd_high", lambda s: s["max_drawdown_pct"] > 15.0,
        "reduce_gross_exposure", priority=10))
    desk.add_trigger_constraint(TriggerConstraint(
        "pnl_target_hit", "target_reached", lambda s: s["daily_pnl"] > 1e5,
        "lock_in_profits", priority=1))
    desk.add_goal_constraint(GoalConstraint("max_sharpe", lambda s: s["sharpe_30d"], direction="max", weight=3.0))
    desk.add_goal_constraint(GoalConstraint("max_pnl", lambda s: s["daily_pnl"], direction="max", weight=2.0))
    desk.add_goal_constraint(GoalConstraint("min_drawdown", lambda s: -s["max_drawdown_pct"], direction="max"))
    attach_standard_daemons(desk)
    desk.add_neural(MLPComponent("capital_allocator", [4, 64, 8], seed=750), name="capital_allocator")
    desk.set_embedder(FeatureEmbedder(
        ["capital", "daily_pnl", "sharpe_30d", "max_drawdown_pct"],
        ranges={"capital": (0, 1e9), "daily_pnl": (-1e7, 1e7),
                "sharpe_30d": (-5, 10), "max_drawdown_pct": (0, 100)}))
    desk.flow = ContinuousFlow(
        lambda s, a, th: {
            "daily_pnl": -0.1 * s["daily_pnl"],
            "max_drawdown_pct": max(0.0, -0.01 * s["daily_pnl"] / 1e5) if s["daily_pnl"] < 0 else -0.001,
        },
        ["daily_pnl", "max_drawdown_pct"])
    return desk


# =============================================================================
# Driver: build, inspect, validate, simulate and reason
# =============================================================================
def depth(coh: COHObject, d: int = 0) -> int:
    if not coh.C:
        return d
    return max(depth(c, d + 1) for c in coh.C.values())


def summarise(name: str, coh: COHObject) -> None:
    banner(name)
    print(f"Depth of hierarchy        : {depth(coh) + 1} levels")
    print(f"Total descendants         : {len(coh.descendants())}")
    print(f"Nine-tuple report         : {coh.component_report()}")
    print(f"Is minimal (9/9 present)  : {coh.is_minimal()}")
    print(f"Acyclic                   : {coh.is_acyclic()}")
    sound = coh.soundness_report()
    print(f"Sound                     : {sound['sound']}  (violations={sound['violations'] or 'none'})")
    print(f"Embedding dimension       : {len(coh.embed())}")

    validator = COHValidator(require_all_components=False)
    issues = validator.validate(coh)
    errors = [i for i in issues if i.severity == "error"]
    print(f"Validator errors          : {len(errors)}")
    for e in errors[:3]:
        print(f"   ! {e}")


def exercise(coh: COHObject, steps: int = 8) -> None:
    """Run one simulation and a few reasoning queries."""
    try:
        traj = coh.simulate(steps=steps, dt=0.05)
        print(f"Simulated {steps} steps    : "
              f"final goal value = {traj.cumulative_reward():+.4f}")
    except Exception as exc:
        print(f"Simulation skipped        : {exc}")

    # constraint reasoner on the initial state
    try:
        res = ConstraintReasoner().infer(coh, Query("check invariants"))
        ok = res.answer is True
        print(f"Invariants at t=0         : {'OK' if ok else 'see reasoner output'}")
    except Exception as exc:
        print(f"Constraint reasoner       : {exc}")

    # goal solver
    try:
        res = GoalSolver(depth=2, beam=3).infer(coh, Query("best plan"))
        plan = res.answer.get("plan") if isinstance(res.answer, dict) else None
        print(f"Goal-solver best plan     : {plan if plan else '(no plan found)'}")
    except Exception as exc:
        print(f"Goal solver               : {exc}")

    # meta reasoner
    try:
        res = MetaReasoner().infer(coh, Query("analyse structure"))
        print(f"Meta-reasoner confidence  : {res.confidence:.2f}")
    except Exception as exc:
        print(f"Meta reasoner             : {exc}")


def main():
    systems = [
        ("SYSTEM 1 — Autonomous Warehouse Robot Fleet", build_warehouse_fleet()),
        ("SYSTEM 2 — Personalised Healthcare Assistant", build_health_assistant()),
        ("SYSTEM 3 — Smart Grid Energy Management",       build_smart_grid()),
        ("SYSTEM 4 — Adaptive E-Learning Tutor",          build_elearning_tutor()),
        ("SYSTEM 5 — Autonomous Trading Desk",            build_trading_desk()),
    ]

    for name, coh in systems:
        summarise(name, coh)
        exercise(coh)

    banner("Cross-system summary")
    print(f"{'system':<42}{'depth':>6}{'nodes':>8}{'I':>5}{'T':>5}{'G':>5}{'D':>5}")
    print("-" * 76)
    for name, coh in systems:
        r = coh.component_report()
        print(f"{name[:40]:<42}{depth(coh) + 1:>6}{len(coh.descendants()):>8}"
              f"{r['I']:>5}{r['T']:>5}{r['G']:>5}{r['D']:>5}")


if __name__ == "__main__":
    main()