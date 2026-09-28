# ============================================================
# FILE: gismol/tests/test_all.py
# ============================================================
"""Integration and unit tests for the gismol COH toolkit.

Run with::

    python -m unittest gismol.tests.test_all -v

or simply::

    python -m gismol.tests.test_all
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from gismol.core.coh_object import Attribute, COHObject, Method, State
from gismol.core.constraints import (
    ConstraintSet,
    GoalConstraint,
    IdentityConstraint,
    TriggerConstraint,
)
from gismol.core.daemons import BoundaryDaemon, InvariantDaemon, RateDaemon
from gismol.core.dynamics import ContinuousFlow, DiscreteTransition, HybridDynamicalSystem
from gismol.core.embedding import CompositeEmbedder, FeatureEmbedder, HashEmbedder
from gismol.core.exceptions import (
    ConstraintViolationError,
    CyclicHierarchyError,
    MethodExecutionError,
)
from gismol.core.repository import COHRepository
from gismol.core import expression as E

from gismol.categorical.category import COHCategory, COHMorphism
from gismol.categorical.functors import ComponentFunctor, Graph
from gismol.categorical.limits import constraint_limit, product_set, pullback

from gismol.dynamical.hybrid import coh_to_hybrid
from gismol.dynamical.stability import LyapunovAnalysis, quadratic_lyapunov
from gismol.dynamical.free_energy import (
    FreeEnergy,
    GenerativeModel,
    VariationalPosterior,
    coh_free_energy_correspondence,
    kl_divergence,
)

from gismol.neural.nn import MLP
from gismol.neural.components import MLPComponent, PolicyComponent, ValueComponent
from gismol.neural.optimizers import Adam, SGD
from gismol.neural.constrained_rl import (
    ConstrainedMDP,
    LagrangianAgent,
    LinearSoftmaxPolicy,
    run_episode,
)
from gismol.neural.credit_assignment import CreditAssigner, HierarchicalValueDecomposition

from gismol.stability.lyapunov import ConstraintStabilityAnalyzer, lyapunov_from_constraints

from gismol.nlp.normalizers import normalize
from gismol.nlp.parsers import parse, split_sentences, tokenize
from gismol.nlp.entity_extractor import EntityExtractor, extract_entities
from gismol.nlp.relations_miner import RelationMiner
from gismol.nlp.intent_recognizer import IntentRecognizer
from gismol.nlp.constraint_parser import ConstraintParser, parse_constraints
from gismol.nlp.embedders import BagOfWordsEmbedder, HashingEmbedder, TfidfEmbedder
from gismol.nlp.similarity import jaccard, levenshtein, SimilarityIndex
from gismol.nlp.response_generator import ResponseGenerator
from gismol.nlp.chat_module import ChatModule
from gismol.nlp.chat_manager import ChatManager
from gismol.nlp.text_to_coh import TextToCOH, text_to_coh
from gismol.nlp.validation import COHValidator, validate_coh

from gismol.reasoners.base import KnowledgeBase, Query, Rule
from gismol.reasoners.general import (
    AbductiveReasoner,
    ConstraintReasoner,
    ForwardChainer,
    GeneralReasoner,
    GoalSolver,
)
from gismol.reasoners.domain import DomainReasoner, get_domain, DOMAIN_PROFILES
from gismol.reasoners.advanced import (
    AnalogicalReasoner,
    CategoricalReasoner,
    CausalReasoner,
    MetaReasoner,
)


# ==========================================================================
def build_thermostat() -> COHObject:
    """A small but complete COH: a room thermostat."""
    coh = COHObject(name="thermostat", description="Room temperature controller")
    coh.add_attribute(Attribute("temperature", 20.0, bounds=(0.0, 60.0)))
    coh.add_attribute(Attribute("heater_power", 0.0, bounds=(0.0, 1.0)))
    coh.add_attribute(Attribute("target", 22.0))

    def heat(state, action=None):
        out = state.copy()
        out["temperature"] = out["temperature"] + 0.1 * out["heater_power"]
        return out, 0.1

    def cool(state, action=None):
        out = state.copy()
        out["temperature"] = out["temperature"] - 0.1
        return out, 0.05

    coh.add_method(Method("heat", heat))
    coh.add_method(Method("cool", cool))

    coh.add_identity_constraint(IdentityConstraint(
        "temp_bounds", lambda s: 0.0 <= s["temperature"] <= 60.0))
    coh.add_trigger_constraint(TriggerConstraint(
        "overheat", "temperature_high",
        lambda s: s["temperature"] > 50.0,
        lambda s: {k: v for k, v in s.items() if k != "temperature"} | {"temperature": 50.0},
        priority=2))
    coh.add_goal_constraint(GoalConstraint(
        "comfort", lambda s: -abs(s["temperature"] - s["target"]), direction="max"))
    coh.add_daemon(InvariantDaemon(constraints=list(coh.I)))
    coh.add_daemon(BoundaryDaemon({"temperature": (0.0, 60.0), "heater_power": (0.0, 1.0)}))
    coh.add_neural(MLPComponent("controller", [3, 8, 1], seed=1))
    coh.set_embedder(FeatureEmbedder(["temperature", "heater_power"],
                                     {"temperature": (0, 60), "heater_power": (0, 1)}))

    flow = ContinuousFlow(
        lambda s, a, th: {
            "temperature": -0.01 * (s["temperature"] - 20.0) + 0.5 * s["heater_power"]
        },
        ["temperature"])
    coh.flow = flow
    coh.jump = DiscreteTransition(lambda s, a, th: s)
    return coh


# ==========================================================================
class TestExpression(unittest.TestCase):
    def test_build_and_evaluate(self):
        x = E.Var("x")
        expr = (x > 0) & (x < 10)
        self.assertTrue(expr.evaluate({"x": 5}))
        self.assertFalse(expr.evaluate({"x": 15}))

    def test_parse_expression(self):
        expr = E.parse_expression("x > 0 and x < 10")
        self.assertTrue(expr.evaluate({"x": 3}))
        self.assertFalse(expr.evaluate({"x": -1}))

    def test_arithmetic_and_calls(self):
        expr = E.parse_expression("abs(x - 3) < 2")
        self.assertTrue(expr.evaluate({"x": 4}))
        self.assertFalse(expr.evaluate({"x": 9}))

    def test_bool_guard(self):
        with self.assertRaises(TypeError):
            bool(E.Var("x") > 1)

    def test_dotted_lookup(self):
        state = State({"arm": State({"angle": 1.5})})
        self.assertAlmostEqual(E.Var("arm.angle").evaluate(state), 1.5)

    def test_parse_error(self):
        with self.assertRaises(E.ParseError):
            E.parse_expression("x.__class__")


class TestConstraints(unittest.TestCase):
    def test_identity_satisfaction(self):
        c = IdentityConstraint("positive", lambda s: s["x"] > 0)
        self.assertTrue(c.satisfied({"x": 1}))
        self.assertFalse(c.satisfied({"x": -1}))
        self.assertGreater(c.violation({"x": -1}), 0.0)

    def test_constraint_check_raises(self):
        c = IdentityConstraint("positive", lambda s: s["x"] > 0)
        with self.assertRaises(ConstraintViolationError):
            c.check({"x": -5})

    def test_trigger_matching_and_firing(self):
        t = TriggerConstraint("stop", "collision", lambda s: s["v"] > 10, "brake")
        self.assertTrue(t.matches("collision", {"v": 20}))
        self.assertFalse(t.matches("collision", {"v": 5}))
        self.assertFalse(t.matches("other", {"v": 20}))
        self.assertEqual(t.fire({"v": 20}), "brake")

    def test_goal_utility(self):
        maximize = GoalConstraint("g", lambda s: s["x"], direction="max")
        minimize = GoalConstraint("h", lambda s: s["x"], direction="min")
        self.assertGreater(maximize.utility({"x": 10}), maximize.utility({"x": 1}))
        self.assertGreater(minimize.utility({"x": 1}), minimize.utility({"x": 10}))

    def test_constraint_set_total_goal(self):
        cs = ConstraintSet(
            identity=[IdentityConstraint("i", lambda s: s["x"] >= 0)],
            goals=[GoalConstraint("g1", lambda s: s["x"], weight=2.0),
                   GoalConstraint("g2", lambda s: -s["x"], weight=1.0)],
        )
        self.assertAlmostEqual(cs.total_goal({"x": 3}), 2 * 3 - 3)
        self.assertTrue(cs.satisfies_identity({"x": 0}))
        self.assertEqual(len(cs), 3)


class TestStateAndAttributes(unittest.TestCase):
    def test_state_paths(self):
        s = State({"a": 1})
        s.set_path("b.c", 2)
        self.assertEqual(s.get_path("b.c"), 2)
        self.assertEqual(s.flatten()["b.c"], 2)

    def test_initial_state_from_attributes(self):
        coh = COHObject("o", attributes=[Attribute("x", 3.0), Attribute("y", 4.0)])
        self.assertEqual(coh.state["x"], 3.0)
        self.assertEqual(coh.state["y"], 4.0)


class TestCOHObject(unittest.TestCase):
    def test_nine_components(self):
        coh = build_thermostat()
        report = coh.component_report()
        self.assertGreater(report["A"], 0)
        self.assertGreater(report["M"], 0)
        self.assertGreater(report["N"], 0)
        self.assertGreater(report["I"], 0)
        self.assertGreater(report["T"], 0)
        self.assertGreater(report["G"], 0)
        self.assertGreater(report["D"], 0)

    def test_hierarchy_and_topological_order(self):
        root = COHObject("root")
        child = COHObject("child")
        grandchild = COHObject("grandchild")
        child.add_child(grandchild)
        root.add_child(child)
        order = [o.name for o in root.topological_order()]
        self.assertEqual(order, ["grandchild", "child", "root"])
        self.assertTrue(root.is_acyclic())

    def test_cycle_detection(self):
        a = COHObject("a")
        b = COHObject("b")
        a.add_child(b)
        with self.assertRaises(CyclicHierarchyError):
            b.add_child(a)

    def test_method_invocation(self):
        coh = COHObject("m")
        coh.add_attribute(Attribute("x", 0))
        coh.add_method(Method("inc", lambda s, a=None: ({**s, "x": s["x"] + 1}, 1.0)))
        new_state, reward = coh.invoke("inc")
        self.assertEqual(new_state["x"], 1)
        self.assertEqual(reward, 1.0)

    def test_method_precondition(self):
        coh = COHObject("m")
        coh.add_attribute(Attribute("x", -1))
        coh.add_method(Method("inc", lambda s, a=None: (s, 0.0),
                              preconditions=[lambda s: s["x"] > 0]))
        with self.assertRaises(MethodExecutionError):
            coh.invoke("inc")

    def test_embedding_and_similarity(self):
        a = build_thermostat()
        b = build_thermostat()
        self.assertAlmostEqual(a.similarity(b), 1.0, places=6)

    def test_soundness_report(self):
        coh = build_thermostat()
        report = coh.soundness_report()
        self.assertTrue(report["hierarchical_sound"])
        self.assertTrue(report["sound"])

    def test_product_coproduct_exponential(self):
        a = COHObject("a", attributes=[Attribute("x", 1)])
        b = COHObject("b", attributes=[Attribute("y", 2)])
        prod = a.product(b)
        self.assertIn("a", prod.C)
        self.assertIn("b", prod.C)
        coprod = a.coproduct(b)
        self.assertEqual(len(coprod.C), 2)
        exp = b.exponential(a)
        self.assertIn("apply", exp.M)

    def test_serialisation_roundtrip(self):
        coh = build_thermostat()
        data = coh.to_dict()
        restored = COHObject.from_dict(data)
        self.assertEqual(restored.name, coh.name)
        self.assertEqual(len(restored.I), len(coh.I))

    def test_description(self):
        coh = build_thermostat()
        self.assertIn("thermostat", coh.describe())


class TestDaemons(unittest.TestCase):
    def test_invariant_daemon(self):
        c = IdentityConstraint("positive", lambda s: s["x"] > 0)
        d = InvariantDaemon(constraints=[c])
        self.assertIsNone(d.observe({"x": 1}))
        action = d.observe({"x": -1})
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "restore_invariants")

    def test_boundary_daemon_projection(self):
        d = BoundaryDaemon({"x": (0.0, 1.0)})
        action = d.observe({"x": 5.0})
        self.assertIsNotNone(action)
        self.assertAlmostEqual(action.payload["x"], 1.0)

    def test_rate_daemon(self):
        d = RateDaemon("x", max_rate=1.0)
        d.observe({"x": 0.0}, 0.0)
        self.assertIsNone(d.observe({"x": 0.5}, 1.0))
        action = d.observe({"x": 10.0}, 2.0)
        self.assertIsNotNone(action)

    def test_network_picks_highest_severity(self):
        coh = build_thermostat()
        coh.set("temperature", 1000.0)
        action = coh.monitor()
        self.assertIsNotNone(action)


class TestDynamics(unittest.TestCase):
    def test_continuous_flow_rk4(self):
        flow = ContinuousFlow(lambda s, a, th: {"x": 1.0}, ["x"])
        state = State({"x": 0.0})
        out = flow.step(state, dt=1.0)
        self.assertAlmostEqual(out["x"], 1.0, places=6)

    def test_hybrid_system_simulation(self):
        coh = build_thermostat()
        coh.set("temperature", 20.0)
        coh.set("heater_power", 1.0)
        traj = coh.simulate(steps=20, dt=0.1)
        self.assertEqual(len(traj), 21)
        self.assertGreater(traj.final["temperature"], 20.0)

    def test_trigger_switching(self):
        coh = build_thermostat()
        coh.set("temperature", 55.0)
        state, info = coh.step(dt=0.0, event="temperature_high")
        self.assertIn("overheat", info["events"])
        self.assertLessEqual(state["temperature"], 50.0)

    def test_lipschitz_report(self):
        coh = build_thermostat()
        system = coh.to_hybrid_system()
        report = system.lipschitz_report(coh.state)
        self.assertTrue(report["well_posed"])
        self.assertIsNotNone(report["flow_lipschitz"])

    def test_hybrid_automaton(self):
        coh = build_thermostat()
        automaton = coh_to_hybrid(coh)
        traj = automaton.run(coh.state, steps=5, dt=0.1)
        self.assertEqual(len(traj), 6)


class TestEmbedding(unittest.TestCase):
    def test_feature_embedder_normalisation(self):
        emb = FeatureEmbedder(["x"], {"x": (0.0, 10.0)})
        self.assertAlmostEqual(emb({"x": 5.0})[0], 0.5)

    def test_hash_embedder_determinism(self):
        emb = HashEmbedder(dim=16, seed=1)
        a = COHObject("a", attributes=[Attribute("x", 1)])
        b = COHObject("a", attributes=[Attribute("x", 1)])
        self.assertEqual(emb(a), emb(b))

    def test_composite_embedder(self):
        root = COHObject("root")
        root.add_child(COHObject("c1", attributes=[Attribute("x", 1)]))
        root.add_child(COHObject("c2", attributes=[Attribute("y", 2)]))
        emb = CompositeEmbedder(FeatureEmbedder(["x"], default=0.0))
        vec = emb(root)
        self.assertGreater(len(vec), 0)

    def test_hilbert_space(self):
        from gismol.core.embedding import HilbertSpace
        H = HilbertSpace(3)
        self.assertAlmostEqual(H.similarity((1, 0, 0), (1, 0, 0)), 1.0)
        self.assertAlmostEqual(H.similarity((1, 0, 0), (0, 1, 0)), 0.0)


class TestRepository(unittest.TestCase):
    def test_add_get_remove(self):
        repo = COHRepository()
        coh = build_thermostat()
        oid = repo.add(coh, tags=["control"])
        self.assertIs(repo.get(oid), coh)
        self.assertIn("control", repo.tags_of(oid))
        repo.remove(oid)
        self.assertEqual(len(repo), 0)

    def test_hierarchy_index(self):
        root = COHObject("root")
        root.add_child(COHObject("child"))
        repo = COHRepository()
        ids = repo.index_hierarchy(root)
        self.assertEqual(len(ids), 2)

    def test_save_and_load(self):
        repo = COHRepository()
        repo.add(build_thermostat())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "repo.json")
            repo.save(path)
            loaded = COHRepository.load(path)
            self.assertEqual(len(loaded), 1)


class TestCategory(unittest.TestCase):
    def test_coh_category_products(self):
        cat = COHCategory()
        a = COHObject("a", attributes=[Attribute("x", 1)])
        b = COHObject("b", attributes=[Attribute("y", 2)])
        cat.register(a)
        cat.register(b)
        prod = cat.product(a, b)
        self.assertIn("a", prod.C)
        self.assertTrue(cat.is_cartesian_closed())

    def test_morphism_preservation(self):
        a = COHObject("a", attributes=[Attribute("x", 1)])
        b = COHObject("b", attributes=[Attribute("x", 1)])
        mor = COHMorphism(a, b, name="f")
        self.assertTrue(mor.preserves())

    def test_morphism_rejects_structure_loss(self):
        a = COHObject("a", attributes=[Attribute("x", 1)])
        b = COHObject("b")
        with self.assertRaises(Exception):
            COHMorphism(a, b)

    def test_component_functor(self):
        coh = build_thermostat()
        graph = ComponentFunctor().map_object(coh)
        self.assertIn("thermostat", graph.nodes)
        self.assertFalse(graph.has_cycle())


class TestLimits(unittest.TestCase):
    def test_constraint_limit(self):
        constraints = [
            IdentityConstraint("pos", lambda s: s["x"] > 0),
            IdentityConstraint("small", lambda s: s["x"] < 10),
        ]
        domain = [{"x": v} for v in range(-5, 15)]
        limit = constraint_limit(constraints, domain)
        self.assertEqual(len(limit.apex), 9)

    def test_product_and_pullback(self):
        self.assertEqual(len(product_set([{1, 2}, {3}])), 2)
        pb = pullback(lambda a: a % 2, lambda b: b % 2, [1, 2], [3, 4])
        self.assertEqual(pb, {(1, 3), (2, 4)})


class TestStability(unittest.TestCase):
    def test_quadratic_lyapunov(self):
        V = quadratic_lyapunov({"x": 0.0})
        self.assertAlmostEqual(V.value({"x": 2.0}), 4.0)

    def test_constraint_analyzer(self):
        coh = build_thermostat()
        analyzer = ConstraintStabilityAnalyzer(coh)
        states = [{"temperature": t, "heater_power": 0.0, "target": 22.0}
                  for t in (20.0, 25.0, 30.0)]
        certificate = analyzer.analyze(states=states)
        self.assertIsInstance(certificate.stable, bool)

    def test_lyapunov_from_constraints(self):
        coh = build_thermostat()
        V = lyapunov_from_constraints(coh)
        self.assertEqual(V.value({"temperature": 20.0, "heater_power": 0.0}), 0.0)
        self.assertGreater(V.value({"temperature": -5.0, "heater_power": 0.0}), 0.0)


class TestFreeEnergy(unittest.TestCase):
    def test_kl_divergence_zero_for_identical(self):
        p = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=9)

    def test_kl_divergence_positive(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.5, "b": 0.5}
        self.assertGreater(kl_divergence(p, q), 0.0)

    def test_free_energy_minimisation(self):
        model = GenerativeModel({"s1": 0.5, "s2": 0.5},
                               {"s1": {"o1": 0.9, "o2": 0.1},
                                "s2": {"o1": 0.2, "o2": 0.8}})
        fe = FreeEnergy(model)
        candidates = [VariationalPosterior({"s1": 1.0}),
                      VariationalPosterior({"s2": 1.0})]
        best = fe.minimise("o1", candidates)
        self.assertIn("s1", best.distribution)

    def test_coh_correspondence(self):
        coh = build_thermostat()
        mapping = coh_free_energy_correspondence(coh)
        self.assertIn("identity_constraints", mapping)
        self.assertEqual(mapping["identity_role"], "structural priors  p(s)")


class TestNeural(unittest.TestCase):
    def test_mlp_forward_shapes(self):
        mlp = MLP([3, 5, 2], seed=0)
        out = mlp.forward([1.0, 2.0, 3.0])
        self.assertEqual(len(out), 2)

    def test_mlp_learns_xor_like_function(self):
        mlp = MLP([2, 8, 1], activation="tanh", seed=0)
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        Y = [[0.0], [1.0], [1.0], [0.0]]
        history = mlp.fit(X, Y, epochs=400, lr=0.1, optimizer=Adam(0.05), seed=1)
        self.assertLess(history[-1], history[0])

    def test_optimizers_step(self):
        for opt in (SGD(0.1), Adam(0.01)):
            out = opt.step([1.0, 2.0], [0.5, 0.5])
            self.assertEqual(len(out), 2)
            self.assertLess(out[0], 1.0)

    def test_mlp_component_parameters(self):
        comp = MLPComponent("c", [2, 4, 1], seed=0)
        self.assertEqual(len(comp.theta), comp.mlp.n_parameters)

    def test_policy_component_probs(self):
        policy = PolicyComponent("p", input_dim=2, n_actions=3, seed=0)
        probs = policy([0.5, -0.5])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_value_component(self):
        value = ValueComponent("v", input_dim=2, seed=0)
        self.assertIsInstance(value.value([1.0, 2.0]), float)


class TestConstrainedRL(unittest.TestCase):
    def test_episode_runs(self):
        env = ConstrainedMDP(4, 3, cost_limit=1.0, horizon=5, seed=0)
        policy = LinearSoftmaxPolicy(4, 3, seed=0)
        episode = run_episode(env, policy)
        self.assertEqual(len(episode["states"]), 5)
        self.assertIn("cost", episode)

    def test_lagrangian_agent_trains(self):
        env = ConstrainedMDP(4, 3, cost_limit=0.5, horizon=6, seed=0)
        agent = LagrangianAgent(env, seed=0)
        history = agent.train(iterations=30, episodes_per_iter=4)
        self.assertEqual(len(history), 30)
        self.assertIn("lambda", history[-1])

    def test_evaluation(self):
        env = ConstrainedMDP(3, 2, horizon=4, seed=0)
        agent = LagrangianAgent(env, seed=0)
        metrics = agent.evaluate(episodes=10)
        self.assertIn("mean_return", metrics)


class TestCreditAssignment(unittest.TestCase):
    def test_decomposition(self):
        root = COHObject("root")
        root.add_child(COHObject("a", methods=[Method("m", lambda s, a=None: (s, 0))]))
        root.add_child(COHObject("b"))
        decomp = HierarchicalValueDecomposition(root)
        self.assertAlmostEqual(sum(decomp.weights.values()), 1.0, places=9)
        value = decomp.decompose({"a": 10.0, "b": 0.0})
        self.assertAlmostEqual(value, 5.0, places=6)

    def test_fit_weights(self):
        root = COHObject("root")
        root.add_child(COHObject("a"))
        root.add_child(COHObject("b"))
        decomp = HierarchicalValueDecomposition(root)
        samples = [({"a": 1.0, "b": 0.0}, 0.7), ({"a": 0.0, "b": 1.0}, 0.3)]
        result = decomp.fit(samples, lr=0.1, epochs=300)
        self.assertIn("weights", result)

    def test_credit_assigner(self):
        root = COHObject("root")
        root.add_child(COHObject("a"))
        assigner = CreditAssigner(root)
        out = assigner.assign(10.0, {"a": 10.0})
        self.assertIn("children", out)
        deep = assigner.deep_assign(10.0)
        self.assertIn("a", deep["children"])


class TestNLP(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(normalize("  Hello   WORLD  "), "hello world")

    def test_tokenisation_and_sentences(self):
        tokens = tokenize("the quick brown fox")
        self.assertEqual(len(tokens), 4)
        sentences = split_sentences("One. Two! Three?")
        self.assertEqual(len(sentences), 3)

    def test_document_parsing(self):
        doc = parse("The temperature must be less than 50 degrees.")
        self.assertGreater(len(doc.tokens), 5)
        self.assertTrue(any(t.pos == "NUM" for t in doc.tokens))
        self.assertGreater(len(doc.noun_phrases()), 0)

    def test_entity_extraction(self):
        entities = extract_entities("Set x to 42 and y to 3.5% at 2024-01-01.")
        labels = {e.label for e in entities}
        self.assertIn("NUMBER", labels)
        self.assertIn("DATE", labels)

    def test_relation_mining(self):
        miner = RelationMiner()
        relations = miner.mine("The pump controls the valve. Pressure depends on flow.")
        predicates = {r.predicate for r in relations}
        self.assertTrue({"controls", "depends_on"} & predicates)

    def test_intent_recognition(self):
        recognizer = IntentRecognizer()
        self.assertEqual(recognizer.recognize("maximize throughput").name, "optimize")
        self.assertEqual(recognizer.recognize("define a new system").name, "define")
        self.assertEqual(recognizer.recognize("hello there").name, "chat")

    def test_constraint_parser_invariants(self):
        parsed = parse_constraints("temperature must be less than 50")
        self.assertEqual(len(parsed.identity), 1)
        self.assertTrue(parsed.identity[0].satisfied({"temperature": 20}))
        self.assertFalse(parsed.identity[0].satisfied({"temperature": 80}))

    def test_constraint_parser_triggers(self):
        parsed = parse_constraints("if pressure > 100 then vent")
        self.assertEqual(len(parsed.triggers), 1)
        trigger = parsed.triggers[0]
        self.assertTrue(trigger.matches(trigger.event, {"pressure": 120}))

    def test_constraint_parser_goals(self):
        parsed = parse_constraints("maximize throughput")
        self.assertEqual(len(parsed.goals), 1)
        self.assertEqual(parsed.goals[0].direction, "max")

    def test_constraint_parser_explicit_expression(self):
        parsed = parse_constraints("constraint: x > 0 and x < 10")
        self.assertEqual(len(parsed.identity), 1)
        self.assertTrue(parsed.identity[0].satisfied({"x": 5}))

    def test_text_embedders(self):
        docs = ["the cat sat on the mat", "the dog sat on the log"]
        bow = BagOfWordsEmbedder().fit(docs)
        v = bow("the cat sat")
        self.assertEqual(len(v), len(bow.vocabulary))
        tfidf = TfidfEmbedder().fit(docs)
        self.assertGreater(len(tfidf("the cat")), 0)
        hashing = HashingEmbedder(dim=32)
        self.assertEqual(len(hashing("hello world")), 32)

    def test_similarity_metrics(self):
        self.assertAlmostEqual(jaccard({1, 2, 3}, {1, 2, 3}), 1.0)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        index = SimilarityIndex()
        index.add("a", (1.0, 0.0))
        index.add("b", (0.0, 1.0))
        best = index.query((1.0, 0.0), k=1)
        self.assertEqual(best[0][0], "a")

    def test_response_generator(self):
        coh = build_thermostat()
        gen = ResponseGenerator()
        self.assertIn("thermostat", gen.describe_coh(coh))
        self.assertIn("Invariant", gen.explain_constraints(coh))

    def test_text_to_coh(self):
        coh = text_to_coh(
            "A thermostat must keep temperature below 50. "
            "If temperature exceeds 45 then cool. "
            "Maximize comfort.",
            name="thermostat")
        self.assertGreater(len(coh.A), 0)
        self.assertGreater(len(coh.I), 0)
        self.assertGreater(len(coh.T), 0)
        self.assertGreater(len(coh.G), 0)
        self.assertGreater(len(coh.D), 0)

    def test_chat_module(self):
        chat = ChatModule()
        response = chat.process("define a system called pump")
        self.assertIsNotNone(response.coh)
        response = chat.process("the pressure must be less than 100")
        self.assertGreater(len(chat.coh.I), 0)
        response = chat.process("describe it")
        self.assertIn("pump", response.text)

    def test_chat_manager_sessions(self):
        manager = ChatManager()
        session = manager.new_session("s1")
        session.say("hello")
        self.assertEqual(len(session.turns), 1)
        manager.process("s1", "define a thing")
        self.assertEqual(len(manager.list_sessions()), 1)

    def test_validator(self):
        coh = build_thermostat()
        validator = COHValidator()
        issues = validator.validate(coh)
        self.assertIsInstance(issues, list)
        self.assertTrue(validate_coh(coh))


class TestReasoners(unittest.TestCase):
    def test_knowledge_base_chaining(self):
        kb = KnowledgeBase({"a": True})
        kb.add_rule(Rule("r1", lambda f: f.get("a"), {"b": True}))
        kb.add_rule(Rule("r2", lambda f: f.get("b"), {"c": True}))
        facts = kb.infer()
        self.assertTrue(facts["b"])
        self.assertTrue(facts["c"])

    def test_forward_chainer(self):
        coh = build_thermostat()
        result = ForwardChainer().infer(coh, Query("object"))
        self.assertEqual(result.answer, "thermostat")

    def test_constraint_reasoner(self):
        coh = build_thermostat()
        good = ConstraintReasoner().infer(coh, Query("check"))
        self.assertTrue(good.answer)
        coh.set("temperature", -100.0)
        bad = ConstraintReasoner().infer(coh, Query("violations"))
        self.assertFalse(bad.answer is True)

    def test_goal_solver(self):
        coh = build_thermostat()
        coh.set("heater_power", 1.0)
        result = GoalSolver(depth=2, beam=2).infer(coh, Query("best plan"))
        self.assertIn("plan", result.answer)

    def test_abductive_reasoner(self):
        coh = build_thermostat()
        coh.set("temperature", -10.0)
        result = AbductiveReasoner().infer(coh, Query("why", parameters={"state": coh.state}))
        self.assertIsInstance(result.answer, dict)

    def test_general_reasoner_dispatch(self):
        coh = build_thermostat()
        result = GeneralReasoner().infer(coh, Query("maximize comfort"))
        self.assertIsNotNone(result.answer)

    def test_domain_reasoner(self):
        coh = COHObject("physics_system", description="force and mass")
        coh.add_attribute(Attribute("mass", 2.0))
        coh.add_attribute(Attribute("acceleration", 3.0))
        reasoner = DomainReasoner(profile=get_domain("physics"))
        result = reasoner.infer(coh, Query("analyse"))
        self.assertEqual(result.answer["domain"], "physics")

    def test_domain_auto_detection(self):
        reasoner = DomainReasoner()
        profile = reasoner.select_profile("the gene expression of a protein in a cell")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "biology")

    def test_categorical_reasoner(self):
        coh = build_thermostat()
        result = CategoricalReasoner().infer(coh, Query("structure"))
        self.assertTrue(result.answer["cartesian_closed"])
        self.assertTrue(result.answer["acyclic"])

    def test_analogical_reasoner(self):
        a = build_thermostat()
        b = build_thermostat()
        result = AnalogicalReasoner().infer(a, Query("similar", parameters={"other": b}))
        self.assertTrue(result.answer["analogous"])

    def test_causal_reasoner(self):
        coh = build_thermostat()
        result = CausalReasoner().infer(
            coh, Query("temperature", parameters={"intervene": True}))
        self.assertIn("effects", result.answer)

    def test_meta_reasoner(self):
        coh = build_thermostat()
        result = MetaReasoner().infer(coh, Query("maximize comfort"))
        self.assertIsInstance(result.answer, dict)


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        text = (
            "Define a thermal plant called reactor. "
            "The core temperature must be less than 900. "
            "The coolant flow must be at least 5. "
            "If core temperature exceeds 800 then increase coolant. "
            "Maximize energy output. "
            "Minimize coolant usage."
        )
        coh = text_to_coh(text, name="reactor")

        self.assertIn("reactor", coh.name)
        self.assertGreater(len(coh.I), 0)
        self.assertGreater(len(coh.T), 0)
        self.assertGreater(len(coh.G), 0)
        self.assertGreater(len(coh.A), 0)

        validator = COHValidator()
        self.assertTrue(validator.is_valid(coh))

        report = coh.soundness_report()
        self.assertTrue(report["hierarchical_sound"])

        embedded = coh.embed()
        self.assertGreater(len(embedded), 0)

        analyzer = ConstraintStabilityAnalyzer(coh)
        certificate = analyzer.analyze()
        self.assertIsInstance(certificate.stable, bool)

        reasoner = GeneralReasoner()
        result = reasoner.infer(coh, Query("maximize energy output"))
        self.assertIsNotNone(result.answer)

        chat = ChatModule(coh=coh)
        response = chat.process("describe the reactor")
        self.assertIn("reactor", response.text)


# ==========================================================================
def _run_all():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    unittest.main(verbosity=2)
    