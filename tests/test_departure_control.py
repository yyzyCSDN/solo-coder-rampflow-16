import unittest
from datetime import datetime, timezone, timedelta

from rampflow.departure_control import DepartureCoordinator, DepartureRejected
from rampflow.models import Decision
from rampflow.runway_slots import Runway, RunwaySlotManager, SlotRequest
from rampflow.taxi import TaxiEdge, TaxiGraph, TaxiNode, TaxiScheduler
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc
D = lambda h, m=0, s=0: datetime(2026, 1, 1, h, m, s, tzinfo=UTC)


def scheduler():
    # Two feeds converge on intersection X and leave via different exits, so
    # the routes share only the node, never an edge:
    #   stand -e1-> X -e2-> hold      (runway entry for F1)
    #   other -e3-> X -e4-> hold2     (runway entry for F2)
    nodes = [TaxiNode("stand"), TaxiNode("X"), TaxiNode("hold"),
             TaxiNode("other"), TaxiNode("hold2")]
    edges = [
        TaxiEdge("e1", "stand", "X", 300.0, 5.0),   # 60 s
        TaxiEdge("e2", "X", "hold", 300.0, 5.0),    # 60 s
        TaxiEdge("e3", "other", "X", 300.0, 5.0),   # 60 s
        TaxiEdge("e4", "X", "hold2", 300.0, 5.0),   # 60 s
    ]
    return TaxiScheduler(TaxiGraph(nodes, edges), separation_seconds=60)


WAKE = {
    ("H", "L"): 180, ("H", "M"): 120, ("H", "H"): 90,
    ("M", "L"): 120, ("M", "M"): 90, ("M", "H"): 60,
    ("L", "L"): 75, ("L", "M"): 60, ("L", "H"): 60,
}


def slot_manager():
    return RunwaySlotManager(
        [Runway("R1", frozenset({"09"}), 10.0, 20.0, 60)],
        WAKE,
        grid_seconds=30,
    )


def request(flight_id, wake, target):
    return SlotRequest(
        flight_id, "departure", wake, target, target, D(12),
        runway_preferences=("R1",),
    )


class WakeSeparationDirectionTests(unittest.TestCase):
    def test_leader_follower_order_is_directional(self):
        mgr = slot_manager()
        self.assertEqual(mgr._sep("H", "L"), 180)  # heavy first: long wake
        self.assertEqual(mgr._sep("L", "H"), 60)   # light first: short wake
        self.assertEqual(mgr._sep("H", "M"), 120)
        self.assertEqual(mgr._sep("M", "H"), 60)

    def test_heavy_leader_spaces_follower_by_heavy_wake(self):
        mgr = slot_manager()
        heavy = mgr.find(request("HEAVY", "H", D(10)))
        mgr.commit(heavy)
        light = mgr.find(request("LIGHT", "L", D(10)))
        # H->L spacing is 180 s measured from the heavy's slot; the reversed
        # (L->H = 60 s) value must not be used.
        self.assertEqual(light.slot, heavy.slot + timedelta(seconds=180))

    def test_light_leader_then_heavy_only_needs_short_spacing(self):
        mgr = slot_manager()
        light = mgr.find(request("LIGHT", "L", D(10)))
        mgr.commit(light)
        heavy = mgr.find(request("HEAVY", "H", D(10)))
        self.assertEqual(heavy.slot, light.slot + timedelta(seconds=60))


class JointDepartureGateTests(unittest.TestCase):
    def test_happy_path_commits_both_gates(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)
        plan = coord.prepare("F1", "stand", "hold", D(10), request("F1", "M", D(10, 10)))
        clearance = coord.commit(plan)
        self.assertIn("F1", coord.clearances)
        self.assertEqual(clearance.runway_use.slot, D(10, 10))
        self.assertEqual(len(taxi.occupancy("e1", Window(D(9), D(12)))), 1)
        self.assertEqual(mgr.slot_for("F1").flight_id, "F1")

    def test_late_converging_route_fails_joint_revalidation_and_rolls_back(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)

        plan1 = coord.prepare("F1", "stand", "hold", D(10), request("F1", "M", D(10, 10)))

        # F1 passes each of its two gates individually ...
        coord._stage(plan1)

        # ... and only after that does F2's competing route reach the
        # controller.  It was planned against the empty network and uses
        # different edges, so the edge-based gate accepts it; F1's earlier
        # individual checks had no way to see it either.
        plan2_taxi = taxi.plan("F2", "other", "hold2", D(10, 0, 30), "C")
        self.assertTrue(taxi.commit(plan2_taxi).accepted)

        # The joint revalidation now rejects F1 and returns *both* resources.
        with self.assertRaises(DepartureRejected) as ctx:
            coord._finalize(plan1)
        codes = {d.code for d in ctx.exception.decisions}
        self.assertIn("intersection-conflict", codes)

        # Route is handed back ...
        self.assertEqual(taxi.occupancy("e1", Window(D(9), D(12))), ())
        # ... and the runway slot is offered again as well.
        with self.assertRaises(KeyError):
            mgr.slot_for("F1")
        self.assertNotIn("F1", coord.clearances)

    def test_release_after_success_returns_everything(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)
        plan = coord.prepare("F3", "stand", "hold", D(10), request("F3", "M", D(10, 10)))
        coord.commit(plan)
        coord.release("F3")
        self.assertNotIn("F3", coord.clearances)
        with self.assertRaises(KeyError):
            mgr.slot_for("F3")
        self.assertEqual(taxi.occupancy("e1", Window(D(9), D(12))), ())
        # A new flight can now take the slot the released flight held.
        follow = mgr.find(request("F4", "M", D(10, 10)))
        self.assertEqual(follow.slot, D(10, 10))

    def test_failed_slot_gate_returns_already_staged_route(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)

        class FlakySlots(RunwaySlotManager):
            def verify(self, use):
                return Decision(False, "slot-stale", "runway sequence moved", {})

        flaky = FlakySlots(list(mgr.runways.values()), WAKE)
        flaky_coord = DepartureCoordinator(taxi, flaky)
        plan = flaky_coord.prepare("S", "stand", "hold", D(10), request("S", "M", D(10, 10)))
        with self.assertRaises(DepartureRejected):
            flaky_coord.commit(plan)
        # The joint recheck failed after both gates staged; route and slot
        # must both be cleanly returned.
        self.assertEqual(taxi.occupancy("e1", Window(D(9), D(12))), ())
        with self.assertRaises(KeyError):
            flaky.slot_for("S")

    def test_no_false_intersection_conflict_when_routes_are_separated(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)
        first = coord.prepare("A", "stand", "hold", D(10), request("A", "M", D(10, 10)))
        coord.commit(first)
        # F2 reaches X well after A has cleared both X and the holding point.
        second = coord.prepare("B", "other", "hold", D(10, 10), request("B", "M", D(10, 20)))
        clearance = coord.commit(second)
        self.assertEqual(clearance.flight_id, "B")

    def test_commit_is_idempotent_per_flight(self):
        taxi, mgr = scheduler(), slot_manager()
        coord = DepartureCoordinator(taxi, mgr)
        plan = coord.prepare("F9", "stand", "hold", D(10), request("F9", "M", D(10, 10)))
        first = coord.commit(plan)
        second = coord.commit(plan)
        self.assertIs(first, second)


class DepartureEventBoundaryTests(unittest.TestCase):
    def test_event_published_only_after_domain_action_succeeds(self):
        control = RampWorkflowControl()

        def boom():
            raise RuntimeError("joint revalidation failed")

        with self.assertRaises(RuntimeError):
            control.commit_departure_movement("op-1", "F1", D(10), boom)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_unmet_requirement_blocks_event_and_action(self):
        control = RampWorkflowControl()
        calls = []
        denied = Decision(False, "slot-stale", "runway sequence changed")
        from rampflow.workflows.control import WorkflowRejected
        with self.assertRaises(WorkflowRejected):
            control.commit_departure_movement(
                "op-x", "F1", D(10), lambda: calls.append("ok"),
                requirements=(denied,),
            )
        self.assertEqual(calls, [])
        self.assertEqual(control.stream.head, 0)

    def test_successful_action_publishes_once_and_is_idempotent(self):
        control = RampWorkflowControl()
        calls = []
        receipt = control.commit_departure_movement(
            "op-2", "F2", D(10), lambda: calls.append("ok") or "cleared"
        )
        again = control.commit_departure_movement(
            "op-2", "F2", D(10), lambda: calls.append("again") or "cleared"
        )
        self.assertIs(receipt, again)
        self.assertEqual(calls, ["ok"])
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(receipt.event.topic, "departure.movement.committed")


if __name__ == "__main__":
    unittest.main()
