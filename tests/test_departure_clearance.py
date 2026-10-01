import unittest
from datetime import datetime, timedelta, timezone

from rampflow.runway_slots import Runway, RunwaySlotManager, SlotRequest
from rampflow.taxi import TaxiEdge, TaxiGraph, TaxiNode, TaxiScheduler, EdgeUse
from rampflow.timeutil import Window, utc
from rampflow.workflows import (
    DepartureClearanceCoordinator,
    DepartureRejected,
    RampWorkflowControl,
)

UTC = timezone.utc


def D(h, m=0, s=0):
    return datetime(2026, 1, 1, h, m, s, tzinfo=UTC)


def _runway():
    return Runway("09L", frozenset({"09"}), 15, 25, 60)


def _graph():
    nodes = [TaxiNode("S1"), TaxiNode("X"), TaxiNode("T09L"), TaxiNode("S2"), TaxiNode("Z")]
    edges = [
        TaxiEdge("e1", "S1", "X", 100, 10),
        TaxiEdge("e2", "X", "T09L", 100, 10),
        TaxiEdge("e3", "S2", "X", 100, 10),
        TaxiEdge("e4", "X", "Z", 50, 10),
    ]
    return TaxiGraph(nodes, edges)


class WakeSeparationOrderTests(unittest.TestCase):
    def test_separation_is_directional_by_leader_follower_order(self):
        manager = RunwaySlotManager([_runway()], {("H", "M"): 120, ("M", "H"): 90})
        self.assertEqual(manager._sep("H", "M"), 120)
        self.assertEqual(manager._sep("M", "H"), 90)

    def test_heavy_then_medium_needs_longer_gap_than_medium_then_heavy(self):
        from rampflow.runway_slots import RunwayUse
        from rampflow.timeutil import Window

        def commit_follower_at(first_wake, follower_wake, at):
            manager = RunwaySlotManager([_runway()], {("H", "M"): 120, ("M", "H"): 90})
            leader_req = SlotRequest("L1", "departure", first_wake, D(10), D(10), D(10, 30), ("09L",))
            manager.commit(manager.find(leader_req))
            candidate = RunwayUse(
                "F1", "09L", "departure", follower_wake, at,
                Window(at, at + timedelta(seconds=60)), 0,
            )
            return manager.commit(candidate).accepted

        # 90 seconds after a heavy leader is below the 120s required for a medium
        self.assertFalse(commit_follower_at("H", "M", D(10, 1, 30)))
        # a heavy follower 90s after a medium leader satisfies the 90s requirement
        self.assertTrue(commit_follower_at("M", "H", D(10, 1, 30)))


class DepartureClearanceTests(unittest.TestCase):
    def setUp(self):
        self.runway = _runway()
        self.manager = RunwaySlotManager([self.runway], {("H", "M"): 120, ("M", "H"): 90})
        self.taxi = TaxiScheduler(_graph(), separation_seconds=25)
        self.control = RampWorkflowControl()
        self.coordinator = DepartureClearanceCoordinator(self.manager, self.taxi, self.control)

    def _departure(self, flight, wake, start_node, target, taxi_start, op_id, at):
        request = SlotRequest(flight, "departure", wake, target, D(10), D(10, 30), ("09L",))
        use = self.manager.find(request)
        plan = self.taxi.plan(flight, start_node, "T09L", taxi_start)
        receipt = self.coordinator.clear(
            op_id, use, plan, at, runway_threshold_node="T09L"
        )
        return receipt, use, plan

    def test_both_gates_pass_together_and_event_publishes_once(self):
        receipt, use, plan = self._departure("A1", "H", "S1", D(10), D(9, 50), "op-1", D(9, 55))
        self.assertEqual(receipt.event.topic, "departure.movement.committed")
        self.assertEqual(self.control.stream.head, 1)
        self.assertTrue(self.manager.verify(use).accepted)
        self.assertTrue(self.taxi.verify(plan).accepted)
        # replaying the operation is idempotent: no new event, no double commit
        again = self.coordinator.clear("op-1", use, plan, D(9, 55), runway_threshold_node="T09L")
        self.assertIs(again, receipt)
        self.assertEqual(self.control.stream.head, 1)

    def test_late_filed_intersection_route_fails_final_revalidation_and_releases_both(self):
        _, _, _ = self._departure("A1", "H", "S1", D(10), D(9, 50), "op-1", D(9, 55))
        request = SlotRequest("B1", "departure", "M", D(10, 3), D(10), D(10, 30), ("09L",))
        use = self.manager.find(request)
        plan = self.taxi.plan("B1", "S2", "T09L", D(9, 53))

        # A competing route is filed on the shared edge after B planned its route
        shared = next(item for item in plan.uses if item.edge_id == "e2")
        intruder = EdgeUse(
            "C9",
            "e2",
            Window(shared.window.start + timedelta(seconds=5), shared.window.start + timedelta(seconds=15)),
            ("X", "T09L"),
        )
        self.taxi._uses.setdefault("e2", []).append(intruder)

        with self.assertRaises(DepartureRejected) as ctx:
            self.coordinator.clear("op-2", use, plan, D(9, 56), runway_threshold_node="T09L")
        self.assertEqual([d.code for d in ctx.exception.decisions], ["taxi-conflict"])

        # both the slot taken first and the route taken second are cleanly returned
        self.assertFalse(self.manager.verify(use).accepted)
        self.assertEqual(self.taxi.release("B1"), 0)
        # no outward event for the rejected departure
        self.assertEqual(self.control.stream.head, 1)

    def test_stale_slot_releases_taxi_route(self):
        _, _, _ = self._departure("A1", "H", "S1", D(10), D(9, 50), "op-1", D(9, 55))
        request = SlotRequest("B1", "departure", "M", D(10, 3), D(10), D(10, 30), ("09L",))
        use = self.manager.find(request)
        plan = self.taxi.plan("B1", "S2", "T09L", D(9, 53))
        # runway sequence changes between find and clear
        steal = SlotRequest("X1", "departure", "M", use.slot, use.slot, use.slot, ("09L",))
        self.manager.commit(self.manager.find(steal))

        with self.assertRaises(DepartureRejected) as ctx:
            self.coordinator.clear("op-2", use, plan, D(9, 56), runway_threshold_node="T09L")
        self.assertEqual([d.code for d in ctx.exception.decisions], ["slot-stale"])
        self.assertEqual(self.taxi.release("B1"), 0)

    def test_route_to_wrong_threshold_is_rejected_and_rolled_back(self):
        request = SlotRequest("D1", "departure", "H", D(10), D(10), D(10, 30), ("09L",))
        use = self.manager.find(request)
        plan = self.taxi.plan("D1", "S1", "Z", D(9, 50))
        with self.assertRaises(DepartureRejected) as ctx:
            self.coordinator.clear("op-9", use, plan, D(9, 55), runway_threshold_node="T09L")
        self.assertEqual([d.code for d in ctx.exception.decisions], ["route-runway-mismatch"])
        self.assertFalse(self.manager.verify(use).accepted)
        self.assertEqual(self.taxi.release("D1"), 0)
        self.assertEqual(self.control.stream.head, 0)


if __name__ == "__main__":
    unittest.main()
