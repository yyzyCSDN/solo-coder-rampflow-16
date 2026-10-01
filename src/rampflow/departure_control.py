from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from .models import Decision
from .runway_slots import RunwaySlotManager, RunwayUse, SlotRequest
from .taxi import EdgeUse, TaxiPlan, TaxiScheduler
from .timeutil import Window


@dataclass(frozen=True)
class DeparturePlan:
    """A read-only proposal spanning both departure gates."""
    flight_id: str
    taxi_plan: TaxiPlan
    slot_request: SlotRequest
    runway_use: RunwayUse


@dataclass(frozen=True)
class DepartureClearance:
    """A departure that survived the joint surface/runway revalidation."""
    flight_id: str
    taxi_plan: TaxiPlan
    runway_use: RunwayUse


class DepartureRejected(RuntimeError):
    """Raised when the joint clearance fails; all staged resources are released."""

    def __init__(self, decisions: tuple[Decision, ...]):
        self.decisions = decisions
        summary = "; ".join(f"{item.code}: {item.reason}" for item in decisions)
        super().__init__(summary)


class DepartureCoordinator:
    """Connects the taxi-route gate and the runway-slot gate into one clearance.

    Each gate still applies its own concurrency rules when committing, but a
    clearance is only granted after *both* gates revalidate together.  A route
    submitted at an intersection after the first gate, or a runway sequence
    change, therefore rejects the whole departure.  Any route or slot staged
    before the failed joint revalidation is released cleanly.
    """

    def __init__(self, taxi: TaxiScheduler, slots: RunwaySlotManager):
        self.taxi = taxi
        self.slots = slots
        self._clearances: dict[str, DepartureClearance] = {}

    @property
    def clearances(self) -> Mapping[str, DepartureClearance]:
        return dict(self._clearances)

    def prepare(
        self,
        flight_id: str,
        start_node: str,
        runway_node: str,
        start_time: datetime,
        slot_request: SlotRequest,
        aircraft_code: str = "C",
        wind_by_runway: Mapping[str, tuple[float, float]] | None = None,
    ) -> DeparturePlan:
        if slot_request.flight_id != flight_id:
            raise ValueError("slot request belongs to a different flight")
        taxi_plan = self.taxi.plan(flight_id, start_node, runway_node, start_time, aircraft_code)
        runway_use = self.slots.find(slot_request, wind_by_runway)
        return DeparturePlan(flight_id, taxi_plan, slot_request, runway_use)

    def _reject(self, decisions: tuple[Decision, ...]) -> None:
        raise DepartureRejected(tuple(d for d in decisions if not d.accepted))

    def _node_presence(self, uses: tuple[EdgeUse, ...], margin: timedelta) -> dict[str, Window]:
        """Presence window at every traversed node.

        A flight enters a node when its incoming edge is cleared (window end)
        and remains present at an intermediate node until its outgoing edge
        is cleared as well.  Windows are widened by the scheduler's surface
        separation so close convoys count as a meeting.
        """
        ordered = sorted(uses, key=lambda u: u.window.start)
        presence: dict[str, Window] = {}
        if not ordered:
            return presence
        first = ordered[0]
        presence[first.direction[0]] = Window(first.window.start - margin, first.window.start + margin)
        for index, use in enumerate(ordered):
            node = use.direction[1]
            cleared = ordered[index + 1].window.end if index + 1 < len(ordered) else use.window.end
            presence[node] = Window(use.window.end - margin, cleared + margin)
        return presence

    def _intersection_recheck(self, plan: TaxiPlan) -> Decision:
        """Joint surface rule: two cleared routes may not meet at one node.

        Individual edge/crossing reservations can each be valid while two
        aircraft converge on the same intersection from different edges at
        the same time.  That hazard is only visible when both gates are
        revalidated together.
        """
        margin = timedelta(seconds=self.taxi.separation_seconds)
        mine = self._node_presence(plan.uses, margin)
        other_uses: list[EdgeUse] = [
            use
            for items in self.taxi._uses.values()
            for use in items
            if use.flight_id != plan.flight_id
        ]
        by_flight: dict[str, list[EdgeUse]] = {}
        for use in other_uses:
            by_flight.setdefault(use.flight_id, []).append(use)
        for other_id, items in by_flight.items():
            theirs = self._node_presence(tuple(sorted(items, key=lambda u: u.window.start)), margin)
            for node, window in mine.items():
                other_window = theirs.get(node)
                if other_window is not None and window.overlaps(other_window):
                    return Decision(
                        False,
                        "intersection-conflict",
                        "another cleared route converges on the same intersection",
                        {"node": node, "flight": other_id},
                    )
        return Decision(True, "ok", "intersections clear", {})

    def _stage(self, plan: DeparturePlan) -> None:
        """Pass both gates in order, compensating the first if the second fails."""
        # Gate 1: surface route. Nothing is staged yet, so a rejection here
        # leaves no resources to return.
        taxi_decision = self.taxi.commit(plan.taxi_plan)
        if not taxi_decision.accepted:
            self._reject((taxi_decision,))

        # Gate 2: runway sequence. If it fails after the route was reserved,
        # the route must be given back immediately.
        slot_decision = self.slots.commit(plan.runway_use)
        if not slot_decision.accepted:
            self.taxi.release(plan.flight_id)
            self._reject((slot_decision,))

    def _finalize(self, plan: DeparturePlan) -> DepartureClearance:
        """Joint revalidation of both staged gates, with full compensation."""
        committed_use = self.slots.slot_for(plan.flight_id)

        # Passing each gate separately is not enough.  Another aircraft may
        # have committed a route converging on the same intersection, or the
        # runway sequence may have moved, after the earlier individual checks.
        rechecks = (
            self.taxi.verify(plan.taxi_plan),
            self._intersection_recheck(plan.taxi_plan),
            self.slots.verify(committed_use),
        )
        if not all(item.accepted for item in rechecks):
            self.release(plan.flight_id)
            self._reject(rechecks)

        clearance = DepartureClearance(plan.flight_id, plan.taxi_plan, committed_use)
        self._clearances[plan.flight_id] = clearance
        return clearance

    def commit(self, plan: DeparturePlan) -> DepartureClearance:
        existing = self._clearances.get(plan.flight_id)
        if existing is not None:
            return existing
        self._stage(plan)
        return self._finalize(plan)

    def release(self, flight_id: str) -> None:
        """Return both the taxi route and the runway slot for one flight."""
        self.taxi.release(flight_id)
        self.slots.release(flight_id)
        self._clearances.pop(flight_id, None)
