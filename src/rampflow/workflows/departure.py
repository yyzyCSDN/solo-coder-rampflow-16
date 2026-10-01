from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping

from rampflow.event_stream import EventStream
from rampflow.models import Decision
from rampflow.runway_slots import RunwaySlotManager, RunwayUse
from rampflow.taxi import TaxiPlan, TaxiScheduler
from rampflow.timeutil import utc

from .control import OperationReceipt, RampWorkflowControl


@dataclass(frozen=True)
class DepartureClearance:
    operation_id: str
    flight_id: str
    runway_use: RunwayUse
    taxi_plan: TaxiPlan


class DepartureRejected(RuntimeError):
    """Raised when the joined departure gates cannot both be satisfied.

    Any runway slot or taxi route taken during the attempt has already been
    released before this exception propagates.
    """

    def __init__(self, decisions: tuple[Decision, ...]):
        self.decisions = decisions
        summary = "; ".join(f"{item.code}: {item.reason}" for item in decisions)
        super().__init__(summary)


class DepartureClearanceCoordinator:
    """Coordinates the runway-slot gate and the taxi-route gate for departures.

    Each domain service validates on its own, but a departure is only safe when
    *both* reservations survive a final recheck together.  Resources are taken
    in a defined order, and a failure of the final joint revalidation returns
    every resource that was taken earlier so nothing stays occupied.
    """

    def __init__(
        self,
        runway_slots: RunwaySlotManager,
        taxi: TaxiScheduler,
        control: RampWorkflowControl | None = None,
        stream: EventStream | None = None,
    ):
        self.runway_slots = runway_slots
        self.taxi = taxi
        self.control = control or RampWorkflowControl(stream)
        self._clearances: dict[str, DepartureClearance] = {}

    @property
    def clearances(self) -> Mapping[str, DepartureClearance]:
        return dict(self._clearances)

    def _reject(self, decisions: tuple[Decision, ...]) -> None:
        raise DepartureRejected(decisions)

    def _joint_revalidate(
        self,
        runway_use: RunwayUse,
        taxi_plan: TaxiPlan,
        runway_threshold_node: str | None = None,
        latest_taxi_arrival_slack_seconds: int = 0,
    ) -> tuple[Decision, ...]:
        decisions: list[Decision] = []
        slot_decision = self.runway_slots.verify(runway_use)
        if not slot_decision.accepted:
            decisions.append(slot_decision)
        taxi_decision = self.taxi.verify(taxi_plan)
        if not taxi_decision.accepted:
            decisions.append(taxi_decision)
        if runway_threshold_node is not None and taxi_plan.nodes[-1] != runway_threshold_node:
            decisions.append(
                Decision(
                    False,
                    "route-runway-mismatch",
                    "taxi route does not end at the cleared runway threshold",
                    {"expected": runway_threshold_node, "actual": taxi_plan.nodes[-1]},
                )
            )
        if slot_decision.accepted and latest_taxi_arrival_slack_seconds >= 0 and taxi_plan.end > runway_use.slot + timedelta(seconds=latest_taxi_arrival_slack_seconds):
            decisions.append(
                Decision(
                    False,
                    "arrives-after-slot",
                    "taxi route reaches the threshold after the runway slot",
                    {"slot": runway_use.slot.isoformat(), "arrival": taxi_plan.end.isoformat()},
                )
            )
        return tuple(decisions)

    def clear(
        self,
        operation_id: str,
        runway_use: RunwayUse,
        taxi_plan: TaxiPlan,
        at,
        runway_threshold_node: str | None = None,
        latest_taxi_arrival_slack_seconds: int = 0,
        evidence: Mapping[str, object] | None = None,
    ) -> OperationReceipt:
        """Take both gates atomically and advertise the departure once.

        Order of operations:
        1. Idempotency: replaying an operation returns its original receipt.
        2. Commit the runway slot; if stale, nothing was taken.
        3. Commit the taxi route; on failure the slot is released.
        4. Final joint revalidation; on failure both the route and the slot are
           released cleanly before the exception propagates.
        5. Only then is the outward ``departure.movement.committed`` event
           published.
        """
        previous = self._clearances.get(operation_id)
        if previous is not None:
            receipt = self.control.receipts[operation_id]
            if receipt.subject != previous.flight_id:
                raise ValueError("operation id reused for a different departure")
            return receipt

        if taxi_plan.flight_id != runway_use.flight_id:
            self._reject(
                (
                    Decision(
                        False,
                        "flight-mismatch",
                        "slot and taxi route belong to different flights",
                        {"slot": runway_use.flight_id, "route": taxi_plan.flight_id},
                    ),
                ),
            )

        slot_taken = False
        route_taken = False
        slot_decision = self.runway_slots.commit(runway_use)
        if not slot_decision.accepted:
            self._reject((slot_decision,))
        slot_taken = True

        route_decision = self.taxi.commit(taxi_plan)
        if not route_decision.accepted:
            self.runway_slots.release(runway_use)
            self._reject((route_decision,))
        route_taken = True

        failures = self._joint_revalidate(
            runway_use,
            taxi_plan,
            runway_threshold_node=runway_threshold_node,
            latest_taxi_arrival_slack_seconds=latest_taxi_arrival_slack_seconds,
        )
        if failures:
            # Final revalidation failed: return everything that was taken.
            if route_taken:
                self.taxi.release(taxi_plan.flight_id)
            if slot_taken:
                self.runway_slots.release(runway_use)
            self._reject(failures)

        clearance = DepartureClearance(operation_id, runway_use.flight_id, runway_use, taxi_plan)

        def _record():
            self._clearances[operation_id] = clearance
            return clearance

        payload = {
            "runway": runway_use.runway_id,
            "slot": runway_use.slot.isoformat(),
            "route_end": taxi_plan.nodes[-1],
            **dict(evidence or {}),
        }
        receipt = self.control.commit_departure_movement(
            operation_id,
            runway_use.flight_id,
            utc(at),
            _record,
            requirements=(),
        )
        return receipt

    def cancel(self, operation_id: str) -> DepartureClearance:
        """Release a previously granted departure and forget its resources."""
        clearance = self._clearances.pop(operation_id)
        self.taxi.release(clearance.flight_id)
        self.runway_slots.release(clearance.runway_use)
        return clearance
