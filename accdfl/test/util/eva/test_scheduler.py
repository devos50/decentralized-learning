import asyncio
from typing import Iterable
from unittest.mock import MagicMock, Mock

import pytest

from accdfl.util.eva.scheduler import Scheduler
from accdfl.util.eva.settings import EVASettings
from accdfl.util.eva.transfer.base import Transfer


# pylint: disable=redefined-outer-name, protected-access


@pytest.fixture
async def scheduler():
    eva = Mock(
        incoming={},
        outgoing={},
        settings=EVASettings(),
        shutting_down=False
    )
    scheduler = Scheduler(eva=eva)
    yield scheduler

    await scheduler.shutdown()


class MockTransfer(MagicMock):
    def start(self):
        self.container[self.peer] = self


@pytest.mark.asyncio
async def test_can_be_send_immediately(scheduler: Scheduler):
    scheduler = yield scheduler
    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {}
    assert scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.incoming))
    assert scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.outgoing))

    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {'peer': Mock()}
    assert scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.incoming))
    assert not scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.outgoing))

    scheduler.eva.incoming = {'peer': Mock()}
    scheduler.eva.outgoing = {}
    assert not scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.incoming))
    assert scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.outgoing))

    scheduler.eva.settings.max_simultaneous_transfers = 2
    scheduler.eva.incoming = {'any peer': MockTransfer()}
    scheduler.eva.outgoing = {'any peer': MockTransfer()}
    assert not scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.incoming))
    assert not scheduler.can_be_send_immediately(MockTransfer(peer='peer', container=scheduler.eva.outgoing))


@pytest.mark.asyncio
async def test_is_simultaneously_served_transfers_limit_not_exceeded(scheduler: Scheduler):
    scheduler = yield scheduler
    # In this test we will try to exceed `max_simultaneous_transfers` limit.
    scheduler.eva.settings.max_simultaneous_transfers = 2

    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {}
    assert not scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {'peer1': MockTransfer()}
    scheduler.eva.outgoing = {}
    assert not scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {'peer1': MockTransfer()}
    assert not scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {'peer1': MockTransfer()}
    scheduler.eva.outgoing = {'peer1': MockTransfer()}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {'peer1': MockTransfer(), 'peer2': MockTransfer()}
    scheduler.eva.outgoing = {}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {'peer1': MockTransfer(), 'peer2': MockTransfer()}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()


@pytest.mark.asyncio
async def test_is_simultaneously_served_transfers_limit_exceeded(scheduler: Scheduler):
    scheduler = yield scheduler
    # In this test we will try to exceed `max_simultaneous_transfers` limit.
    scheduler.eva.settings.max_simultaneous_transfers = 2

    scheduler.eva.incoming = {'peer1': MockTransfer()}
    scheduler.eva.outgoing = {'peer1': MockTransfer()}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {'peer1': MockTransfer(), 'peer2': MockTransfer()}
    scheduler.eva.outgoing = {}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()

    scheduler.eva.incoming = {}
    scheduler.eva.outgoing = {'peer1': MockTransfer(), 'peer2': MockTransfer()}
    assert scheduler._is_simultaneously_served_transfers_limit_exceeded()


@pytest.mark.asyncio
async def test_schedule(scheduler: Scheduler):
    scheduler = yield scheduler
    scheduler.can_be_send_immediately = Mock(return_value=False)

    scheduler.schedule(transfer=MockTransfer())

    assert not scheduler.eva.start_transfer.called
    assert len(scheduler.scheduled) == 1


@pytest.mark.asyncio
async def test_start_transfer(scheduler: Scheduler):
    scheduler = yield scheduler
    scheduler.can_be_send_immediately = Mock(return_value=True)
    transfer = MockTransfer()
    transfer.start = Mock(wraps=transfer.start)

    scheduler.schedule(transfer)

    assert transfer.start.called
    assert not scheduler.scheduled


@pytest.mark.asyncio
async def _fill_test_data(scheduler: Scheduler):
    scheduler = yield scheduler
    scheduler.eva.incoming = {'peer1': MockTransfer()}
    scheduler.eva.outgoing = {'peer2': MockTransfer()}

    transfers = (
        MockTransfer(peer='peer1', container=scheduler.eva.incoming),
        MockTransfer(peer='peer1', container=scheduler.eva.outgoing),
        MockTransfer(peer='peer2', container=scheduler.eva.incoming),
        MockTransfer(peer='peer2', container=scheduler.eva.outgoing),
        MockTransfer(peer='peer3', container=scheduler.eva.incoming),
        MockTransfer(peer='peer3', container=scheduler.eva.outgoing)
    )

    scheduler.scheduled = dict.fromkeys(transfers)


def _transform_to_str(eva, transfers: Iterable[Transfer]) -> Iterable[str]:
    """Function transforms transfers to strings like 'peer1_out'"""
    for transfer in transfers:
        container = "in" if transfer.container == eva.incoming else "out"
        yield f'{transfer.peer}_{container}'


@pytest.mark.asyncio
async def test_transfers_that_can_be_send(scheduler: Scheduler):
    scheduler = yield scheduler
    _fill_test_data(scheduler)

    # Regarding the test data, all scheduled transfer should be ready to send
    # except transfers for `peer1` and `peer2` because they already exists in
    # incoming and outgoing containers
    ready_to_send = scheduler._transfers_that_can_be_send()
    str_representation = list(_transform_to_str(scheduler.eva, ready_to_send))
    assert str_representation == ['peer1_out', 'peer2_in', 'peer3_in', 'peer3_out']


@pytest.mark.asyncio
async def test_send_scheduled(scheduler: Scheduler):
    scheduler = yield scheduler
    _fill_test_data(scheduler)

    # Regarding the test data, all scheduled transfer should be ready to send
    # except transfers for `peer1` and `peer2` because they already exists in
    # incoming and outgoing containers
    #
    # In this test we will limit amount of `max_simultaneous_transfers` by 4
    # transfers. In this case only two transfers from scheduled should be send.
    scheduler.eva.shutting_down = False
    scheduler.eva.settings.max_simultaneous_transfers = 4
    started = scheduler.send_scheduled()
    str_representation = list(_transform_to_str(scheduler.eva, started))
    assert str_representation == ['peer1_out', 'peer2_in']


@pytest.mark.asyncio
async def test_shutdown():
    scheduler = Scheduler(eva=Mock(settings=EVASettings(scheduled_send_interval=0.1)))

    await scheduler.shutdown()
    scheduler.send_scheduled = Mock()

    # wait more than `scheduled_send_interval`
    await asyncio.sleep(0.3)

    # ensure that send_scheduled has not been called
    assert not scheduler.send_scheduled.called
