import asyncio
import pytest

import src.platform_ops as po


def test_helpers_exported():
    # ensure the async helpers are present (no network calls)
    assert hasattr(po, 'get_events')
    assert hasattr(po, 'get_market')
    assert hasattr(po, 'get_market_candles')
    assert hasattr(po, 'place_order')


@pytest.mark.asyncio
async def test_get_events_no_network(monkeypatch):
    # monkeypatch _request to avoid real network
    async def fake_request(method, url, params=None, json=None, api_key=None, headers=None, timeout=20):
        return {'ok': True, 'status': 200, 'data': {'events': []}}

    monkeypatch.setattr(po, '_request', fake_request)
    res = await po.get_events(limit=1)
    assert res['ok'] is True
    assert isinstance(res['data'], dict)
