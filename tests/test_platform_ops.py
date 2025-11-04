import asyncio
import pytest

import src.platform_ops as po
import httpx
from types import SimpleNamespace


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


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text_data=None):
        self.status_code = status_code
        self._json = json_data
        self.text = text_data or ''

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _FakeClient:
    def __init__(self, response: _FakeResponse = None, raise_request: Exception = None):
        self._response = response or _FakeResponse(status_code=200, json_data={})
        self._raise = raise_request

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, params=None, json=None, headers=None):
        if self._raise:
            raise self._raise
        return self._response


def _patch_async_client(monkeypatch, response: _FakeResponse = None, raise_request: Exception = None):
    def _fake_client(*, timeout=None):
        return _FakeClient(response=response, raise_request=raise_request)

    monkeypatch.setattr(httpx, 'AsyncClient', _fake_client)


@pytest.mark.asyncio
async def test__request_success_and_json(monkeypatch):
    resp = _FakeResponse(status_code=201, json_data={'ok': True})
    _patch_async_client(monkeypatch, response=resp)

    r = await po._request('GET', 'https://example.com')
    assert r['ok'] is True
    assert r['status'] == 201
    assert r['data'] == {'ok': True}


@pytest.mark.asyncio
async def test__request_http_error_with_json(monkeypatch):
    resp = _FakeResponse(status_code=400, json_data={'error': 'bad request'})
    _patch_async_client(monkeypatch, response=resp)

    r = await po._request('GET', 'https://example.com')
    assert r['ok'] is False
    assert r['status'] == 400
    assert 'bad request' in r['error']


@pytest.mark.asyncio
async def test__request_http_error_non_json(monkeypatch):
    resp = _FakeResponse(status_code=500, json_data=ValueError('not json'), text_data='internal error')
    _patch_async_client(monkeypatch, response=resp)

    r = await po._request('GET', 'https://example.com')
    assert r['ok'] is False
    assert r['status'] == 500
    assert 'internal error' in r['error']


@pytest.mark.asyncio
async def test__request_request_exception(monkeypatch):
    _patch_async_client(monkeypatch, raise_request=httpx.RequestError('timeout'))

    r = await po._request('GET', 'https://example.com')
    assert r['ok'] is False
    assert r['status'] is None
    assert 'timeout' in r['error']


@pytest.mark.asyncio
async def test_wrapper_helpers_map_params_and_return(monkeypatch):
    # ensure each wrapper passes params correctly and returns the wrapped result
    async def fake_request(method, url, params=None, json=None, api_key=None, headers=None, timeout=20):
        return {'ok': True, 'status': 200, 'data': {'method': method, 'url': url, 'params': params, 'json': json}}

    monkeypatch.setattr(po, '_request', fake_request)

    r = await po.get_market('MKT-1')
    assert r['ok'] is True
    assert 'markets/MKT-1' in r['data']['url']

    r2 = await po.get_market_candles('MKT-1', start_ts=1, end_ts=2, fidelity=60)
    assert r2['ok'] is True
    assert r2['data']['params']['startTs'] == 1
    assert r2['data']['params']['endTs'] == 2

    r3 = await po.place_order({'foo': 'bar'})
    assert r3['ok'] is True
    assert r3['data']['json'] == {'foo': 'bar'}


@pytest.mark.asyncio
async def test_remaining_wrappers_and_headers(monkeypatch, tmp_path, monkeypatching=None):
    # Use a fake _request to capture the inputs for each wrapper
    captured = {}

    async def fake_request(method, url, params=None, json=None, api_key=None, headers=None, timeout=20):
        key = url.split('/')[-1] if '/' in url else url
        captured[url] = {'method': method, 'params': params, 'json': json, 'api_key': api_key, 'headers': headers}
        return {'ok': True, 'status': 200, 'data': {'url': url, 'params': params, 'json': json}}

    monkeypatch.setattr(po, '_request', fake_request)

    # get_event
    r = await po.get_event('EVT-1')
    assert r['ok'] is True
    assert 'events/EVT-1' in r['data']['url']

    # get_market_trades should include limit and offset
    r2 = await po.get_market_trades('MKT-XYZ', limit=5, offset=42)
    assert r2['ok'] is True
    assert r2['data']['params']['limit'] == 5
    assert r2['data']['params']['offset'] == 42

    # cancel_order should hit cancel endpoint
    r3 = await po.cancel_order('ORDER-123')
    assert r3['ok'] is True
    assert 'orders/ORDER-123/cancel' in r3['data']['url']

    # get_order_status should request the order URL
    r4 = await po.get_order_status('ORDER-123')
    assert r4['ok'] is True
    assert 'orders/ORDER-123' in r4['data']['url']

    # positions & portfolio
    r5 = await po.get_positions()
    assert r5['ok'] is True
    assert 'positions' in r5['data']['url'] or 'positions' in list(captured.keys())[-1]

    r6 = await po.get_portfolio()
    assert r6['ok'] is True
    assert 'portfolio' in r6['data']['url'] or 'portfolio' in list(captured.keys())[-1]


def test__build_headers_with_env_and_extra(monkeypatch):
    # Ensure authorization headers are built when env var present
    monkeypatch.setenv('KALSHI_API_KEY', 'testkey')
    h = po._build_headers()
    assert 'Authorization' in h
    assert h['X-API-Key'] == 'testkey'

    # extra headers merge
    h2 = po._build_headers(api_key='explicit', extra={'X-Foo': 'Bar'})
    assert h2['X-API-Key'] == 'explicit'
    assert h2['X-Foo'] == 'Bar'

